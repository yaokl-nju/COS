import joblib
import networkx as nx
import numpy as np
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
import os
import os.path as osp
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.utils import to_scipy_sparse_matrix

from datasets.sample_func import sample_func, sample_func_origin
from datasets.parse_for_local_context import parse_for_local_context

class ogbg_dataset(PygGraphPropPredDataset):
    def __init__(self, args, name, root='dataset'):
        super(ogbg_dataset, self).__init__(name, root)
        self.args = args
        self.idx_dict = self.get_idx_split()
        self.num_graphs = self.slices['x'].size(0) - 1
        self.num_nodes_all = self.data.x.size(0)
        self.resort_edges()

        self.batch = {}
        for phase in self.idx_dict.keys():
            if phase == 'train':
                self.batch[phase] = self.idx_dict[phase].size(0) // self.args.bsize
            else:
                self.batch[phase] = int(np.ceil(self.idx_dict[phase].size(0) * 1.0 / self.args.bsize))

        print("data info:")
        print("\tnum_graphs", self.num_graphs)
        print("\tnum_train", self.idx_dict['train'].size())
        print("\tnum_valid", self.idx_dict['valid'].size())
        print("\tnum_test", self.idx_dict['test'].size())
        print("\tnum_nodes", self.data.x.size())
        print("\tnum_edges", self.data.edge_index.size())
        print("\tnum_edge_feature", self.data.edge_attr.size())
        print('\tnum nodes per graph', self.data.x.size(0) / self.num_graphs)
        print("\ty.shape", self.data.y.size())
        print("\tnum_classes", self.num_classes)
        print("\tpos num: {}, neg num: {}".format(str((self.data.y == 1).sum()), str((self.data.y == 0).sum())))
        self.num_atom_type = torch.max(self.data.x[:, 0]) + 1
        self.num_bond_type = torch.max(self.data.edge_attr[:, 0]) + 1
        print('\tnum_atom_type', self.num_atom_type)
        print('\tnum_bond_type', self.num_bond_type)


        self.load_local_context()
        self.data.x = self.data.x.to(args.device)
        self.data.y = self.data.y.to(args.device)
        self.data.edge_attr = self.data.edge_attr.to(args.device)

    def process(self):
        super(ogbg_dataset, self).process()

    def resort_edges(self):
        edges = self.data.edge_index
        num_e = self.slices['edge_index'][1:] - self.slices['edge_index'][:-1]
        shift = torch.repeat_interleave(self.slices['x'][:-1], num_e)
        edges = edges + shift.view(1, -1)
        perm = torch.argsort(edges[0] * self.num_nodes_all + edges[1])
        self.data.edge_index = self.data.edge_index[:, perm]
        self.data.edge_attr = self.data.edge_attr[perm]
    # ===============================================

    # ===============================================
    def reset_iter(self):
        self.perm_idx = {}
        for phase in self.idx_dict.keys():
            self.perm_idx[phase] = self.iter_idx(phase)
        self.ncount_old, self.ncount_new = 1, 1

    def iter_idx(self, phase):
        if phase == 'train':
            perm = torch.randperm(self.idx_dict[phase].size(0)).split(self.args.bsize)
            perm = perm if self.idx_dict[phase].size(0) % self.args.bsize == 0 else perm[:-1]
        else:
            perm = torch.arange(self.idx_dict[phase].size(0)).split(self.args.bsize)
        for ids in perm:
            yield ids
        yield None

    def get_idx(self, phase):
        index = next(self.perm_idx[phase])
        if index is None:
            self.perm_idx[phase] = self.iter_idx(phase)
            index = next(self.perm_idx[phase])
        return self.idx_dict[phase][index]

    def sample(self, phase, gids_in = None):
        gids = self.get_idx(phase) if gids_in is None else gids_in
        _, data, nids, eidx = sample_func(gids, self, phase == 'train')
        data.x = self.data.x[nids]
        data.y = self.data.y[gids]
        data.edge_attr = self.data.edge_attr[eidx]
        data.num_graphs = gids.size(0)
        if phase == 'train' and self.args.local_method:
            self.ncount_old += data.graph.size(0)
            self.ncount_new += data.graph_new.size(0)
        data = data.to(self.args.device)
        return data.to(self.args.device)

    def load_local_context(self):
        if self.args.local_method:
            mask_tr = torch.zeros(self.num_graphs, dtype=torch.bool)
            mask_tr[self.idx_dict['train']] = True
            mask_idx = torch.repeat_interleave(mask_tr, self.slices['x'][1:] - self.slices['x'][:-1])
            num_idx_tr = mask_idx.to(torch.float).sum()

            path = osp.join(self.root, 'processed/mapping_cs7.pt')
            if not os.path.exists(path):
                self.preprocess_atom_bond_type()
                g_list = self.seg_graph_to_list()
                mapping, record = parse_for_local_context(g_list, self.args.cs)
                joblib.dump((mapping, record), path, compress=3)
            mapping, record = joblib.load(path)
            for i in range(self.args.cs):
                if i == self.args.cs - 1:
                    num_newid = torch.max(mapping[i][mask_idx]) + 1 - self.data.x.size(0)
                    temp = torch.zeros(num_newid, dtype=torch.bool)
                    temp[mapping[i][mask_idx][record[i][mask_idx]] - self.data.x.size(0)] = True
                    num_newid = temp.to(torch.float).sum()
                    num_fused = record[i][mask_idx].to(torch.float).sum()
                    print('\ncontext size {}'.format(str(i + 1)))
                    print('\t{} nodes are mapped to {} new nodes.'.format(str(num_fused), str(num_newid)))
                    print('\tnum of reduced nodes', num_fused - num_newid)
                    print('\tnum of all left nodes', num_idx_tr - (num_fused - num_newid))
                    print('\tratio {:.3f}'.format((num_idx_tr - (num_fused - num_newid)) * 1.0 / num_idx_tr))

            self.mapping, self.record = mapping[self.args.cs - 1], record[self.args.cs - 1]
            self.max_newid = self.mapping.max()
            self.mapping_all, self.record_all = mapping, record

    def seg_graph_to_list(self):
        g_list = {
            'atom_type': self.atom_type,
            'bond_type': self.bond_type,
            'num_atom_type': self.num_atom_type_uni,
            'num_bond_type': self.num_bond_type_uni,
            'indptr_x': self.slices['x'],
            'indptr_e': self.slices['edge_index'],
            'edges': self.data.edge_index,
            'indptr_n': []
        }
        for i in tqdm(range(self.data.y.size(0)), desc='Seg graph'):
            _, _, _, indptr_n = sample_func_origin(torch.LongTensor([i]), self)
            g_list['indptr_n'].append(indptr_n)
        return g_list

    def preprocess_atom_bond_type(self):
        if self.args.local_method:
            atom_fdims = get_atom_feature_dims()
            bond_fdims = get_bond_feature_dims()
            base_atom, base_bond = [1], [1]
            for i in range(len(atom_fdims) - 1):
                base_atom.append(base_atom[-1] * atom_fdims[-i-1])
            for i in range(len(bond_fdims) - 1):
                base_bond.append(base_bond[-1] * bond_fdims[-i-1])
            base_atom.reverse()
            base_bond.reverse()
            print("\natom and bond type info:")
            print("\tatom dims", atom_fdims)
            print('\tbase atom multiplier', base_atom)
            print("\tpossible atom type", base_atom[0] * atom_fdims[0])
            print("\tbond dims", bond_fdims)
            print("\tpossible bond type", base_bond[0] * bond_fdims[0])
            print('\tbase bond multiplier', base_bond)

            atom_type, bond_type = 0, 0
            for i in range(self.data.x.size(1)):
                atom_type += (self.data.x[:, i] * base_atom[i])
            for i in range(self.data.edge_attr.size(1)):
                bond_type += (self.data.edge_attr[:, i] * base_bond[i])
            atom_record = torch.zeros(base_atom[0] * atom_fdims[0], dtype=torch.bool)
            bond_record = torch.zeros(base_bond[0] * bond_fdims[0], dtype=torch.bool)
            atom_record[atom_type] = True
            bond_record[bond_type] = True

            uni_atom_type = torch.where(atom_record)[0]
            uni_bond_type = torch.where(bond_record)[0]
            atom_reindex = torch.zeros(base_atom[0] * atom_fdims[0], dtype=torch.long) - 1
            bond_reindex = torch.zeros(base_bond[0] * bond_fdims[0], dtype=torch.long) - 1
            atom_reindex[uni_atom_type] = torch.arange(uni_atom_type.size(0))
            bond_reindex[uni_bond_type] = torch.arange(uni_bond_type.size(0))

            self.num_atom_type_uni = uni_atom_type.size(0)
            self.num_bond_type_uni = uni_bond_type.size(0)
            self.atom_type = atom_reindex[atom_type]
            self.bond_type = bond_reindex[bond_type]
            print("\tunique atom type", self.num_atom_type_uni)
            print("\tunique bond type", self.num_bond_type_uni)

    def construct_mapping(self, mapping, record):
        neighs_self = torch.stack([torch.arange(self.num_nodes_all), torch.arange(self.num_nodes_all)], dim=0)
        neighs, shift, mask = [], 0, None
        for i in range(7):
            # if (i + 1) % 2 == 0 and (i + 1) >= 3:
            # if i > 3:
            if i < 5 and i > 1:
                num_newid = torch.max(mapping[i]) + 1 - self.num_nodes_all
                ind = torch.where(record[i])[0]
                neighs_i = torch.stack([ind, mapping[i][ind] + shift], dim=0)
                neighs.append(neighs_i)
                shift += num_newid
                mask = record[i] if mask is None else torch.logical_or(record[i], mask)
        neighs.append(neighs_self[:, ~mask])
        neighs = torch.cat(neighs, dim=1)
        self.max_newid = shift + self.num_nodes_all - 1
        perm = torch.argsort(neighs[0] * (self.max_newid + 1) + neighs[1])
        self.neighs = neighs[:, perm]

        deg = torch.zeros(self.num_nodes_all, dtype=torch.long)
        deg.scatter_add_(0, self.neighs[0], torch.ones(self.neighs.size(1), dtype=torch.long))
        indptr = deg.new_zeros(self.num_nodes_all + 1, dtype=torch.long)
        torch.cumsum(deg, 0, out=indptr[1:])

        self.neighs_deg = deg
        self.neighs_indptr = indptr