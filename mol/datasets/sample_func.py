import time
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

def sample_func(gids, dataset, is_training):
    ###################################
    ##  sample from original graphs  ##
    ###################################
    # statistics in original graphs
    num_nodes = dataset.slices['x'][gids+1] - dataset.slices['x'][gids]
    num_edges = dataset.slices['edge_index'][gids+1] - dataset.slices['edge_index'][gids]

    ### fetch node ids & remap
    ptr_n = num_nodes.new_zeros(gids.shape[0] + 1, dtype=torch.long)
    torch.cumsum(num_nodes, 0, out=ptr_n[1:])
    nids = torch.arange(num_nodes.sum()) - torch.repeat_interleave(ptr_n[:-1], num_nodes)
    nids += torch.repeat_interleave(dataset.slices['x'][gids], num_nodes)

    ### fetch edges
    ptr_e = num_edges.new_zeros(gids.shape[0] + 1, dtype=torch.long)
    torch.cumsum(num_edges, 0, out=ptr_e[1:])
    index = torch.arange(num_edges.sum()) - torch.repeat_interleave(ptr_e[:-1], num_edges)
    index += torch.repeat_interleave(dataset.slices['edge_index'][gids], num_edges)

    edges = dataset.data.edge_index[:, index]                 # nids of edges in each graph start from 0
    shift = torch.repeat_interleave(ptr_n[:-1], num_edges)    # have been preprocessed in ogbg_dataset
    edges += shift
    eidx = index

    if dataset.args.model in ['SAGE', 'GCN']:
        ndeg_o = torch.zeros(nids.size(0))
        ndeg_i = torch.zeros(nids.size(0))
        ndeg_o.scatter_add_(0, edges[1], torch.ones(edges.size(1)))
        ndeg_o[ndeg_o == 0] = 1
        ndeg_i.scatter_add_(0, edges[0], torch.ones(edges.size(1)))
        ndeg_i[ndeg_i == 0] = 1
        if dataset.args.model == 'SAGE':
            weights = 1.0 / (ndeg_i[edges[0]] + 1e-16)
        else:
            weights = 1.0 / (ndeg_o[edges[1]] * ndeg_i[edges[0]] + 1e-16).sqrt()
    else:
        weights = torch.ones(edges.size(1))

    ### readout
    src_r = torch.repeat_interleave(torch.arange(gids.size(0)), num_nodes)
    obj_r = torch.arange(nids.size(0))
    weights_r = torch.ones(src_r.size(0))

    batch = SparseTensor(row=src_r, col=obj_r, value=weights_r, sparse_sizes=(gids.size(0), nids.size(0)), is_sorted=True)
    graph = SparseTensor(row=edges[0], col=edges[1], value=weights, sparse_sizes=(nids.size(0), nids.size(0)), is_sorted=True)

    ##########################################
    ########    process for fusion    ########
    ##########################################
    if dataset.args.local_method and is_training:
        if dataset.args.local_random:
            mapping, record = dataset.mapping[nids], dataset.record[nids]
            mapping_l, record_l = dataset.mapping_all[dataset.args.cs][nids], dataset.record_all[dataset.args.cs][nids]
            ind_s = torch.where(record)[0]
            nnz = int(ind_s.size(0) * 0.60)
            mask = torch.logical_and(record, record_l)
            ind_l = torch.where(mask)[0]
            if ind_l.size(0) > nnz:
                detect = torch.zeros(mapping_l[record_l].max() - dataset.num_nodes_all + 1)
                ind = ind_l[torch.randperm(ind_l.size(0))[:nnz]]
                detect[mapping_l[ind] - dataset.num_nodes_all] = True
                ind = ind_l[torch.where(detect[mapping_l[ind_l] - dataset.num_nodes_all])[0]]
            else:
                ind = ind_l
                # ind_r = torch.where(~mask)[0]
                ind_r = torch.where(torch.logical_and(record, ~record_l))[0]
                ind_r = ind_r[torch.randperm(ind_r.size(0))[:nnz - ind_l.size(0)]]
                record[ind_r] = False
            mapping[ind] = mapping_l[ind] - dataset.num_nodes_all + mapping[record].max() + 1
        else:
            mapping, record = dataset.mapping[nids], dataset.record[nids]

        reindex = torch.zeros(nids.size(0), dtype=torch.long) - 1
        mask = ~record
        num_cs = mask.to(torch.long).sum()
        reindex[mask] = torch.arange(num_cs)

        record_new = torch.zeros(mapping[record].max() - dataset.num_nodes_all + 1, dtype=torch.bool)
        record_new[mapping[record] - dataset.num_nodes_all] = True
        ind = torch.where(record_new)[0]
        reindex_new = torch.zeros(record_new.size(0), dtype=torch.long) - 1
        reindex_new[ind] = torch.arange(ind.size(0)) + num_cs
        reindex[record] = reindex_new[mapping[record] - dataset.num_nodes_all]

        num_newids = num_cs + ind.size(0)
        edges_new = torch.stack([reindex[edges[0]], reindex[edges[1]]], dim=0)
        rankids = edges_new[0] * num_newids + edges_new[1]
        eperm_n2o = torch.argsort(rankids.cuda()).cpu()
        edges_new = edges_new[:, eperm_n2o]

        src_n2o, obj_n2o = reindex, torch.arange(nids.size(0))
        graph_n2o = SparseTensor(row=src_n2o, col=obj_n2o, value=torch.ones(src_n2o.size(0)), sparse_sizes=(num_newids, nids.size(0)))

        deg_n2o = graph_n2o.storage.rowcount().to(torch.float)
        weights_new = (1.0 / deg_n2o)[edges_new[0]] * weights[eperm_n2o]
        graph_new = SparseTensor(row=edges_new[0], col=edges_new[1], value=weights_new, sparse_sizes=(num_newids, num_newids), is_sorted=True)

        obj_new = reindex[obj_r]
        batch_new = SparseTensor(row=src_r, col=obj_new, value=torch.ones(obj_new.size(0)), sparse_sizes=(gids.size(0), num_newids))
        data = Data(graph=graph, batch=batch, graph_new=graph_new, eperm_n2o=eperm_n2o, graph_n2o=graph_n2o, batch_new=batch_new)
    else:
        data = Data(graph=graph, batch=batch)

    return gids, data, nids, eidx


def sample_func_origin(gids, dataset):
    num_nodes = dataset.slices['x'][gids+1] - dataset.slices['x'][gids]
    num_edges = dataset.slices['edge_index'][gids+1] - dataset.slices['edge_index'][gids]

    ### fetch node ids & remap
    ptr_n = num_nodes.new_zeros(gids.shape[0] + 1, dtype=torch.long)
    torch.cumsum(num_nodes, 0, out=ptr_n[1:])
    nids = torch.arange(num_nodes.sum()) - torch.repeat_interleave(ptr_n[:-1], num_nodes)
    nids += torch.repeat_interleave(dataset.slices['x'][gids], num_nodes)

    ### fetch edges
    ptr_e = num_edges.new_zeros(gids.shape[0] + 1, dtype=torch.long)
    torch.cumsum(num_edges, 0, out=ptr_e[1:])
    index = torch.arange(num_edges.sum()) - torch.repeat_interleave(ptr_e[:-1], num_edges)
    index += torch.repeat_interleave(dataset.slices['edge_index'][gids], num_edges)

    edges = dataset.data.edge_index[:, index]                 # nids of edges in each graph start from 0
    eidx = index if 'edge_attr' in dataset.slices else None

    deg = torch.zeros(nids.size(0))
    deg.scatter_add_(0, edges[0], torch.ones(edges.size(1)))
    indptr = deg.new_zeros(nids.size(0) + 1, dtype=torch.long)
    torch.cumsum(deg, 0, out=indptr[1:])

    return edges, nids, eidx, indptr
