from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch_geometric.utils import to_dense_batch
from torch_sparse import matmul
from torch.nn import LeakyReLU
#######################################
import nn
#######################################

class GNNs(torch.nn.Module):
    def __init__(self, args):
        super(GNNs, self).__init__()
        self.args = args
        self.layer_num = args.layer_num
        if args.act_func == 'ReLU':
            self.act_func = getattr(torch.nn, args.act_func)()
        else:
            self.act_func = getattr(torch.nn, args.act_func)(0.2)

        kwargs = [{}] * self.layer_num
        modelname = '{}Conv'.format(args.model)
        if args.model == 'GAT':
            kwargs = [{'n_heads': args.num_heads, 'act_func': self.act_func}] * self.layer_num
        if args.model in ['GIN', 'SAGE', 'GCN']:
            kwargs = [{'learn_eps': args.learn_eps, 'act_func': self.act_func}] * self.layer_num
            modelname = 'GINConv'


        GNNConv = getattr(nn, modelname)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.bns_edge = torch.nn.ModuleList()
        self.bond_enc = torch.nn.ModuleList([])
        self.atom_enc = AtomEncoder(args.hidden_dim - (args.dim_pe if args.posenc_LapPE else 0))
        self.bns_in = torch.nn.BatchNorm1d(args.hidden_dim)
        self.drop_func = torch.nn.Dropout(args.drop) if args.drop != 0. else torch.nn.Identity()

        for i in range(self.layer_num):
            self.convs.append(GNNConv(args.hidden_dim, args.hidden_dim, edge_dim = args.edge_dim, **kwargs[i]))
            self.bns.append(torch.nn.BatchNorm1d(args.hidden_dim))
            self.bns_edge.append(torch.nn.BatchNorm1d(args.edge_dim))
            self.bond_enc.append(BondEncoder(args.edge_dim))

        if args.virtual:
            self.vn_emb = torch.nn.Embedding(1, args.hidden_dim)
            torch.nn.init.constant_(self.vn_emb.weight.data, 0)

            self.mlp_vn = torch.nn.ModuleList([])
            for layer in range(self.layer_num - 1):
                self.mlp_vn.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
                        self.act_func,
                        torch.nn.BatchNorm1d(args.hidden_dim),
                        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
                        self.act_func,
                        torch.nn.BatchNorm1d(args.hidden_dim),
                    )
                )

        self.fcl = torch.nn.Linear(args.hidden_dim, args.num_tasks)


    def forward_global(self, data):
        feat, graph, batch, edge_attr = data.x, data.graph, data.batch, data.edge_attr

        if self.args.virtual:
            vn_emb = self.vn_emb(torch.zeros(batch.size(0)).to(torch.long).to(feat.device))

        x_emb = self.atom_enc(feat)
        x = self.drop_func(self.bns_in(x_emb))
        xs = [x]
        for i in range(self.layer_num):
            if self.args.virtual:
                x += vn_emb[batch.storage.row()]
            x_in = x
            edge_emb = self.drop_func(self.bns_edge[i](self.bond_enc[i](edge_attr)))
            x = self.drop_func(self.bns[i](self.act_func(self.convs[i](x_in, graph, edge_emb))))

            if self.args.residual:
                x += xs[i]
            xs.append(x)
            if i < self.args.layer_num - 1 and self.args.virtual:
                vn_emb_temp = self.readout(xs[i], batch) + vn_emb
                if self.args.residual:
                    vn_emb = vn_emb + self.drop_func(self.mlp_vn[i](vn_emb_temp))
                else:
                    vn_emb = self.drop_func(self.mlp_vn[i](vn_emb_temp))

        if self.args.JK == 'last':
            out_x = xs[-1]
        elif self.args.JK == 'sum':
            out_x = 0
            for i in range(self.layer_num + 1):
                out_x += xs[i]
        out = self.readout(out_x, batch)
        out = self.fcl(out)
        return out, None if self.training else xs[-1]

    def forward_local(self, data):
        feat, graph, batch, edge_attr = data.x, data.graph_new, data.batch_new, data.edge_attr[data.eperm_n2o]
        graph_n2o = data.graph_n2o

        if self.args.virtual:
            vn_emb = self.vn_emb(torch.zeros(batch.size(0)).to(torch.long).to(feat.device))

        x_emb = self.atom_enc(feat)
        x_emb = self.readout(x_emb, graph_n2o, aggr='mean')
        x = self.drop_func(self.bns_in(x_emb))
        xs = [x]
        for i in range(self.layer_num):
            if self.args.virtual:
                x += self.readout(vn_emb, batch.t(), 'mean')

            edge_emb = self.drop_func(self.bns_edge[i](self.bond_enc[i](edge_attr)))
            x = self.drop_func(self.bns[i](self.act_func(self.convs[i](x, graph, edge_emb))))

            if self.args.residual:
                x += xs[i]
            xs.append(x)
            if i < self.args.layer_num - 1 and self.args.virtual:
                vn_emb_temp = self.readout(xs[i], batch) + vn_emb
                if self.args.residual:
                    vn_emb = vn_emb + self.drop_func(self.mlp_vn[i](vn_emb_temp))
                else:
                    vn_emb = self.drop_func(self.mlp_vn[i](vn_emb_temp))

        if self.args.JK == 'last':
            out_x = xs[-1]
        elif self.args.JK == 'sum':
            out_x = 0
            for i in range(self.layer_num + 1):
                out_x += (self.readout(xs[i], graph_n2o, aggr='mean') if i <= 0 else xs[i])
        out = self.readout(out_x, batch)
        out = self.fcl(out)
        return out, None if self.training else xs[-1]

    def forward(self, data):
        if self.training and self.args.local_method:
            return self.forward_local(data)
        else:
            return self.forward_global(data)

    def readout(self, x, batch, aggr=None):
        options = ['sum', 'mean', 'max']
        aggr = (self.args.readout if self.args.readout in options else 'mean') if aggr is None else aggr
        out = matmul(batch, x, reduce=aggr)
        return out
