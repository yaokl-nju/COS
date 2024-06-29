import math
import torch
from torch_geometric.nn.inits import glorot, zeros, reset
from torch.nn import Sequential, Linear, ReLU
import torch_scatter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul

from utils.init_func import softmax
##################################
class GINConv(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 eps: float = 0.,
                 learn_eps: bool = False,
                 edge_dim = 0,
                 act_func = torch.nn.ReLU(),
                 layer_mlp = 2
                 ):
        super(GINConv, self).__init__()
        if layer_mlp == 2:
            self.nn = Sequential(
                Linear(in_dim * 2 + edge_dim, out_dim),
                act_func,
                torch.nn.BatchNorm1d(out_dim),
                Linear(out_dim, out_dim))
        else:
            self.nn = Linear(in_dim * 2 + edge_dim, out_dim)
        self.initial_eps = eps
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, graph, edge_attr=None):
        if isinstance(graph, SparseTensor):
            row, norm = graph.storage.row(), graph.storage.value()
            x_src = (graph.sum(1).view(-1, 1) * (1 + self.eps)) * x
        else:
            row, norm = graph._indices()[0], graph._values()
            deg = graph @ torch.ones_like(x[:, (0, )])
            x_src = (deg * (1 + self.eps)) * x

        x_obj = graph @ x
        if edge_attr is not None:
            x_e = scatter_add(norm.view(-1, 1) * edge_attr, row, dim=0, dim_size=graph.size(0))
            out = torch.cat([x_src, x_obj, x_e], dim=1)
        else:
            out = torch.cat([x_src, x_obj], dim=1)
        return self.nn(out)


class GATConv(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=1,
                 bias = True,
                 edge_dim = 0,
                 act_func = torch.nn.ReLU(),
                 layer_mlp = 2
                 ):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.in_dim = in_dim
        self.out_dim = out_dim // n_heads

        self.a1 = torch.nn.Linear(self.in_dim, self.n_heads, bias=False)
        self.a2 = torch.nn.Linear(self.in_dim, self.n_heads, bias=False)
        if edge_dim > 0:
            self.ae = torch.nn.Linear(edge_dim, self.n_heads, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        if layer_mlp == 2:
            self.nn = torch.nn.ModuleList()
            for i in range(self.n_heads):
                self.nn.append(Sequential(
                    Linear(in_dim * 2 + edge_dim, self.out_dim),
                    act_func,
                    torch.nn.BatchNorm1d(self.out_dim),
                    Linear(self.out_dim, self.out_dim)))
        else:
            self.nn = torch.nn.ModuleList()
            for i in range(self.n_heads):
                self.nn.append(Linear(in_dim * 2 + edge_dim, self.out_dim))


    def forward(self, x, graph, edge_attr=None):
        num_nodes = graph.size(0)
        row, col, val = graph.storage.row(), graph.storage.col(), graph.storage.value().view(-1,1)

        attn1 = self.a1(x).view(-1, self.n_heads)
        attn2 = self.a2(x).view(-1, self.n_heads)
        if edge_attr is not None:
            attn_e = self.ae(edge_attr).view(-1, self.n_heads)
            attn = (attn1[row] + attn2[col] + attn_e) / math.sqrt(self.out_dim * 1.0)
        else:
            attn = (attn1[row] + attn2[col]) / math.sqrt(self.out_dim * 1.0)
        attn_d = softmax(attn, row, num_nodes=x.size(0))

        out = []
        for i in range(self.n_heads):
            attn_adj = SparseTensor(row=row, col=col, value=attn_d[:, i], sparse_sizes=(num_nodes, x.size(0)), is_sorted=True)

            x_obj = attn_adj @ x
            if edge_attr is not None:
                x_e = scatter_add(attn_d[:, i].view(-1, 1) * edge_attr, row, dim=0, dim_size=graph.size(0))
                out.append(self.nn[i](torch.cat([x, x_obj, x_e], dim=1)).unsqueeze(1))
            else:
                out.append(self.nn[i](torch.cat([x, x_obj], dim=1)).unsqueeze(1))
        out = torch.cat(out, dim=1).flatten(1, -1)
        return out

class GCNConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 edge_dim = 0
                 ):
        super(GCNConv, self).__init__()
        self.lin = torch.nn.Linear(in_channels + edge_dim, out_channels, bias=bias)
        self.reset_parameters()

    def forward(self, x, graph, edge_attr=None):
        out = graph @ x
        if edge_attr is not None:
            row, norm = graph.storage.row(), graph.storage.value()
            msg_e = norm.view(-1, 1) * edge_attr
            msg_e = scatter_add(msg_e, row, dim=0, dim_size=graph.size(0))
            out = torch.cat([out, msg_e])
        return self.lin(out)

    def reset_parameters(self):
        glorot(self.lin.weight)
        if self.lin.bias is not None:
            zeros(self.lin.bias)

class SAGEConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 edge_dim = 0
                 ):
        super(SAGEConv, self).__init__()
        bias_l = bias
        self.lin_l = torch.nn.Linear(in_channels + edge_dim, out_channels, bias=bias_l)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)
        if self.lin_r.bias is not None:
            zeros(self.lin_r.bias)

    def forward(self, x, graph, edge_attr=None):
        out_l = graph @ x
        if edge_attr is not None:
            row, norm = graph.storage.row(), graph.storage.value()
            msg_e = norm.view(-1, 1) * edge_attr
            msg_e = scatter_add(msg_e, row, dim=0, dim_size=x.size(0))
            out_l = torch.cat([out_l, msg_e])

        out_l = self.lin_l(out_l)
        out_r = self.lin_r(x)
        return out_l + out_r
