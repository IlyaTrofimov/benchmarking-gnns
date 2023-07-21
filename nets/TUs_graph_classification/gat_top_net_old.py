import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, GATTopLayer, GATTopLayer2
from layers.mlp_readout_layer import MLPReadout

import sys
sys.path.append('/ph_simple/lib/')
import ph_simple
import numpy as np

class GATTopNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        self.num_heads = num_heads
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.dropout = dropout
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        #
        # create layers
        # 
        self.top_layer_pos = 3
        self.layers = nn.ModuleList([])

        for i in range(n_layers - 1):
            self.layers.append(GATTopLayer2(hidden_dim * num_heads, hidden_dim, num_heads, dropout, self.batch_norm, self.residual))

        self.layers.append(GATTopLayer2(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(out_dim + self.layers[self.top_layer_pos].num_heads, n_classes)

    def top_feat_fast(self, g, w, head_idx = 0):

        batch_size = g.batch_size

        # calc edge_ptr
        edge_ptr = np.zeros(batch_size + 1, dtype = np.int32)
        shift = 0

        for i in range(batch_size):
            shift += g.batch_num_edges()[i].item()
            edge_ptr[i + 1] = shift

        #print(edge_index_new)

        # calc node ptr
        node_ptr = np.zeros(batch_size + 1, dtype = np.int32)

        shift = 0
        for i in range(batch_size):
            shift += g.batch_num_nodes()[i].item()
            node_ptr[i + 1] = shift

        # calc h0_idx
        w_slice = 1.0 - np.array(w[:, head_idx, 0].detach().cpu(), dtype = np.float32)
        h0_idx = -np.ones(g.num_nodes(), dtype = np.int32)

        edge_index_new = np.zeros(shape = (2, g.num_edges()), dtype = np.int32)
        edge_index_new[0, :] = g.edges()[0]
        edge_index_new[1, :] = g.edges()[1]

        ph_simple.calc_barcodes_batch_bias(batch_size, edge_index_new, w_slice, edge_ptr, node_ptr, h0_idx, 1)

        # trainsform h0_idx to sum of weights
        top_feat = torch.zeros((batch_size))

        for i in range(batch_size):
            for j in range(node_ptr[i], node_ptr[i+1]):
                if h0_idx[j] != -1:
                    top_feat[i] += 1 - w[h0_idx[j], head_idx, 0]

        # node-based features
        last_w = -torch.ones(g.num_nodes())
        sum_w = -torch.zeros(g.num_nodes())

        for j in range(g.num_nodes()):
            if h0_idx[j] != -1:
                w_edge = 1 - w[h0_idx[j], head_idx, 0]
                v1, v2 = edge_index_new[0, h0_idx[j]], edge_index_new[1, h0_idx[j]]

                last_w[v1] = torch.max(last_w[v1], w_edge)
                last_w[v2] = torch.max(last_w[v2], w_edge)
                sum_w[v1] += w_edge
                sum_w[v2] += w_edge

        #print(last_w)
        #print(sum_w)
        #print('h0_idx', h0_idx)

        return top_feat
        
    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            if i == self.top_layer_pos:
                h, attn, top_feat = conv(g, h, top_features = True)
            else:
                h, attn = conv(g, h)

        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        #print(hg.shape, loss1.shape)

        hg = torch.concat((hg, top_feat), axis = 1)
        #print(hg.shape, loss1.shape)
            
        return self.MLP_layer(hg)
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
