import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, GATTopLayer
from layers.mlp_readout_layer import MLPReadout

class GATTopNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.dropout = dropout
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim*num_heads)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.h0_sum = True
        self.top_node_feat = False
        self.cycles = False
        self.top_feat_active = 1.0

        #
        # create layers
        # 
        self.top_layer_pos = 2
        self.layers = nn.ModuleList([])

        for i in range(n_layers - 2):
            self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, self.batch_norm, self.residual))

        self.layers.append(GATTopLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, self.batch_norm, self.residual,
                                                                    h0_sum = self.h0_sum, top_node_feat = self.top_node_feat, cycles = self.cycles))

        #self.layers.append(GATTopLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual,
        #                                                            h0_sum = self.h0_sum, top_node_feat = self.top_node_feat, cycles = self.cycles))

        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))

        extra_global_feats = 0
        if self.h0_sum:
            extra_global_feats += 1
        if self.cycles:
            extra_global_feats += 1

        self.MLP_layer = MLPReadout(out_dim + extra_global_feats * self.layers[self.top_layer_pos].num_heads, 1)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            if isinstance(conv, GATTopLayer):
                h, attn, top_feat, top_feat_cycles = conv(g, h, top_features = True)
            else:
                h = conv(g, h)

        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        if self.h0_sum:
            hg = torch.concat((hg, self.top_feat_active * top_feat), axis = 1)

        if self.cycles:
            hg = torch.concat((hg, self.top_feat_active * top_feat_cycles), axis = 1)
 
        return self.MLP_layer(hg)
   
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
       
