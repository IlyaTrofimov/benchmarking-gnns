import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

import sys
sys.path.append('/ph_simple/lib/')
import ph_simple
import numpy as np



"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATTopLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu,
                                                        h0_sum = True, top_node_feat = False, cycles = False):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
        self.num_heads = num_heads

        self.h0_sum = h0_sum
        self.top_node_feat = top_node_feat
        self.cycles = cycles

        if top_node_feat:
            minus_out_f = 2 * num_heads
        else:
            minus_out_f = 0
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        if dgl.__version__ < "0.5":
            self.gatconv = GATConv(in_dim, out_dim - minus_out_f, num_heads, dropout, dropout)
        else:
            self.gatconv = GATConv(in_dim, out_dim - minus_out_f, num_heads, dropout, dropout, allow_zero_in_degree=True)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def top_feat_fast(self, g, w, head_idx = 0):

        batch_size = g.batch_size

        # calc edge_ptr
        edge_ptr = np.zeros(batch_size + 1, dtype = np.int32)
        shift = 0

        for i in range(batch_size):
            shift += g.batch_num_edges()[i].item()
            edge_ptr[i + 1] = shift

        # calc node ptr
        node_ptr = np.zeros(batch_size + 1, dtype = np.int32)

        shift = 0
        for i in range(batch_size):
            shift += g.batch_num_nodes()[i].item()
            node_ptr[i + 1] = shift

        # calc h0_idx
        w_slice = 1.0 - np.array(w[:, head_idx, 0].detach().cpu(), dtype = np.float32)
        h0_idx = -np.ones(g.num_nodes(), dtype = np.int32)
        h1_idx = -np.ones(g.num_edges(), dtype = np.int32)

        edge_index_new = np.zeros(shape = (2, g.num_edges()), dtype = np.int32)
        edge_index_new[0, :] = g.edges()[0]
        edge_index_new[1, :] = g.edges()[1]
        
        for i in range(batch_size):
            for j in range(edge_ptr[i], edge_ptr[i+1]):
                edge_index_new[0, j] -= node_ptr[i]
                edge_index_new[1, j] -= node_ptr[i]
                #print(j, node_ptr[i], edge_index_new[0, j])

        if self.h0_sum and self.cycles:
            ph_simple.calc_barcodes_batch_cycles(batch_size, edge_index_new, w_slice, edge_ptr, node_ptr, h0_idx, h1_idx, 1, 1)
        elif self.h0_sum:
            ph_simple.calc_barcodes_batch(batch_size, edge_index_new, w_slice, edge_ptr, node_ptr, h0_idx, 1)

        # trainsform h0_idx to sum of weights
        top_feat = torch.zeros((batch_size))

        if self.h0_sum:

            for i in range(batch_size):
                for j in range(node_ptr[i], node_ptr[i+1]):
                    edge_j = h0_idx[j]
                    if edge_j != -1:
                        top_feat[i] += 1 - w[edge_j, head_idx, 0]

        # trainsform h1_idx to sum of weights
        top_feat_cycles = torch.zeros((batch_size))

        if self.cycles:

            for i in range(batch_size):
                for j in range(edge_ptr[i], edge_ptr[i+1]):
                    edge_j = h1_idx[j]
                    #print(i, j, edge_j)
                    if edge_j != -1:
                        top_feat_cycles[i] += 1 - w[edge_j, head_idx, 0]
                    else:
                        break

        # node-based features
        last_w = -torch.zeros(g.num_nodes())
        sum_w = -torch.zeros(g.num_nodes())

        if self.top_node_feat:

            for j in range(g.num_nodes()):
                edge_j = h0_idx[j]

                if edge_j != -1:
                    w_edge = 1 - w[edge_j, head_idx, 0]
                    v1, v2 = g.edges()[0][edge_j], g.edges()[1][edge_j]
        
                    last_w[v1] = w_edge
                    last_w[v2] = w_edge
                    sum_w[v1] += w_edge
                    sum_w[v2] += w_edge

        return top_feat, top_feat_cycles, last_w, sum_w

    def forward(self, g, h, top_features = False):
        h_in = h # for residual connection

        h_raw, attn = self.gatconv(g, h, get_attention = True)
        h = h_raw.flatten(1)

        top_feat = torch.zeros((g.batch_size, self.num_heads))
        top_feat_cycles = torch.zeros((g.batch_size, self.num_heads))
        last_w = torch.zeros((g.num_nodes(), self.num_heads))
        sum_w = torch.zeros((g.num_nodes(), self.num_heads))

        # top features
        if top_features:
            for head_idx in range(self.num_heads):
                top_feat[:, head_idx], top_feat_cycles[:, head_idx], last_w[:, head_idx], sum_w[:, head_idx] = self.top_feat_fast(g, attn, head_idx)

            if self.top_node_feat:
                h = torch.concat((h, last_w, sum_w), axis = 1)

        #print(h_in.shape, h.shape)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        if top_features:
            return h, attn, top_feat, top_feat_cycles
        else:
            return h
 
class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        if dgl.__version__ < "0.5":
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)
        else:
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout, allow_zero_in_degree=True)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h # for residual connection

        h = self.gatconv(g, h).flatten(1)

        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h
    

##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class CustomGATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayer(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc_h = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_e = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_proj = nn.Linear(3* out_dim, out_dim)
        self.attn_fc = nn.Linear(3* out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.batchnorm_e = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'], edges.dst['z_h']], dim=1)
        e_proj = self.fc_proj(z)
        attn = F.leaky_relu(self.attn_fc(z))
        return {'attn': attn, 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, g, h, e):
        z_h = self.fc_h(h)
        z_e = self.fc_e(e)
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e
        
        g.apply_edges(self.edge_attention)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h']
        e = g.edata['e_proj']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        h = F.elu(h)
        e = F.elu(e)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    

class CustomGATLayerEdgeReprFeat(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerEdgeReprFeat(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)

        if self.merge == 'cat':
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        else:
            raise NotImplementedError

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerIsotropic(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayerIsotropic(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerIsotropic(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
