import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GraphConv(nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim, bias=False):
        super().__init__()
        self.weights_edge = nn.Parameter(torch.FloatTensor(edge_dim, output_dim))
        self.weights_node = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_edge.data)
        nn.init.xavier_uniform_(self.weights_node.data)

    def forward(self, inputs, edge_inputs, adjacency_matrix):
        outputs_edge = torch.einsum("ijk,kp->ijp", edge_inputs, self.weights_edge)
        outputs_edge = torch.mean(outputs_edge, dim=1)
        outputs_node = torch.chain_matmul(adjacency_matrix, inputs, self.weights_node)
        outputs = outputs_edge + outputs_node

        if self.bias is not None:
            outputs += self.bias

        node_weight = torch.sum(adjacency_matrix, 1)
        node_weight_zero = torch.eq(node_weight, 0).float()
        node_weight = (node_weight + node_weight_zero).unsqueeze(1).expand_as(outputs)
        return outputs / node_weight


class GraphConvolution(nn.Module):
    def __init__(self, layer_num, input_dim, output_dim):
        super().__init__()
        hidden_dim = output_dim
        graph_hidden_dim = int(hidden_dim / layer_num)
        self.gcn_dropout = nn.Dropout(0.2)
        self.graphconv = nn.ModuleList(
            [GraphConv(input_dim + graph_hidden_dim * i, input_dim, graph_hidden_dim) for i in range(layer_num)]
        )
        self.linear_layer = nn.Linear(hidden_dim, output_dim)
        self.layer_num = layer_num

    def forward(self, node_feat, edge_feat, adj_matrix):
        outputs = node_feat
        output_list = []
        cache_list = [outputs]
        for layer_index in range(self.layer_num):
            graph_out = torch.relu(self.graphconv[layer_index](outputs, edge_feat, adj_matrix))
            cache_list.append(graph_out)
            outputs = torch.cat(cache_list, dim=-1)
            output_list.append(self.gcn_dropout(graph_out))
        node_feat_output = torch.cat(output_list, dim=-1)
        node_feat_output = node_feat_output + node_feat
        return self.linear_layer(node_feat_output)


class MultiGraphConvolution(nn.Module):
    def __init__(self, layer_num, head_num, input_dim, output_dim):
        super().__init__()
        hidden_dim = output_dim
        graph_hidden_dim = int(hidden_dim / layer_num)
        self.gcn_dropout = nn.Dropout(0.2)
        self.graphconv = nn.ModuleList()
        for _head_index in range(head_num):
            for layer_index in range(layer_num):
                self.graphconv.append(GraphConv(input_dim + graph_hidden_dim * layer_index, input_dim, graph_hidden_dim))
        self.linear_layer = nn.Linear(hidden_dim * head_num, output_dim)
        self.layer_num = layer_num
        self.head_num = head_num

    def forward(self, node_feat, edge_feat, adj_matrix_list):
        feat_head_list = []
        for head_index in range(self.head_num):
            outputs = node_feat
            output_list = []
            cache_list = [outputs]
            for layer_index in range(self.layer_num):
                index = head_index * self.layer_num + layer_index
                graph_out = torch.relu(self.graphconv[index](outputs, edge_feat, adj_matrix_list[head_index]))
                cache_list.append(graph_out)
                outputs = torch.cat(cache_list, dim=-1)
                output_list.append(self.gcn_dropout(graph_out))
            node_feat_output = torch.cat(output_list, dim=-1)
            node_feat_output = node_feat_output + node_feat
            feat_head_list.append(node_feat_output)

        feat_head_list = torch.cat(feat_head_list, -1)
        return self.linear_layer(feat_head_list)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, att_size, dropout=0.1):
        super().__init__()
        assert att_size % head_num == 0

        self.hidden_size = att_size // head_num
        self.head_num = head_num
        self.linears_q = nn.ModuleList([nn.Linear(att_size, self.hidden_size) for _ in range(head_num)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_feat, mask=None):
        del mask
        att_list = []
        for head_index in range(self.head_num):
            query = self.linears_q[head_index](node_feat)
            key = self.linears_q[head_index](node_feat).transpose(0, 1)
            att = torch.softmax(torch.mm(query, key) / math.sqrt(self.hidden_size), dim=-1)
            att = self.dropout(att)
            att_list.append(att)
        return att_list


class GATAttention(nn.Module):
    def __init__(self, att_input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear_node_h = nn.Linear(att_input_dim, hidden_dim)
        self.linear_node_t = nn.Linear(att_input_dim, hidden_dim)
        self.linear_edge_r = nn.Linear(att_input_dim, hidden_dim)
        self.wt = nn.Linear(hidden_dim * 3, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_feat, edge_feat, mask=None):
        node_num = node_feat.size(0)
        node_feat_h = node_feat.unsqueeze(0).expand(node_num, node_num, -1)
        node_feat_t = node_feat.unsqueeze(0).expand(node_num, node_num, -1)

        node_att_h = self.linear_node_h(node_feat_h)
        node_att_t = self.linear_node_t(node_feat_t)
        edge_att_r = self.linear_edge_r(edge_feat)
        energy_att = self.wt(torch.cat([node_att_h, node_att_t, edge_att_r], -1)).squeeze(-1)
        if mask is not None:
            energy_att = energy_att.masked_fill(mask, -100000.0)
        att = torch.softmax(energy_att, dim=-1)
        return self.dropout(att)


class WordAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, position_dim):
        super().__init__()
        self.attention_sent = nn.Linear(input_dim, hidden_dim)
        self.attention_pos = nn.Linear(position_dim, hidden_dim)
        self.attention_all = nn.Linear(hidden_dim, 1)

    def forward(self, att_padding_matrix, context_before_att, dis_embedding):
        sent_feat = self.attention_sent(context_before_att)
        dis_feat = self.attention_pos(dis_embedding)
        all_feat = self.attention_all(torch.tanh(sent_feat + dis_feat))
        att_padding_matrix = att_padding_matrix.expand_as(all_feat)
        all_feat = all_feat.masked_fill(att_padding_matrix, -100000.0)
        att_matrix = nn.functional.softmax(all_feat, dim=3).expand_as(context_before_att)
        return torch.sum(att_matrix * context_before_att, dim=3)


class SentenceAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention_sent = nn.Linear(input_dim, hidden_dim)
        self.attention_pos = nn.Linear(input_dim, hidden_dim)
        self.attention_all = nn.Linear(hidden_dim, 1)

    def forward(self, att_padding_matrix, context_word_att, node_embedding):
        sent_feat = self.attention_sent(context_word_att)
        dis_feat = self.attention_pos(node_embedding)
        all_feat = self.attention_all(torch.tanh(sent_feat + dis_feat))
        sent_num = att_padding_matrix.sum(dim=2)
        all_feat = all_feat.masked_fill(att_padding_matrix, -100000.0)
        att_matrix = torch.relu(all_feat).expand_as(context_word_att)
        return torch.sum(att_matrix * context_word_att, dim=2) / (sent_num + 1e-10)


class SemanticNoiseGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.feature_projection = nn.Linear(3, hidden_size)

    def forward(self, context_output, document_pos, document_ner):
        seq_len = context_output.size(1)
        device = context_output.device
        entity_signal = (document_pos > 0).float()
        ner_signal = (document_ner > 0).float()
        relative_position = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
        feature_stack = torch.stack([entity_signal, ner_signal, relative_position], dim=-1).unsqueeze(0)
        gate = torch.sigmoid(self.context_projection(context_output) + self.feature_projection(feature_stack))
        return context_output * gate


class GraphStructurePurifier(nn.Module):
    def __init__(self, threshold_ratio=0.6):
        super().__init__()
        self.threshold_ratio = threshold_ratio

    def forward(self, weighted_adj, base_adj):
        edge_mask = base_adj > 0
        masked_scores = weighted_adj * edge_mask.float()
        row_edge_count = edge_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        row_mean = masked_scores.sum(dim=1, keepdim=True) / row_edge_count
        keep_mask = masked_scores >= (row_mean * self.threshold_ratio)
        purified = masked_scores * keep_mask.float()
        fallback_mask = purified.sum(dim=1, keepdim=True).eq(0)
        purified = torch.where(fallback_mask, masked_scores, purified)
        return purified


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training:
            return x
        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(mask.div_(1 - self.dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for layer_index in range(nlayers):
            if layer_index == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        hidden_state_count = 2 if bidir else 1
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_state_count, 1, num_units).zero_()) for _ in range(nlayers)]
        )
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_state_count, 1, num_units).zero_()) for _ in range(nlayers)]
        )

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    def get_init(self, batch_size, layer_index):
        return (
            self.init_hidden[layer_index].expand(-1, batch_size, -1).contiguous(),
            self.init_c[layer_index].expand(-1, batch_size, -1).contiguous(),
        )

    def forward(self, input_tensor, input_lengths=None):
        del input_lengths
        batch_size = input_tensor.size(0)
        output = input_tensor
        outputs = []

        for layer_index in range(self.nlayers):
            hidden, cell = self.get_init(batch_size, layer_index)
            output = self.dropout(output)
            output, hidden = self.rnns[layer_index](output, (hidden, cell))
            del hidden
            outputs.append(output)

        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]
