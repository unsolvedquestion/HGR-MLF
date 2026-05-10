import torch
import torch.nn as nn

from runtime_utils.bert_compat import BertSequenceEncoder

from .reasoning_blocks import (
    GATAttention,
    GraphConvolution,
    GraphStructurePurifier,
    MultiGraphConvolution,
    MultiHeadAttention,
    SemanticNoiseGate,
    SentenceAttention,
    WordAttention,
)


class BertDocRelationGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        word_vec_size = 768
        self.bert = BertSequenceEncoder("./bert/bert-base-uncased")
        self.use_entity_type = True
        self.use_coreference = True
        self.use_type_feat = True
        self.use_distance = True
        self.dropout = nn.Dropout(0.2)

        hidden_size = 128
        input_size = word_vec_size
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
        self.layer_num = 4
        self.head_num = 4
        self.get_weighted_adj_matrix = GATAttention(hidden_size, hidden_size)
        self.get_adj_matrix = nn.ModuleList(
            [MultiHeadAttention(self.head_num, hidden_size) for _ in range(self.config.graph_hop - 1)]
        )
        self.graph_purifier = GraphStructurePurifier()
        self.semantic_noise_gate = SemanticNoiseGate(hidden_size)
        self.graphcnn = nn.ModuleList()
        for layer_index in range(self.config.graph_hop):
            if layer_index == 0:
                self.graphcnn.append(GraphConvolution(self.layer_num, hidden_size, hidden_size))
            else:
                self.graphcnn.append(MultiGraphConvolution(self.layer_num, self.head_num, hidden_size, hidden_size))
        self.linear_re = nn.Linear(input_size, hidden_size)
        self.word_attention = nn.ModuleList(
            [WordAttention(hidden_size, hidden_size, position_dim=config.dis_size) for _ in range(self.config.graph_hop)]
        )
        self.sentence_attention = nn.ModuleList(
            [SentenceAttention(hidden_size, hidden_size) for _ in range(self.config.graph_hop)]
        )
        self.linear_word_att = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(self.config.graph_hop)]
        )
        self.linear_sentence_att = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(self.config.graph_hop)]
        )
        if self.use_type_feat:
            self.dense_layer = nn.Linear(
                hidden_size * (config.graph_hop + 1) + config.dis_size + config.entity_type_size,
                hidden_size,
            )
        else:
            self.dense_layer = nn.Linear(hidden_size * (config.graph_hop + 1) + config.dis_size, hidden_size)
        self.bili_layer_01 = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)
        self.classification_layer_01 = nn.Linear(hidden_size * 2, config.relation_num)
        self.linear_cls = nn.Linear(word_vec_size, config.relation_num)
        if self.use_distance:
            self.dis_embed = nn.Embedding(config.dis_num, config.dis_size)

    def forward(
        self,
        document,
        document_ner,
        document_pos,
        adj_matrix,
        sen_matrix,
        pos_matrix_h,
        pos_matrix_t,
        node_pos,
        node_type,
        node_relative_pos,
    ):
        doc = self.bert(document.unsqueeze(0)).squeeze(0)
        cls_feat = doc[0, :]
        if self.use_coreference:
            doc = torch.cat([doc, self.entity_embed(document_pos)], dim=-1)

        if self.use_entity_type:
            doc = torch.cat([doc, self.ner_emb(document_ner)], dim=-1)
        context_output = torch.tanh(self.linear_re(doc.unsqueeze(0)))
        context_output = self.semantic_noise_gate(context_output, document_pos.float(), document_ner.float())
        node_num, max_sen_num = sen_matrix.size(0), sen_matrix.size(2)
        hidden_dim = context_output.size(-1)
        node_feat = node_pos.unsqueeze(2).expand(-1, -1, hidden_dim) * context_output.expand(node_num, -1, -1)
        node_feat = node_feat.sum(dim=1)
        word_att_padding_matrix = ~sen_matrix.unsqueeze(4)
        context_before_att = context_output.unsqueeze(0).unsqueeze(0).expand(node_num, node_num, max_sen_num, -1, -1)
        sent_att_padding_matrix = ~sen_matrix[:, :, :, 0:1]
        dis_embedding_h = self.dis_embed(pos_matrix_h)
        dis_embedding_t = self.dis_embed(pos_matrix_t)
        node_relative_pos_h = self.dis_embed(self.config.dis_plus + node_relative_pos)
        node_relative_pos_t = self.dis_embed(self.config.dis_plus - node_relative_pos)

        node_feats = [node_feat]
        for hop_index in range(self.config.graph_hop):
            context_word_att_h = self.word_attention[hop_index](word_att_padding_matrix, context_before_att, dis_embedding_h)
            context_word_att_t = self.word_attention[hop_index](word_att_padding_matrix, context_before_att, dis_embedding_t)
            context_word_att = torch.cat([context_word_att_h, context_word_att_t], 3)
            context_word_att = self.linear_word_att[hop_index](context_word_att)

            node_embedding_h = node_feat.unsqueeze(0).unsqueeze(2).expand_as(context_word_att)
            node_embedding_t = node_feat.unsqueeze(1).unsqueeze(2).expand_as(context_word_att)
            context_sent_att_h = self.sentence_attention[hop_index](
                sent_att_padding_matrix, context_word_att, node_embedding_h
            )
            context_sent_att_t = self.sentence_attention[hop_index](
                sent_att_padding_matrix, context_word_att, node_embedding_t
            )
            context_sent_att = torch.cat([context_sent_att_h, context_sent_att_t], 2)
            context_sent_att = self.linear_sentence_att[hop_index](context_sent_att)

            if hop_index < 1:
                mask = torch.eq(adj_matrix, 0)
                weight_adj_matrix = self.get_weighted_adj_matrix(node_feat, context_sent_att, mask)
                weight_adj_matrix = self.graph_purifier(weight_adj_matrix, adj_matrix)
                new_node_feat = self.graphcnn[hop_index](node_feat, context_sent_att, weight_adj_matrix)
            else:
                adj_matrix_list = self.get_adj_matrix[hop_index - 1](node_feat, context_sent_att)
                new_node_feat = self.graphcnn[hop_index](node_feat, context_sent_att, adj_matrix_list)
            node_feats.append(node_feat)
            node_feat = self.config.alpha * new_node_feat + (1 - self.config.alpha) * node_feat
            node_feat = self.dropout(node_feat)

        node_feats = torch.cat(node_feats, 1)
        if self.use_type_feat:
            type_feats = self.ner_emb(node_type)
            node_feats_with_type = torch.cat([node_feats, type_feats], 1)
        else:
            node_feats_with_type = node_feats

        node_feats_with_pos_h = torch.cat(
            [node_feats_with_type.unsqueeze(0).expand(node_num, -1, -1), node_relative_pos_h], -1
        )
        node_feats_with_pos_t = torch.cat(
            [node_feats_with_type.unsqueeze(1).expand(-1, node_num, -1), node_relative_pos_t], -1
        )

        entity_feature_h = torch.tanh(self.dense_layer(node_feats_with_pos_h))
        entity_feature_t = torch.tanh(self.dense_layer(node_feats_with_pos_t))
        entity_feature = torch.cat([entity_feature_h, entity_feature_t], -1)
        cls_feature = self.linear_cls(cls_feat).unsqueeze(0).unsqueeze(0).expand(node_num, node_num, -1)
        relation_before_softmax_01 = self.bili_layer_01(
            entity_feature_h.contiguous(), entity_feature_t.contiguous()
        ) + self.classification_layer_01(entity_feature) + cls_feature

        return relation_before_softmax_01


GraphCNN_multihead_bert_gate_cls = BertDocRelationGraphModel
