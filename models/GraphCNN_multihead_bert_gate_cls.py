"""Legacy compatibility wrapper for the renamed BERT graph reasoner."""

from .bert_filter_reasoner import BertDocRelationGraphModel, GraphCNN_multihead_bert_gate_cls

__all__ = ["BertDocRelationGraphModel", "GraphCNN_multihead_bert_gate_cls"]
