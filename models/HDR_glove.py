"""Legacy compatibility wrapper for the renamed GloVe graph reasoner."""

from .glove_filter_reasoner import GCGCN_glove, GloveDocRelationGraphModel

__all__ = ["GCGCN_glove", "GloveDocRelationGraphModel"]

