"""Adapters: concrete Scorers and TextEncoders behind the ports in ``callbus.ports``."""

from callbus.adapters.bm25 import BM25Scorer
from callbus.adapters.dense import DenseScorer
from callbus.adapters.embeddings_api import EmbeddingCache, OpenAIEmbeddings

__all__ = ["BM25Scorer", "DenseScorer", "EmbeddingCache", "OpenAIEmbeddings"]
