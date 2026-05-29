from .base import BaseChunker, Chunk
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .token_aware import TokenAwareChunker
from .sentence_window import SentenceWindowChunker, expand_window
from .semantic import SemanticChunker
from .late_chunking import LateChunkingChunker
from .proposition import PropositionChunker

STRATEGY_MAP = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "token_aware": TokenAwareChunker,
    "sentence_window": SentenceWindowChunker,
    "semantic": SemanticChunker,
    "late_chunking": LateChunkingChunker,
    "proposition": PropositionChunker,
}

ALL_STRATEGIES = list(STRATEGY_MAP.keys())

__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "RecursiveChunker",
    "TokenAwareChunker",
    "SentenceWindowChunker",
    "SemanticChunker",
    "LateChunkingChunker",
    "PropositionChunker",
    "expand_window",
    "STRATEGY_MAP",
    "ALL_STRATEGIES",
]
