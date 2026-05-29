from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import tiktoken


_encoding = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    content: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "token_count" not in self.metadata:
            self.metadata["token_count"] = _count_tokens(self.content)


class BaseChunker(ABC):
    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """Split *text* into a list of Chunk objects."""

    def _make_chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}_{index:04d}"

    @staticmethod
    def _count_tokens(text: str) -> int:
        return _count_tokens(text)
