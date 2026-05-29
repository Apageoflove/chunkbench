from __future__ import annotations

import re
from typing import List

from .base import BaseChunker, Chunk


class SentenceWindowChunker(BaseChunker):
    strategy_name = "sentence_window"

    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        all_sents = [s.strip() for s in sentences if s.strip()]
        chunks: List[Chunk] = []

        for idx, sent in enumerate(all_sents):
            chunks.append(
                Chunk(
                    chunk_id=self._make_chunk_id(doc_id, idx),
                    content=sent,
                    doc_id=doc_id,
                    metadata={
                        "chunk_index": idx,
                        "strategy": self.strategy_name,
                        "sentence_index": idx,
                        "window_size": self.window_size,
                        "total_sentences": len(all_sents),
                        "all_sentences": all_sents,
                    },
                )
            )
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p for p in parts if p.strip()]


def expand_window(chunk: Chunk) -> str:
    meta = chunk.metadata
    all_sents = meta.get("all_sentences", [])
    idx = meta.get("sentence_index", 0)
    ws = meta.get("window_size", 2)
    start = max(0, idx - ws)
    end = min(len(all_sents), idx + ws + 1)
    return " ".join(all_sents[start:end])
