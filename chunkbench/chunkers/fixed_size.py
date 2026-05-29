from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_id=self._make_chunk_id(doc_id, idx),
                        content=content,
                        doc_id=doc_id,
                        metadata={
                            "start_char": start,
                            "end_char": end,
                            "chunk_index": idx,
                            "strategy": self.strategy_name,
                        },
                    )
                )
                idx += 1
            step = self.chunk_size - self.overlap
            if step <= 0:
                step = self.chunk_size
            start += step

        return chunks
