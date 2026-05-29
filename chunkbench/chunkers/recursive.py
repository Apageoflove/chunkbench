from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk

_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []
        raw = self._recursive_split(text, _SEPARATORS, self.chunk_size)
        merged = self._merge_small(raw, self.chunk_size)
        return self._build_chunks(merged, doc_id)

    def _recursive_split(
        self, text: str, separators: list[str], max_size: int
    ) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= max_size:
            return [text]
        if not separators:
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) <= max_size:
                    current = part
                else:
                    sub = self._recursive_split(part, rest, max_size)
                    result.extend(sub[:-1])
                    current = sub[-1] if sub else ""
        if current:
            result.append(current)
        return result

    @staticmethod
    def _merge_small(pieces: list[str], max_size: int) -> list[str]:
        if not pieces:
            return []
        merged: list[str] = []
        current = pieces[0]
        for piece in pieces[1:]:
            candidate = current + "\n" + piece
            if len(candidate) <= max_size:
                current = candidate
            else:
                merged.append(current)
                current = piece
        merged.append(current)
        return merged

    def _build_chunks(self, pieces: list[str], doc_id: str) -> list[Chunk]:
        offset = 0
        chunks: list[Chunk] = []
        for idx, content in enumerate(pieces):
            content = content.strip()
            if not content:
                continue
            start = offset
            end = start + len(content)
            offset = end
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
        return chunks
