from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    strategy_name = "semantic"

    def __init__(
        self,
        embedder=None,
        breakpoint_threshold: float = 0.5,
        max_chunk_size: int = 600,
    ):
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        if self.embedder is None:
            return self._fallback_chunk(text, doc_id)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._simple_chunks(sentences, doc_id)

        embeddings = self.embedder.embed_texts(sentences)
        similarities = self._cosine_similarities(embeddings)
        breakpoints = self._find_breakpoints(similarities)

        return self._group_sentences(sentences, breakpoints, doc_id)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _cosine_similarities(embeddings: np.ndarray) -> List[float]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms
        sims = np.sum(normed[:-1] * normed[1:], axis=1)
        return sims.tolist()

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        bp = []
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                bp.append(i + 1)
        return bp

    def _group_sentences(
        self, sentences: List[str], breakpoints: List[int], doc_id: str
    ) -> List[Chunk]:
        boundaries = [0] + breakpoints + [len(sentences)]
        chunks: List[Chunk] = []
        idx = 0
        char_offset = 0

        for i in range(len(boundaries) - 1):
            group = sentences[boundaries[i] : boundaries[i + 1]]
            content = " ".join(group)
            if not content.strip():
                continue
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_by_size(content, self.max_chunk_size)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            chunk_id=self._make_chunk_id(doc_id, idx),
                            content=sub,
                            doc_id=doc_id,
                            metadata={
                                "start_char": char_offset,
                                "end_char": char_offset + len(sub),
                                "chunk_index": idx,
                                "strategy": self.strategy_name,
                            },
                        )
                    )
                    char_offset += len(sub) + 1
                    idx += 1
            else:
                chunks.append(
                    Chunk(
                        chunk_id=self._make_chunk_id(doc_id, idx),
                        content=content,
                        doc_id=doc_id,
                        metadata={
                            "start_char": char_offset,
                            "end_char": char_offset + len(content),
                            "chunk_index": idx,
                            "strategy": self.strategy_name,
                        },
                    )
                )
                char_offset += len(content) + 1
                idx += 1
        return chunks

    @staticmethod
    def _split_by_size(text: str, max_size: int) -> List[str]:
        parts: List[str] = []
        while text:
            parts.append(text[:max_size])
            text = text[max_size:]
        return parts

    def _fallback_chunk(self, text: str, doc_id: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        return self._group_sentences(sentences, [], doc_id)

    def _simple_chunks(self, sentences: List[str], doc_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for idx, sent in enumerate(sentences):
            chunks.append(
                Chunk(
                    chunk_id=self._make_chunk_id(doc_id, idx),
                    content=sent,
                    doc_id=doc_id,
                    metadata={
                        "chunk_index": idx,
                        "strategy": self.strategy_name,
                    },
                )
            )
        return chunks
