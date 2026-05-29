from __future__ import annotations

import re
from typing import List

from .base import BaseChunker, Chunk


class TokenAwareChunker(BaseChunker):
    strategy_name = "token_aware"

    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 20):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_tokens = 0
        idx = 0
        char_offset = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > self.max_tokens and current_sents:
                content = " ".join(current_sents)
                chunks.append(
                    self._build(content, doc_id, idx, char_offset)
                )
                char_offset += len(content) + 1
                idx += 1
                overlap_sents, overlap_toks = self._overlap_tail(current_sents)
                current_sents = overlap_sents
                current_tokens = overlap_toks

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            content = " ".join(current_sents)
            chunks.append(self._build(content, doc_id, idx, char_offset))

        return chunks

    def _build(self, content: str, doc_id: str, idx: int, start_char: int) -> Chunk:
        return Chunk(
            chunk_id=self._make_chunk_id(doc_id, idx),
            content=content,
            doc_id=doc_id,
            metadata={
                "start_char": start_char,
                "end_char": start_char + len(content),
                "chunk_index": idx,
                "strategy": self.strategy_name,
            },
        )

    def _overlap_tail(self, sentences: List[str]) -> tuple:
        tail: List[str] = []
        tokens = 0
        for s in reversed(sentences):
            t = self._count_tokens(s)
            if tokens + t > self.overlap_tokens:
                break
            tail.insert(0, s)
            tokens += t
        return tail, tokens

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p.strip() for p in parts if p.strip()]
