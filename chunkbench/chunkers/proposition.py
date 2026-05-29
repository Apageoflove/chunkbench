from __future__ import annotations

import json
import re
from typing import List, Optional

from .base import BaseChunker, Chunk
from .recursive import RecursiveChunker


class PropositionChunker(BaseChunker):
    strategy_name = "proposition"

    def __init__(
        self,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
        max_propositions_per_para: int = 10,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.max_propositions_per_para = max_propositions_per_para
        self._fallback_chunker = RecursiveChunker(chunk_size=1000)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        paragraphs = self._fallback_chunker.chunk(text, doc_id)

        if self.llm_client is None:
            return self._fallback_sentence_split(paragraphs, doc_id)

        chunks: List[Chunk] = []
        idx = 0

        for para_chunk in paragraphs:
            propositions = self._extract_propositions(para_chunk.content)
            for prop in propositions[: self.max_propositions_per_para]:
                prop = prop.strip()
                if not prop:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=self._make_chunk_id(doc_id, idx),
                        content=prop,
                        doc_id=doc_id,
                        metadata={
                            "source_paragraph": para_chunk.content[:200],
                            "chunk_index": idx,
                            "strategy": self.strategy_name,
                        },
                    )
                )
                idx += 1

        return chunks

    def _extract_propositions(self, paragraph: str) -> List[str]:
        prompt = (
            "Please decompose the following paragraph into a list of independent, "
            "self-contained factual propositions. Each proposition must:\n"
            "- Include a complete subject (no pronouns)\n"
            "- Express one complete fact\n"
            "- Be independently understandable without context\n\n"
            f"Return as a JSON array: [\"proposition1\", \"proposition2\", ...]\n\n"
            f"Paragraph:\n{paragraph}"
        )

        try:
            if hasattr(self.llm_client, "chat"):
                resp = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = resp.choices[0].message.content
            else:
                return self._split_by_punctuation(paragraph)

            return self._parse_json_list(text)
        except Exception:
            return self._split_by_punctuation(paragraph)

    @staticmethod
    def _parse_json_list(text: str) -> List[str]:
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5]

    @staticmethod
    def _split_by_punctuation(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p.strip() for p in parts if len(p.strip()) > 5]

    def _fallback_sentence_split(self, paragraphs, doc_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        idx = 0
        for para in paragraphs:
            sents = self._split_by_punctuation(para.content)
            for sent in sents:
                if not sent.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=self._make_chunk_id(doc_id, idx),
                        content=sent,
                        doc_id=doc_id,
                        metadata={
                            "chunk_index": idx,
                            "strategy": self.strategy_name,
                            "fallback": True,
                        },
                    )
                )
                idx += 1
        return chunks
