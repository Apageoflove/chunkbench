from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseChunker, Chunk


class LateChunkingChunker(BaseChunker):
    strategy_name = "late_chunking"

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        overlap_tokens: int = 32,
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.embed_model = embed_model
        self._model = None
        self._tokenizer = None

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text.strip():
            return []

        model, tokenizer = self._load_model()
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"]
        n_tokens = input_ids.shape[1]

        if n_tokens == 0:
            return []

        with __import__("torch").no_grad():
            outputs = model(input_ids)
            token_embeddings = outputs.last_hidden_state[0].numpy()

        token_texts = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        chunks: List[Chunk] = []
        idx = 0
        start = 0

        while start < n_tokens:
            end = min(start + self.chunk_size_tokens, n_tokens)
            chunk_tokens = token_texts[start:end]
            content = tokenizer.decode(input_ids[0][start:end], skip_special_tokens=True)

            chunk_emb = np.mean(token_embeddings[start:end], axis=0)
            chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-10)

            chunks.append(
                Chunk(
                    chunk_id=self._make_chunk_id(doc_id, idx),
                    content=content.strip(),
                    doc_id=doc_id,
                    metadata={
                        "start_token": start,
                        "end_token": end,
                        "chunk_index": idx,
                        "strategy": self.strategy_name,
                        "precomputed_embedding": chunk_emb.tolist(),
                    },
                )
            )
            idx += 1
            step = self.chunk_size_tokens - self.overlap_tokens
            if step <= 0:
                step = self.chunk_size_tokens
            start += step

        return chunks

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.embed_model)
            self._model = AutoModel.from_pretrained(self.embed_model)
            self._model.eval()
        return self._model, self._tokenizer
