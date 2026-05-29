from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            dummy = self.embed_texts(["test"])
            self._dim = dummy.shape[1]
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        model = self._get_model()
        embeddings = model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        result = self.embed_texts([query])
        return result[0]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
