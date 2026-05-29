from __future__ import annotations

from typing import List

import chromadb

from ..chunkers.base import Chunk
from ..chunkers.sentence_window import expand_window
from .embedder import Embedder


class Retriever:
    def __init__(self, persist_dir: str = "./.chunkbench_cache"):
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict = {}

    def index(
        self,
        chunks: List[Chunk],
        embedder: Embedder,
        collection_name: str,
    ):
        collection = self._get_or_create(collection_name, embedder.dimension)

        precomputed = [c.metadata.get("precomputed_embedding") for c in chunks]
        all_precomputed = all(p is not None for p in precomputed)

        if all_precomputed:
            ids = [c.chunk_id for c in chunks]
            docs = [c.content for c in chunks]
            metas = [self._clean_meta(c.metadata) for c in chunks]
            embeddings = precomputed
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=docs[i : i + batch_size],
                    metadatas=metas[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size],
                )
        else:
            texts = [c.content for c in chunks]
            emb_array = embedder.embed_texts(texts)
            ids = [c.chunk_id for c in chunks]
            docs = [c.content for c in chunks]
            metas = [self._clean_meta(c.metadata) for c in chunks]
            emb_list = emb_array.tolist()
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=docs[i : i + batch_size],
                    metadatas=metas[i : i + batch_size],
                    embeddings=emb_list[i : i + batch_size],
                )

    def query(
        self,
        query_text: str,
        embedder: Embedder,
        collection_name: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        collection = self._collections.get(collection_name)
        if collection is None:
            collection = self._client.get_collection(collection_name)
            self._collections[collection_name] = collection

        q_emb = embedder.embed_query(query_text).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)

        chunks: List[Chunk] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            strategy = meta.get("strategy", "")
            doc_id = cid.rsplit("_", 1)[0] if "_" in cid else ""

            chunk = Chunk(
                chunk_id=cid,
                content=results["documents"][0][i],
                doc_id=doc_id,
                metadata=meta,
            )

            if strategy == "sentence_window":
                chunk.content = expand_window(chunk)

            chunks.append(chunk)

        return chunks

    def clear(self, collection_name: str):
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collections.pop(collection_name, None)

    def _get_or_create(self, name: str, dim: int):
        if name not in self._collections:
            try:
                col = self._client.get_collection(name)
            except Exception:
                col = self._client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            self._collections[name] = col
        return self._collections[name]

    @staticmethod
    def _clean_meta(metadata: dict) -> dict:
        clean = {}
        for k, v in metadata.items():
            if k == "all_sentences":
                continue
            if k == "precomputed_embedding":
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
        return clean
