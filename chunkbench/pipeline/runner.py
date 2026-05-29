from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from ..chunkers.base import BaseChunker, Chunk
from .embedder import Embedder

if TYPE_CHECKING:
    from .retriever import Retriever


@dataclass
class RunResult:
    strategy_name: str
    chunks: List[Chunk]
    chunk_count: int
    avg_chunk_size: float
    avg_token_count: float
    index_time: float
    retrieval_results: List[dict] = field(default_factory=list)


class StrategyRunner:
    def run(
        self,
        chunker: BaseChunker,
        documents: List[dict],
        qa_pairs: List[dict],
        embedder: Embedder,
        retriever: Retriever,
        top_k: int = 5,
    ) -> RunResult:
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = chunker.chunk(doc["text"], doc["doc_id"])
            all_chunks.extend(chunks)

        if not all_chunks:
            return RunResult(
                strategy_name=chunker.strategy_name,
                chunks=[],
                chunk_count=0,
                avg_chunk_size=0,
                avg_token_count=0,
                index_time=0,
                retrieval_results=[],
            )

        col_name = f"cb_{chunker.strategy_name}"
        retriever.clear(col_name)

        t0 = time.time()
        retriever.index(all_chunks, embedder, col_name)
        index_time = time.time() - t0

        avg_size = sum(len(c.content) for c in all_chunks) / len(all_chunks)
        avg_tokens = sum(
            c.metadata.get("token_count", 0) for c in all_chunks
        ) / len(all_chunks)

        retrieval_results: List[dict] = []
        for qa in qa_pairs:
            t1 = time.time()
            retrieved = retriever.query(
                qa["question"], embedder, col_name, top_k=top_k
            )
            retrieval_time = time.time() - t1
            retrieval_results.append(
                {
                    "question": qa["question"],
                    "reference_answer": qa["answer"],
                    "retrieved_chunks": retrieved,
                    "retrieval_time": retrieval_time,
                }
            )

        retriever.clear(col_name)

        return RunResult(
            strategy_name=chunker.strategy_name,
            chunks=all_chunks,
            chunk_count=len(all_chunks),
            avg_chunk_size=avg_size,
            avg_token_count=avg_tokens,
            index_time=index_time,
            retrieval_results=retrieval_results,
        )
