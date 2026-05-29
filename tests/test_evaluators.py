from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkbench.chunkers import FixedSizeChunker
from chunkbench.evaluators import (
    RetrievalEvaluator,
    GenerationEvaluator,
    LatencyEvaluator,
)
from chunkbench.chunkers.base import Chunk
from chunkbench.pipeline.runner import RunResult


def _make_run_result():
    chunks = [
        Chunk(
            chunk_id="doc1_0000",
            content="RAG systems use vector databases for similarity search.",
            doc_id="doc1",
            metadata={"chunk_index": 0, "strategy": "fixed_size"},
        ),
        Chunk(
            chunk_id="doc1_0001",
            content="The retrieval component converts queries into vectors using embedding models.",
            doc_id="doc1",
            metadata={"chunk_index": 1, "strategy": "fixed_size"},
        ),
    ]

    return RunResult(
        strategy_name="fixed_size",
        chunks=chunks,
        chunk_count=2,
        avg_chunk_size=80.0,
        avg_token_count=20.0,
        index_time=0.5,
        retrieval_results=[
            {
                "question": "What do RAG systems use?",
                "reference_answer": "RAG systems use vector databases for similarity search.",
                "retrieved_chunks": [chunks[0]],
                "retrieval_time": 0.01,
            },
            {
                "question": "What does retrieval do?",
                "reference_answer": "The retrieval component converts queries into vectors.",
                "retrieved_chunks": [chunks[1], chunks[0]],
                "retrieval_time": 0.02,
            },
        ],
    )


def test_retrieval_evaluator_recall_range():
    evaluator = RetrievalEvaluator()
    result = _make_run_result()
    qa_pairs = [
        {"question": "What do RAG systems use?", "answer": "RAG systems use vector databases for similarity search."},
        {"question": "What does retrieval do?", "answer": "The retrieval component converts queries into vectors."},
    ]
    metrics = evaluator.evaluate(result, qa_pairs)
    assert 0 <= metrics["recall_at_5"] <= 1.0
    assert 0 <= metrics["recall_at_1"] <= 1.0
    assert 0 <= metrics["mrr"] <= 1.0


def test_retrieval_evaluator_mrr():
    evaluator = RetrievalEvaluator()
    result = _make_run_result()
    qa_pairs = [
        {"question": "Q1", "answer": "RAG systems use vector databases for similarity search."},
        {"question": "Q2", "answer": "The retrieval component converts queries into vectors."},
    ]
    metrics = evaluator.evaluate(result, qa_pairs)
    assert metrics["mrr"] > 0, "MRR should be positive when first results match"


def test_latency_evaluator_fields():
    evaluator = LatencyEvaluator()
    result = _make_run_result()
    metrics = evaluator.evaluate(result)
    assert "index_time_s" in metrics
    assert "avg_retrieval_time_ms" in metrics
    assert "p95_retrieval_time_ms" in metrics
    assert "chunk_count" in metrics
    assert "throughput_qps" in metrics
    assert metrics["chunk_count"] == 2


def test_generation_evaluator_rouge_fallback():
    evaluator = GenerationEvaluator(llm_client=None)
    result = _make_run_result()
    metrics = evaluator.evaluate(result)
    assert metrics["evaluation_mode"] == "rouge_only"
    assert 0 <= metrics["rouge_l"] <= 1.0
    assert "faithfulness" in metrics
    assert "relevance" in metrics


if __name__ == "__main__":
    test_retrieval_evaluator_recall_range()
    test_retrieval_evaluator_mrr()
    test_latency_evaluator_fields()
    test_generation_evaluator_rouge_fallback()
    print("All evaluator tests passed!")
