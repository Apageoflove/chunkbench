from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkbench.chunkers import FixedSizeChunker, RecursiveChunker
from chunkbench.pipeline import Embedder, Retriever, StrategyRunner


SAMPLE_DOCS = [
    {
        "doc_id": "doc1",
        "text": (
            "RAG systems use vector databases for similarity search. "
            "The retrieval component converts queries into vectors. "
            "Chunking strategies impact retrieval quality significantly. "
            "Embedding models produce dense vector representations."
        ),
    }
]

SAMPLE_QA = [
    {
        "question": "What do RAG systems use for similarity search?",
        "answer": "RAG systems use vector databases for similarity search.",
    },
    {
        "question": "What does the retrieval component do?",
        "answer": "The retrieval component converts queries into vectors using the same embedding model.",
    },
]


def test_embedder_local_model():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vecs = embedder.embed_texts(["hello world", "test sentence"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0

    qvec = embedder.embed_query("hello")
    assert qvec.shape[0] > 0


def test_retriever_index_and_query():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    retriever = Retriever(persist_dir="./.test_cache")

    chunker = FixedSizeChunker(chunk_size=200, overlap=0)
    chunks = chunker.chunk(SAMPLE_DOCS[0]["text"], SAMPLE_DOCS[0]["doc_id"])
    assert len(chunks) > 0

    retriever.index(chunks, embedder, "test_collection")
    results = retriever.query(
        "What do RAG systems use?",
        embedder,
        "test_collection",
        top_k=3,
    )
    assert len(results) > 0
    assert len(results) <= 3

    retriever.clear("test_collection")
    import shutil
    shutil.rmtree("./.test_cache", ignore_errors=True)


def test_runner_returns_run_result():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    retriever = Retriever(persist_dir="./.test_cache2")
    runner = StrategyRunner()
    chunker = RecursiveChunker(chunk_size=200)

    result = runner.run(
        chunker=chunker,
        documents=SAMPLE_DOCS,
        qa_pairs=SAMPLE_QA,
        embedder=embedder,
        retriever=retriever,
        top_k=3,
    )

    assert result.strategy_name == "recursive"
    assert result.chunk_count > 0
    assert len(result.retrieval_results) == 2

    import shutil
    shutil.rmtree("./.test_cache2", ignore_errors=True)


if __name__ == "__main__":
    print("Testing embedder...")
    test_embedder_local_model()
    print("Testing retriever...")
    test_retriever_index_and_query()
    print("Testing runner...")
    test_runner_returns_run_result()
    print("All pipeline tests passed!")
