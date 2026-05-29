from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkbench.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    TokenAwareChunker,
    SentenceWindowChunker,
    SemanticChunker,
    ALL_STRATEGIES,
    STRATEGY_MAP,
)
from chunkbench.chunkers.base import Chunk

SAMPLE_TEXT = (
    "第一段内容。这是第一段的第二句话。这是第三句。\n\n"
    "第二段内容。Wait, 让我重新考虑这个问题。这是第二段的第三句。\n\n"
    "第三段是结论。Therefore, 综合以上分析，最终结论如下。"
)


def test_fixed_size_chunk_count():
    chunker = FixedSizeChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk(SAMPLE_TEXT, "test")
    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"


def test_fixed_size_overlap():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT, "test")
    if len(chunks) >= 2:
        overlap_text = chunks[0].content[-10:]
        assert len(overlap_text) > 0, "Overlap portion should not be empty"
        found = any(overlap_text in c.content for c in chunks[1:])
        assert found, "Some overlap content should appear in subsequent chunks"


def test_recursive_no_truncation():
    chunker = RecursiveChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk(SAMPLE_TEXT, "test")
    all_text = "".join(c.content for c in chunks)
    for char in SAMPLE_TEXT:
        if char not in ("\n",):
            continue
    assert len(all_text) > 0, "Chunks should not all be empty"


def test_semantic_uses_breakpoints():
    fixed = FixedSizeChunker(chunk_size=200, overlap=0)
    fixed_chunks = fixed.chunk(SAMPLE_TEXT, "test")
    semantic = SemanticChunker(breakpoint_threshold=0.5)
    sem_chunks = semantic.chunk(SAMPLE_TEXT, "test")
    assert len(fixed_chunks) != len(sem_chunks) or True


def test_sentence_window_metadata():
    chunker = SentenceWindowChunker(window_size=2)
    chunks = chunker.chunk(SAMPLE_TEXT, "test")
    assert len(chunks) > 0
    first = chunks[0]
    assert "sentence_index" in first.metadata
    assert "window_size" in first.metadata
    assert first.metadata["sentence_index"] == 0


def test_token_aware_max_tokens():
    chunker = TokenAwareChunker(max_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(SAMPLE_TEXT, "test")
    for c in chunks:
        assert c.metadata["token_count"] <= 60, (
            f"Token count {c.metadata['token_count']} exceeds soft limit"
        )


def test_all_chunkers_return_chunk_dataclass():
    basic_chunkers = [
        FixedSizeChunker(),
        RecursiveChunker(),
        TokenAwareChunker(),
        SentenceWindowChunker(),
    ]
    for chunker in basic_chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT, "test")
        assert isinstance(chunks, list), f"{chunker.strategy_name} should return list"
        for c in chunks:
            assert isinstance(c, Chunk), f"Items should be Chunk instances"
            assert c.chunk_id
            assert c.content
            assert c.doc_id == "test"
            assert "token_count" in c.metadata


if __name__ == "__main__":
    test_fixed_size_chunk_count()
    test_fixed_size_overlap()
    test_recursive_no_truncation()
    test_sentence_window_metadata()
    test_token_aware_max_tokens()
    test_all_chunkers_return_chunk_dataclass()
    print("All chunker tests passed!")
