# ChunkBench

English | [简体中文](./README_ZH.md)

> Benchmark every RAG chunking strategy, not just guess.

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Benchmark Results

![Leaderboard Table](assets/leaderboard_table.png)

![Strategy Comparison](assets/leaderboard_bar.png)

![Radar Comparison](assets/radar.png)

![Accuracy vs Speed](assets/scatter_tradeoff.png)

## Why ChunkBench

When building RAG applications, choosing the right chunking strategy is critical — but most teams pick one based on intuition. ChunkBench automates the comparison: given your documents and a QA test set, it runs all strategies and produces a leaderboard with retrieval accuracy, faithfulness, and latency metrics.

## Supported Strategies

| Strategy | Description | Best For |
|---|---|---|
| **Fixed Size** | Split by character count with overlap | Simple baseline, fast |
| **Recursive** | Split by separator hierarchy (paragraph → sentence → word) | General-purpose |
| **Token Aware** | Exact token count control at sentence boundaries | LLM context window |
| **Sentence Window** | Index single sentences, retrieve with context | Precise matching |
| **Semantic** | Split at semantic breakpoints using embeddings | Topical coherence |
| **Late Chunking** | Embed full document first, then chunk | Global context preservation |
| **Proposition** | Decompose into atomic facts via LLM | Maximum precision |

## Architecture

```
Documents → [Loader] → Text + Metadata
                          ↓
              [Chunker × 7] → List[Chunk] per strategy
                          ↓
              [Embedder] → Vector representations
                          ↓
              [Retriever (ChromaDB)] → Top-K chunks
                          ↓
              [Evaluators] → Recall@K / MRR / Faithfulness / Latency
                          ↓
              [Leaderboard + Visualizer] → Comparison report
```

## Installation

```bash
git clone https://github.com/Apageoflove/chunkbench.git
cd chunkbench
pip install -e .
```

> No API key required. Embedding runs locally with `all-MiniLM-L6-v2`.

## Quick Start

### Python API

```python
from chunkbench.benchmark import Benchmark

bench = Benchmark(
    documents="./examples/sample_docs/",
    qa_pairs="./examples/sample_qa.jsonl",
    strategies=["fixed_size", "recursive", "semantic", "sentence_window"],
)

report = bench.run()
report.show_leaderboard()
```

### CLI

```bash
chunkbench run --docs ./examples/sample_docs/ --qa ./examples/sample_qa.jsonl
chunkbench run --config examples/config_example.yaml
chunkbench list-strategies
```

### Gradio Demo

```bash
python demo/app.py
# Open http://localhost:7860
```

## Metrics

| Metric | Description | Computation |
|---|---|---|
| **Recall@K** | Top-K retrieval hit rate | Relevant docs found in top K / total relevant |
| **MRR** | Mean Reciprocal Rank | Average of 1/rank of first relevant result |
| **NDCG@5** | Normalized discounted gain | DCG / ideal DCG at position 5 |
| **Faithfulness** | Answer grounded in context | LLM-as-judge or ROUGE-L (fallback) |
| **Relevance** | Answer addresses question | LLM-as-judge or ROUGE-L (fallback) |
| **Overall Score** | Weighted combination | R@5×0.25 + MRR×0.20 + Faith×0.25 + Rel×0.20 + Latency×0.10 |

## No API Key Required

ChunkBench works fully without any API key:

- **Faithfulness & Relevance** fall back to ROUGE-L scoring
- **Proposition Chunker** falls back to sentence-level splitting
- All other strategies run normally

To enable LLM features:

```bash
export OPENAI_API_KEY=sk-xxx
```

## Project Structure

```
chunkbench/
├── chunkbench/
│   ├── chunkers/          # 7 chunking strategies
│   ├── pipeline/          # Embedding, Retrieval, Runner
│   ├── evaluators/        # Retrieval, Generation, Latency
│   ├── loaders/           # Document & QA loading
│   ├── reporters/         # Leaderboard & Visualization
│   ├── benchmark.py       # Main entry
│   └── cli.py             # CLI interface
├── demo/                  # Gradio web demo
├── examples/              # Sample data & configs
├── tests/                 # Test suite
└── requirements.txt
```

## Roadmap

- [ ] Support more embedding models (Cohere, Voyage)
- [ ] BEIR dataset auto-download
- [ ] Custom strategy plugin registration
- [ ] Multi-language support
- [ ] Distributed benchmarking

## License

[MIT](https://opensource.org/license/mit/)
