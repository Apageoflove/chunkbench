from __future__ import annotations

import os
import sys

import click
import yaml

from .benchmark import Benchmark
from .chunkers import ALL_STRATEGIES


@click.group()
def main():
    """ChunkBench — Benchmark every RAG chunking strategy, not just guess."""
    pass


@main.command()
@click.option("--docs", required=True, help="Document file or directory path")
@click.option("--qa", default=None, help="QA test set file (.jsonl/.json/.csv)")
@click.option("--strategies", "-s", multiple=True, help="Strategies to run")
@click.option("--embed-model", default="all-MiniLM-L6-v2", help="Embedding model")
@click.option("--top-k", default=5, type=int, help="Top-K for retrieval")
@click.option("--output", default="./chunkbench_output", help="Output directory")
@click.option("--cache", default="./.chunkbench_cache", help="Cache directory")
@click.option("--config", "config_path", default=None, help="YAML config file")
@click.option("--api-key", default=None, help="LLM API key (optional)")
@click.option("--verbose/--quiet", default=True)
def run(
    docs,
    qa,
    strategies,
    embed_model,
    top_k,
    output,
    cache,
    config_path,
    api_key,
    verbose,
):
    """Run benchmark on documents."""
    if config_path:
        cfg = _load_config(config_path)
        docs = cfg.get("documents", {}).get("paths", docs)
        if isinstance(docs, list):
            docs = docs[0] if len(docs) == 1 else docs
        qa = qa or cfg.get("qa", {}).get("path")
        strategies = strategies or cfg.get("strategies")
        embed_model = cfg.get("embedding", {}).get("model", embed_model)
        top_k = cfg.get("retrieval", {}).get("top_k", top_k)
        api_key = api_key or cfg.get("llm", {}).get("api_key")
        output = cfg.get("output", {}).get("dir", output)
        cache = cfg.get("cache", {}).get("dir", cache)

    strat_list = list(strategies) if strategies else None

    bench = Benchmark(
        documents=docs,
        qa_pairs=qa,
        strategies=strat_list,
        embed_model=embed_model,
        top_k=top_k,
        llm_api_key=api_key,
        output_dir=output,
        cache_dir=cache,
        verbose=verbose,
    )

    report = bench.run()

    if verbose:
        print("\n===== ChunkBench Leaderboard =====\n")
        report.show_leaderboard()
        print("\nInsights:")
        for insight in report.leaderboard.get_insights():
            print(f"  - {insight}")


@main.command("list-strategies")
def list_strategies():
    """List all supported chunking strategies."""
    for s in ALL_STRATEGIES:
        click.echo(f"  - {s}")


@main.command("show-results")
@click.option("--output", default="./chunkbench_output", help="Results directory")
def show_results(output):
    """Show cached benchmark results."""
    md_path = os.path.join(output, "report.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            click.echo(f.read())
    else:
        click.echo(f"No results found in {output}")


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
