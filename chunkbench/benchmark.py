from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from .chunkers import (
    ALL_STRATEGIES,
    STRATEGY_MAP,
    LateChunkingChunker,
    PropositionChunker,
    SemanticChunker,
)
from .evaluators import GenerationEvaluator, LatencyEvaluator, RetrievalEvaluator
from .loaders import DocumentLoader, QALoader
from .pipeline import Embedder, Retriever, RunResult, StrategyRunner
from .reporters import BenchmarkVisualizer, Leaderboard


@dataclass
class BenchmarkReport:
    leaderboard: Leaderboard
    run_results: Dict[str, RunResult]
    retrieval_metrics: Dict[str, dict]
    generation_metrics: Dict[str, dict]
    latency_metrics: Dict[str, dict]
    output_dir: str

    def show_leaderboard(self):
        print(self.leaderboard.to_markdown())

    def plot_all(self):
        from plotly.io import show as plotly_show

        viz = BenchmarkVisualizer(self.output_dir)
        plotly_show(viz.plot_leaderboard_bar(self.leaderboard))
        plotly_show(viz.plot_radar(self.leaderboard))
        plotly_show(viz.plot_scatter_tradeoff(self.leaderboard))

    def save_report(self):
        os.makedirs(self.output_dir, exist_ok=True)
        viz = BenchmarkVisualizer(self.output_dir)
        viz.plot_leaderboard_bar(self.leaderboard)
        viz.plot_radar(self.leaderboard)
        viz.plot_scatter_tradeoff(self.leaderboard)

        md_path = os.path.join(self.output_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ChunkBench Report\n\n")
            f.write(self.leaderboard.to_markdown())
            f.write("\n\n## Insights\n\n")
            for insight in self.leaderboard.get_insights():
                f.write(f"- {insight}\n")


class Benchmark:
    def __init__(
        self,
        documents: Union[str, List[str]],
        qa_pairs: Union[str, List[dict], None] = None,
        strategies: Optional[List[str]] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        output_dir: str = "./chunkbench_output",
        cache_dir: str = "./.chunkbench_cache",
        verbose: bool = True,
    ):
        self.strategies = strategies or ALL_STRATEGIES
        self.embed_model = embed_model
        self.top_k = top_k
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.verbose = verbose

        self._docs = self._resolve_documents(documents)
        self._qa = self._resolve_qa(qa_pairs)
        self._llm_client = self._init_llm(llm_api_key)
        self._llm_model = llm_model

    def run(self) -> BenchmarkReport:
        embedder = Embedder(model_name=self.embed_model)
        retriever = Retriever(persist_dir=self.cache_dir)
        runner = StrategyRunner()

        ret_evaluator = RetrievalEvaluator()
        gen_evaluator = GenerationEvaluator(
            llm_client=self._llm_client, llm_model=self._llm_model
        )
        lat_evaluator = LatencyEvaluator()

        retrieval_metrics: Dict[str, dict] = {}
        generation_metrics: Dict[str, dict] = {}
        latency_metrics: Dict[str, dict] = {}
        chunk_stats: Dict[str, dict] = {}
        run_results: Dict[str, RunResult] = {}

        strategy_iter = tqdm(self.strategies, desc="Benchmarking") if self.verbose else self.strategies

        for strategy_name in strategy_iter:
            if self.verbose:
                print(f"\n--- Running strategy: {strategy_name} ---")

            chunker = self._create_chunker(strategy_name, embedder)
            result = runner.run(
                chunker=chunker,
                documents=self._docs,
                qa_pairs=self._qa,
                embedder=embedder,
                retriever=retriever,
                top_k=self.top_k,
            )
            run_results[strategy_name] = result

            retrieval_metrics[strategy_name] = ret_evaluator.evaluate(result, self._qa)
            generation_metrics[strategy_name] = gen_evaluator.evaluate(result)
            latency_metrics[strategy_name] = lat_evaluator.evaluate(result)
            chunk_stats[strategy_name] = {
                "chunk_count": result.chunk_count,
                "avg_chunk_size": result.avg_chunk_size,
                "chunk_sizes": [len(c.content) for c in result.chunks],
            }

        lb = Leaderboard()
        lb.build(retrieval_metrics, generation_metrics, latency_metrics, chunk_stats)

        report = BenchmarkReport(
            leaderboard=lb,
            run_results=run_results,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_metrics=latency_metrics,
            output_dir=self.output_dir,
        )

        report.save_report()
        return report

    def run_single(self, strategy: str) -> RunResult:
        embedder = Embedder(model_name=self.embed_model)
        retriever = Retriever(persist_dir=self.cache_dir)
        runner = StrategyRunner()
        chunker = self._create_chunker(strategy, embedder)
        return runner.run(
            chunker=chunker,
            documents=self._docs,
            qa_pairs=self._qa,
            embedder=embedder,
            retriever=retriever,
            top_k=self.top_k,
        )

    def _create_chunker(self, name: str, embedder: Embedder):
        if name == "semantic":
            return SemanticChunker(embedder=embedder)
        if name == "proposition":
            return PropositionChunker(llm_client=self._llm_client, llm_model=self._llm_model)
        if name == "late_chunking":
            return LateChunkingChunker(embed_model=self.embed_model)
        cls = STRATEGY_MAP.get(name)
        if cls is None:
            raise ValueError(f"Unknown strategy: {name}")
        return cls()

    def _resolve_documents(self, source: Union[str, List[str]]) -> List[dict]:
        loader = DocumentLoader()
        docs = []
        paths = [source] if isinstance(source, str) else source
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for d in loader.load_dir(str(path)):
                    docs.append({"doc_id": d.doc_id, "text": d.text})
            else:
                d = loader.load_file(str(path))
                docs.append({"doc_id": d.doc_id, "text": d.text})
        return docs

    def _resolve_qa(self, source) -> List[dict]:
        if source is None:
            return []
        if isinstance(source, list):
            return source
        loader = QALoader()
        return loader.load(source)

    @staticmethod
    def _init_llm(api_key: Optional[str]):
        if not api_key:
            return None
        try:
            import openai

            return openai.OpenAI(api_key=api_key)
        except Exception:
            return None
