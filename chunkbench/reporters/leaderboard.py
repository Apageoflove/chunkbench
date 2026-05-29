from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class LeaderboardEntry:
    strategy_name: str
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    faithfulness: float
    relevance: float
    avg_retrieval_time_ms: float
    chunk_count: int
    avg_chunk_size: float
    overall_score: float = 0.0


class Leaderboard:
    def __init__(self):
        self.entries: List[LeaderboardEntry] = []

    def build(
        self,
        retrieval_metrics: Dict[str, dict],
        generation_metrics: Dict[str, dict],
        latency_metrics: Dict[str, dict],
        chunk_stats: Dict[str, dict],
    ) -> List[LeaderboardEntry]:
        all_latencies = [
            latency_metrics[s].get("avg_retrieval_time_ms", 0)
            for s in retrieval_metrics
        ]
        max_latency = max(all_latencies) if all_latencies else 1.0
        if max_latency == 0:
            max_latency = 1.0

        self.entries = []
        for strategy in retrieval_metrics:
            ret = retrieval_metrics[strategy]
            gen = generation_metrics.get(strategy, {})
            lat = latency_metrics.get(strategy, {})
            stats = chunk_stats.get(strategy, {})

            latency_score = 1 - (lat.get("avg_retrieval_time_ms", 0) / max_latency)
            overall = (
                ret.get("recall_at_5", 0) * 0.25
                + ret.get("mrr", 0) * 0.20
                + gen.get("faithfulness", 0) * 0.25
                + gen.get("relevance", 0) * 0.20
                + latency_score * 0.10
            )

            entry = LeaderboardEntry(
                strategy_name=strategy,
                recall_at_5=ret.get("recall_at_5", 0),
                mrr=ret.get("mrr", 0),
                ndcg_at_5=ret.get("ndcg_at_5", 0),
                faithfulness=gen.get("faithfulness", 0),
                relevance=gen.get("relevance", 0),
                avg_retrieval_time_ms=lat.get("avg_retrieval_time_ms", 0),
                chunk_count=stats.get("chunk_count", lat.get("chunk_count", 0)),
                avg_chunk_size=stats.get("avg_chunk_size", 0),
                overall_score=overall,
            )
            self.entries.append(entry)

        self.entries.sort(key=lambda e: e.overall_score, reverse=True)
        return self.entries

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "Strategy": e.strategy_name,
                "Recall@5": round(e.recall_at_5, 4),
                "MRR": round(e.mrr, 4),
                "NDCG@5": round(e.ndcg_at_5, 4),
                "Faithfulness": round(e.faithfulness, 4),
                "Relevance": round(e.relevance, 4),
                "Avg Latency (ms)": round(e.avg_retrieval_time_ms, 2),
                "Chunk Count": e.chunk_count,
                "Avg Chunk Size": round(e.avg_chunk_size, 1),
                "Overall Score": round(e.overall_score, 4),
            }
            for e in self.entries
        ]
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def get_winner(self) -> str:
        if self.entries:
            return self.entries[0].strategy_name
        return ""

    def get_insights(self) -> List[str]:
        if len(self.entries) < 2:
            return []

        insights: List[str] = []
        best = self.entries[0]
        worst = self.entries[-1]

        insights.append(
            f"{best.strategy_name} ranks #1 with Overall Score {best.overall_score:.4f}"
        )

        fastest = min(self.entries, key=lambda e: e.avg_retrieval_time_ms)
        slowest = max(self.entries, key=lambda e: e.avg_retrieval_time_ms)
        insights.append(
            f"{fastest.strategy_name} is the fastest ({fastest.avg_retrieval_time_ms:.1f}ms avg), "
            f"{slowest.strategy_name} is the slowest ({slowest.avg_retrieval_time_ms:.1f}ms)"
        )

        best_recall = max(self.entries, key=lambda e: e.recall_at_5)
        insights.append(
            f"{best_recall.strategy_name} achieves the highest Recall@5 ({best_recall.recall_at_5:.4f})"
        )

        if best.overall_score - worst.overall_score > 0.1:
            insights.append(
                f"Performance gap: {best.strategy_name} outperforms {worst.strategy_name} "
                f"by {best.overall_score - worst.overall_score:.4f} Overall Score"
            )

        return insights
