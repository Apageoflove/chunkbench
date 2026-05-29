from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from .leaderboard import Leaderboard, LeaderboardEntry


class BenchmarkVisualizer:
    def __init__(self, output_dir: str = "./chunkbench_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_leaderboard_bar(self, leaderboard: Leaderboard) -> go.Figure:
        df = leaderboard.to_dataframe()
        metrics = ["Recall@5", "MRR", "NDCG@5", "Faithfulness", "Relevance"]
        fig = go.Figure()

        for metric in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df["Strategy"],
                    y=df[metric],
                    hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
                )
            )

        fig.update_layout(
            barmode="group",
            title="ChunkBench Strategy Comparison",
            xaxis_title="Strategy",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
        )
        self._save(fig, "leaderboard_bar")
        return fig

    def plot_radar(
        self,
        leaderboard: Leaderboard,
        strategies: Optional[List[str]] = None,
    ) -> go.Figure:
        df = leaderboard.to_dataframe()
        if strategies:
            df = df[df["Strategy"].isin(strategies)]

        dimensions = ["Recall@5", "MRR", "Faithfulness", "Relevance"]
        fig = go.Figure()

        for _, row in df.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[row[d] for d in dimensions],
                    theta=dimensions,
                    fill="toself",
                    name=row["Strategy"],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Strategy Radar Comparison",
            showlegend=True,
            template="plotly_white",
        )
        self._save(fig, "radar")
        return fig

    def plot_scatter_tradeoff(self, leaderboard: Leaderboard) -> go.Figure:
        df = leaderboard.to_dataframe()
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Avg Latency (ms)"],
                y=df["Recall@5"],
                mode="markers+text",
                text=df["Strategy"],
                textposition="top center",
                marker=dict(
                    size=df["Chunk Count"].clip(lower=10).tolist(),
                    sizemode="diameter",
                    opacity=0.7,
                    colorscale="Viridis",
                    color=df["Overall Score"],
                    showscale=True,
                    colorbar=dict(title="Overall Score"),
                ),
                hovertemplate=(
                    "%{text}<br>"
                    "Latency: %{x:.1f}ms<br>"
                    "Recall@5: %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Accuracy vs Speed Tradeoff",
            xaxis_title="Avg Retrieval Latency (ms)",
            yaxis_title="Recall@5",
            template="plotly_white",
        )
        self._save(fig, "scatter_tradeoff")
        return fig

    def plot_chunk_distribution(
        self, chunk_stats: Dict[str, dict]
    ) -> go.Figure:
        strategies = list(chunk_stats.keys())
        sizes = [chunk_stats[s].get("chunk_sizes", []) for s in strategies]

        fig = go.Figure()
        for strat, s_list in zip(strategies, sizes):
            if s_list:
                fig.add_trace(go.Box(y=s_list, name=strat))

        fig.update_layout(
            title="Chunk Size Distribution by Strategy",
            yaxis_title="Chunk Size (chars)",
            template="plotly_white",
        )
        self._save(fig, "chunk_distribution")
        return fig

    def plot_by_doc_type(
        self, results_by_type: Dict[str, List[LeaderboardEntry]]
    ) -> go.Figure:
        doc_types = list(results_by_type.keys())
        strategy_names = set()
        for entries in results_by_type.values():
            for e in entries:
                strategy_names.add(e.strategy_name)
        strategy_names = sorted(strategy_names)

        z_data = []
        for s in strategy_names:
            row = []
            for dt in doc_types:
                entries = results_by_type[dt]
                match = [e for e in entries if e.strategy_name == s]
                row.append(match[0].recall_at_5 if match else 0)
            z_data.append(row)

        fig = go.Figure(
            go.Heatmap(
                z=z_data,
                x=doc_types,
                y=strategy_names,
                colorscale="YlGnBu",
                text=[[f"{v:.3f}" for v in row] for row in z_data],
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title="Recall@5 by Document Type",
            xaxis_title="Document Type",
            yaxis_title="Strategy",
            template="plotly_white",
        )
        self._save(fig, "by_doc_type")
        return fig

    def _save(self, fig: go.Figure, name: str):
        path = os.path.join(self.output_dir, f"{name}.html")
        fig.write_html(path)
