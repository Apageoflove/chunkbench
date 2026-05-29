from __future__ import annotations

from typing import List

from ..pipeline.runner import RunResult


class LatencyEvaluator:
    def evaluate(self, run_result: RunResult) -> dict:
        retrieval_times: List[float] = []
        for rr in run_result.retrieval_results:
            retrieval_times.append(rr.get("retrieval_time", 0))

        avg_ms = 0.0
        p95_ms = 0.0
        throughput = 0.0

        if retrieval_times:
            avg_ms = sum(retrieval_times) / len(retrieval_times) * 1000
            sorted_t = sorted(retrieval_times)
            p95_idx = max(0, int(len(sorted_t) * 0.95) - 1)
            p95_ms = sorted_t[p95_idx] * 1000
            total_time = sum(retrieval_times)
            throughput = len(retrieval_times) / total_time if total_time > 0 else 0

        return {
            "index_time_s": run_result.index_time,
            "avg_retrieval_time_ms": avg_ms,
            "p95_retrieval_time_ms": p95_ms,
            "chunk_count": run_result.chunk_count,
            "throughput_qps": throughput,
        }
