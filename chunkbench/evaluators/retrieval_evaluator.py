from __future__ import annotations

import math
from typing import List

from ..pipeline.runner import RunResult


class RetrievalEvaluator:
    def evaluate(self, run_result: RunResult, qa_pairs: List[dict]) -> dict:
        if not run_result.retrieval_results:
            return {
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0,
                "avg_chunk_rank": 0.0,
            }

        recall_1 = 0.0
        recall_3 = 0.0
        recall_5 = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        rank_sum = 0.0
        n = len(run_result.retrieval_results)

        for rr in run_result.retrieval_results:
            ref = rr["reference_answer"]
            retrieved = rr["retrieved_chunks"]
            hits = [self._is_hit(ref, c.content) for c in retrieved]

            first_hit = None
            for i, h in enumerate(hits):
                if h:
                    first_hit = i
                    break

            if first_hit is not None:
                mrr_sum += 1.0 / (first_hit + 1)
                rank_sum += first_hit + 1
            else:
                rank_sum += len(hits) + 1

            if any(hits[:1]):
                recall_1 += 1
            if any(hits[:3]):
                recall_3 += 1
            if any(hits[:5]):
                recall_5 += 1

            dcg = sum(
                1.0 / math.log2(i + 2) for i, h in enumerate(hits[:5]) if h
            )
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(1, 5)))
            ndcg_sum += dcg / idcg if idcg > 0 else 0

        return {
            "recall_at_1": recall_1 / n,
            "recall_at_3": recall_3 / n,
            "recall_at_5": recall_5 / n,
            "mrr": mrr_sum / n,
            "ndcg_at_5": ndcg_sum / n,
            "avg_chunk_rank": rank_sum / n,
        }

    @staticmethod
    def _is_hit(reference_answer: str, chunk_content: str) -> bool:
        if not reference_answer or not chunk_content:
            return False
        ref = reference_answer.strip()
        chunk = chunk_content.strip()
        if ref in chunk:
            return True
        window = 20
        for i in range(len(ref) - window + 1):
            if ref[i : i + window] in chunk:
                return True
        return False
