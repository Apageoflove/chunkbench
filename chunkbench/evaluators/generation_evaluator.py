from __future__ import annotations

from typing import List, Optional

from ..pipeline.runner import RunResult


class GenerationEvaluator:
    def __init__(
        self,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o-mini",
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.judge_model = judge_model

    def evaluate(self, run_result: RunResult) -> dict:
        if not run_result.retrieval_results:
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "rouge_l": 0.0,
                "evaluation_mode": "rouge_only",
            }

        rouge_scores: List[float] = []
        faithfulness_scores: List[float] = []
        relevance_scores: List[float] = []

        for rr in run_result.retrieval_results:
            context = "\n".join(c.content for c in rr["retrieved_chunks"])
            ref_answer = rr["reference_answer"]
            question = rr["question"]

            rouge = self._rouge_l(context, ref_answer)
            rouge_scores.append(rouge)

            if self.llm_client is not None:
                gen_answer = self._generate(context, question)
                faith = self._judge_faithfulness(context, gen_answer)
                rel = self._judge_relevance(question, gen_answer)
                faithfulness_scores.append(faith)
                relevance_scores.append(rel)

        result = {
            "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            "evaluation_mode": "llm" if self.llm_client else "rouge_only",
        }

        if self.llm_client and faithfulness_scores:
            result["faithfulness"] = sum(faithfulness_scores) / len(faithness_scores)
            result["relevance"] = sum(relevance_scores) / len(relevance_scores)
        else:
            result["faithfulness"] = result["rouge_l"]
            result["relevance"] = result["rouge_l"]

        return result

    def _generate(self, context: str, question: str) -> str:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the context above:"
        )
        try:
            if hasattr(self.llm_client, "chat"):
                resp = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content
        except Exception:
            pass
        return ""

    def _judge_faithfulness(self, context: str, answer: str) -> float:
        prompt = (
            f"Given the context: {context}\n"
            f"The answer is: {answer}\n"
            "Rate from 0 to 1 how faithful the answer is to the context "
            "(1 = completely grounded, 0 = hallucinated).\n"
            "Return only a float."
        )
        return self._call_judge(prompt)

    def _judge_relevance(self, question: str, answer: str) -> float:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Rate from 0 to 1 how relevant the answer is to the question.\n"
            "Return only a float."
        )
        return self._call_judge(prompt)

    def _call_judge(self, prompt: str) -> float:
        try:
            if hasattr(self.llm_client, "chat"):
                resp = self.llm_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = resp.choices[0].message.content.strip()
                return max(0.0, min(1.0, float(text)))
        except Exception:
            pass
        return 0.5

    @staticmethod
    def _rouge_l(hyp: str, ref: str) -> float:
        if not hyp or not ref:
            return 0.0
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        if not hyp_tokens or not ref_tokens:
            return 0.0

        m = len(hyp_tokens)
        n = len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        if lcs == 0:
            return 0.0
        precision = lcs / m
        recall = lcs / n
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
