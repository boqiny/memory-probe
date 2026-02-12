"""
Probe 1: Retrieval Relevance

For each retrieved memory entry, judge whether it contains information
relevant to answering the question.

Metric: Retrieval Precision@K = (# relevant retrieved) / K
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from llm_client import llm_call_json


RELEVANCE_JUDGE_PROMPT = """You are evaluating whether a retrieved memory entry is relevant to answering a question.

Question: {question}
Gold answer: {gold_answer}

Memory entry: {memory_content}

Is this memory entry relevant to answering the question? A memory is relevant if it contains information that could directly help produce the correct answer.

Return JSON: {{"relevant": true/false, "reason": "brief explanation"}}"""


class RelevanceProbe:
    """Measures what fraction of retrieved memories are actually relevant."""

    def judge_single(
        self, question: str, gold_answer: str, memory_content: str
    ) -> Dict[str, Any]:
        """Judge relevance of a single memory entry."""
        result = llm_call_json(
            RELEVANCE_JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                memory_content=memory_content,
            ),
            system="You are a strict relevance judge. Respond with JSON only.",
        )
        return {
            "relevant": result.get("relevant", False),
            "reason": result.get("reason", ""),
        }

    def judge_batch(
        self,
        question: str,
        gold_answer: str,
        retrieved_texts: List[str],
    ) -> Dict[str, Any]:
        """Judge relevance of all retrieved memories for one question (concurrent)."""
        if not retrieved_texts:
            return {"judgments": [], "n_relevant": 0, "n_retrieved": 0, "precision": 0.0}

        def _judge(mem_text):
            return self.judge_single(question, gold_answer, mem_text)

        with ThreadPoolExecutor(max_workers=len(retrieved_texts)) as pool:
            judgments = list(pool.map(_judge, retrieved_texts))

        n_relevant = sum(1 for j in judgments if j["relevant"])
        precision = n_relevant / len(judgments)

        return {
            "judgments": judgments,
            "n_relevant": n_relevant,
            "n_retrieved": len(judgments),
            "precision": precision,
        }
