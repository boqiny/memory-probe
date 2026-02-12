"""
Probe 2: Memory Utilization

Compares answers generated WITH vs WITHOUT memory retrieval.
If the answers are semantically identical, the agent didn't use its memory.

Metrics:
- Utilization Rate: % of questions where memory changed the answer
- Beneficial Utilization: % where memory changed the answer AND improved it
- Harmful Utilization: % where memory changed the answer AND worsened it
"""

from typing import Dict, Any
from llm_client import llm_call_json


UTILIZATION_JUDGE_PROMPT = """You are comparing two answers to the same question.

Question: {question}
Gold (correct) answer: {gold_answer}

Answer A (with memory): {answer_with}
Answer B (without memory): {answer_without}

Evaluate:
1. Are the two answers semantically the same? (i.e., they convey the same information, ignoring minor wording differences)
2. Is Answer A (with memory) correct or closer to the gold answer?
3. Is Answer B (without memory) correct or closer to the gold answer?

Return JSON:
{{
    "same_answer": true/false,
    "answer_with_correct": true/false,
    "answer_without_correct": true/false,
    "explanation": "brief explanation"
}}"""


class UtilizationProbe:
    """Measures whether retrieved memory actually influences the agent's answer."""

    def judge(
        self,
        question: str,
        gold_answer: str,
        answer_with_memory: str,
        answer_without_memory: str,
    ) -> Dict[str, Any]:
        """
        Compare with-memory vs without-memory answers.
        
        Returns classification:
        - "ignored": same answer either way (memory not used)
        - "beneficial": memory changed answer AND improved it
        - "harmful": memory changed answer AND worsened it
        - "neutral": memory changed answer but neither helped nor hurt
        """
        result = llm_call_json(
            UTILIZATION_JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                answer_with=answer_with_memory,
                answer_without=answer_without_memory,
            ),
            system="You are a strict evaluation judge. Respond with JSON only.",
        )

        same = result.get("same_answer", False)
        with_correct = result.get("answer_with_correct", False)
        without_correct = result.get("answer_without_correct", False)

        if same:
            category = "ignored"
        elif with_correct and not without_correct:
            category = "beneficial"
        elif not with_correct and without_correct:
            category = "harmful"
        elif with_correct and without_correct:
            category = "neutral"  # both correct, but different
        else:
            category = "neutral"  # both wrong, but different

        return {
            "same_answer": same,
            "answer_with_correct": with_correct,
            "answer_without_correct": without_correct,
            "category": category,
            "explanation": result.get("explanation", ""),
        }
