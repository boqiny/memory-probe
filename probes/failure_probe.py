"""
Probe 3: Failure Analysis

Identifies and classifies failure modes:
- "correct_memory_wrong_answer": Agent had the right info but failed to use it
- "irrelevant_retrieval": Agent retrieved junk and was misled
- "no_relevant_memory": Memory store didn't contain the needed info
- "hallucination_despite_memory": Agent's answer contradicts its own retrieved memory
- "correct": Agent got it right

This probe generates the cases for human expert annotation.
"""

from typing import Dict, Any, List
from llm_client import llm_call_json


FAILURE_CLASSIFY_PROMPT = """You are analyzing a memory-augmented QA system's failure.

Question: {question}
Gold (correct) answer: {gold_answer}
System's answer: {system_answer}
Was the system's answer correct? {is_correct}

Retrieved memories (these were provided to the system):
{retrieved_memories}

Relevance of each memory:
{relevance_judgments}

Classify this case into ONE of these categories:
- "correct": The system answered correctly.
- "correct_memory_wrong_answer": At least one relevant memory was retrieved, but the system still got the answer wrong. The information was THERE but the system failed to use it properly.
- "hallucination_despite_memory": The system's answer directly CONTRADICTS information in the retrieved memories.
- "irrelevant_retrieval": No relevant memories were retrieved, AND the system got it wrong. The retrieval system failed.
- "no_memory_available": The memory store likely doesn't contain the needed information at all.
- "partial_memory": Some relevant info was retrieved but it was incomplete — the system needed more.

Return JSON:
{{
    "failure_category": "one of the categories above",
    "explanation": "1-2 sentence explanation of what went wrong",
    "key_evidence": "the specific memory content that was relevant but ignored, or the contradiction, if applicable"
}}"""


class FailureProbe:
    """Classifies failure modes for cases where the agent got the answer wrong."""

    def classify(
        self,
        question: str,
        gold_answer: str,
        system_answer: str,
        is_correct: bool,
        retrieved_texts: List[str],
        relevance_judgments: List[Dict],
    ) -> Dict[str, Any]:
        """Classify the failure mode for a single QA instance."""
        
        if is_correct:
            return {
                "failure_category": "correct",
                "explanation": "System answered correctly.",
                "key_evidence": "",
            }

        # Format retrieved memories
        mem_lines = []
        for i, text in enumerate(retrieved_texts):
            mem_lines.append(f"[Memory {i+1}]: {text}")
        mem_text = "\n".join(mem_lines) if mem_lines else "(No memories retrieved)"

        # Format relevance judgments
        rel_lines = []
        for i, j in enumerate(relevance_judgments):
            rel_lines.append(f"Memory {i+1}: {'RELEVANT' if j.get('relevant') else 'NOT RELEVANT'} — {j.get('reason', '')}")
        rel_text = "\n".join(rel_lines) if rel_lines else "(No judgments)"

        result = llm_call_json(
            FAILURE_CLASSIFY_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                system_answer=system_answer,
                is_correct=is_correct,
                retrieved_memories=mem_text,
                relevance_judgments=rel_text,
            ),
            system="You are a precise failure analysis system. Respond with JSON only.",
        )

        return {
            "failure_category": result.get("failure_category", "unknown"),
            "explanation": result.get("explanation", ""),
            "key_evidence": result.get("key_evidence", ""),
        }
