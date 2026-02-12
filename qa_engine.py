"""
QA answering module.
Generates answers in two modes:
1. WITH memory retrieval (the agent uses its memory store)
2. WITHOUT memory retrieval (the agent answers from LLM knowledge only)

This paired generation is the core of the memory utilization probe.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional
from memory_store import MemoryStore
from llm_client import llm_call
from configs.settings import LLM_MODEL, RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# String-based answer evaluation (EM + token F1)
# ---------------------------------------------------------------------------

def normalize_answer(s) -> str:
    """Lower-case, strip, remove articles and punctuation."""
    # Convert to string first
    s = str(s).lower().strip()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction, gold) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def compute_token_f1(prediction: str, gold: str) -> Dict[str, float]:
    """Token-level precision, recall, F1 between prediction and gold.

    Raises ValueError if either input normalizes to an empty string,
    since that indicates a data problem (missing gold answer or empty
    model response) that should be fixed upstream, not masked.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        raise ValueError(
            f"Gold answer is empty after normalization. Raw gold: {gold!r}"
        )
    if not pred_norm:
        raise ValueError(
            f"Prediction is empty after normalization. Raw prediction: {prediction!r}"
        )
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


QA_WITH_MEMORY_PROMPT = """You are answering questions about a long-running conversation between two people. You have access to retrieved memory entries from past conversation sessions.

Retrieved memories:
{memories}

Question: {question}

Based on the retrieved memories, provide a concise and accurate answer. If the memories don't contain enough information, say so, but still try your best based on what's available. Keep the answer brief and factual."""


QA_WITHOUT_MEMORY_PROMPT = """You are answering questions about a conversation between two people. You do NOT have access to any conversation history or memory.

Question: {question}

Try your best to answer based on general knowledge. If you cannot answer without specific conversation context, say "I don't have enough information to answer this question." Keep the answer brief."""


def format_memories_for_prompt(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved memory entries for inclusion in the prompt."""
    if not retrieved:
        return "(No memories retrieved)"
    lines = []
    for i, item in enumerate(retrieved):
        entry = item["entry"]
        score = item["score"]
        meta = entry.metadata
        timestamp = meta.get("timestamp", "unknown date")
        lines.append(f"[Memory {i+1}] (relevance: {score:.3f}, date: {timestamp})\n{entry.content}")
    return "\n\n".join(lines)


def answer_with_memory(
    question: str,
    store: MemoryStore,
    top_k: int = RETRIEVAL_TOP_K,
    model: str = LLM_MODEL,
    retriever=None,
) -> Dict[str, Any]:
    """
    Answer a question using retrieved memories.
    If retriever is provided, use it; otherwise use store.retrieve() (cosine).
    Returns the answer AND the full retrieval trace.
    """
    # Retrieve
    if retriever is not None:
        retrieved = retriever.retrieve(question, top_k=top_k)
    else:
        retrieved = store.retrieve(question, top_k=top_k)
    memories_text = format_memories_for_prompt(retrieved)

    # Generate answer
    answer = llm_call(
        QA_WITH_MEMORY_PROMPT.format(memories=memories_text, question=question),
        system="You are a helpful assistant with access to conversation memories.",
        model=model,
    )

    return {
        "answer": answer.strip(),
        "retrieved_memories": retrieved,
        "retrieved_texts": [r["entry"].content for r in retrieved],
        "retrieval_scores": [r["score"] for r in retrieved],
        "memories_text": memories_text,
        "mode": "with_memory",
    }


def answer_without_memory(
    question: str,
    model: str = LLM_MODEL,
) -> Dict[str, Any]:
    """
    Answer a question WITHOUT any memory retrieval.
    This is the control condition for measuring memory utilization.
    """
    answer = llm_call(
        QA_WITHOUT_MEMORY_PROMPT.format(question=question),
        system="You are a helpful assistant. You have no access to any conversation history.",
        model=model,
    )

    return {
        "answer": answer.strip(),
        "retrieved_memories": [],
        "retrieved_texts": [],
        "retrieval_scores": [],
        "memories_text": "",
        "mode": "without_memory",
    }
