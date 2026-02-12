"""
Thin wrapper around OpenAI API for LLM calls and embeddings.
"""

import os
import json
import time
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-3-small", verbose: bool = False) -> List[float]:
    """Get embedding for a single text string."""
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * 1536
    if verbose:
        print(f"[Embedding API call: {len(text)} chars]", end=" ", flush=True)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    cleaned = [t.replace("\n", " ").strip() or "empty" for t in texts]
    response = client.embeddings.create(input=cleaned, model=model)
    return [d.embedding for d in response.data]


def llm_call(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
    temperature: Optional[float] = None,
    max_tokens: int = 2048,
    json_mode: bool = False,
    max_retries: int = 1,
) -> str:
    """Single LLM call."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    _ = temperature  # intentionally ignored for compatibility with existing callers
    kwargs["max_completion_tokens"] = max_tokens

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def llm_call_json(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> dict:
    """LLM call that returns parsed JSON."""
    raw = llm_call(prompt, system=system, json_mode=True, **kwargs)
    return json.loads(raw)
