"""Hybrid retrieval: union of cosine + BM25 candidates, reranked by LLM."""

from typing import List, Dict, Any
from memory_store import MemoryStore
from retrieval.bm25_retriever import BM25Retriever
from llm_client import llm_call_json
from configs.settings import HYBRID_CANDIDATE_MULTIPLIER


RERANK_PROMPT = """You are ranking memory entries by relevance to a question.

Question: {question}

Candidate memories:
{candidates}

Return the IDs of the top {k} most relevant memories, ranked from most to least relevant.
Return JSON: {{"ranked_ids": ["id1", "id2", ...]}}

Only return IDs that are actually relevant. If fewer than {k} are relevant, return fewer."""


class HybridRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.bm25 = BM25Retriever(store)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        1. Get top-(k * multiplier) from cosine and BM25
        2. Merge and deduplicate
        3. LLM reranks to top-k
        """
        n_candidates = top_k * HYBRID_CANDIDATE_MULTIPLIER

        # Step 1: Get candidates from both methods
        cosine_results = self.store.retrieve(query, top_k=n_candidates)
        bm25_results = self.bm25.retrieve(query, top_k=n_candidates)

        # Step 2: Merge and deduplicate by entry ID
        seen = {}
        for r in cosine_results + bm25_results:
            eid = r["entry"].id
            if eid not in seen:
                seen[eid] = r
        candidates = list(seen.values())

        # If few enough candidates, skip reranking
        if len(candidates) <= top_k:
            for rank, r in enumerate(candidates):
                r["rank"] = rank
            return candidates

        # Step 3: LLM rerank
        candidate_text = "\n".join([
            f"[ID: {r['entry'].id}] {r['entry'].content[:200]}"
            for r in candidates
        ])

        try:
            result = llm_call_json(
                RERANK_PROMPT.format(
                    question=query,
                    candidates=candidate_text,
                    k=top_k,
                ),
                system="You are a precise relevance ranker. Return JSON only.",
            )
            ranked_ids = result.get("ranked_ids", [])

            # Build result list in ranked order
            id_to_result = {r["entry"].id: r for r in candidates}
            reranked = []
            for rank, eid in enumerate(ranked_ids[:top_k]):
                if eid in id_to_result:
                    r = id_to_result[eid]
                    r["rank"] = rank
                    reranked.append(r)

            # If LLM returned fewer than top_k, pad with remaining candidates
            if len(reranked) < top_k:
                used_ids = {r["entry"].id for r in reranked}
                for r in candidates:
                    if r["entry"].id not in used_ids and len(reranked) < top_k:
                        r["rank"] = len(reranked)
                        reranked.append(r)

            return reranked

        except Exception as e:
            # Fallback to cosine results if reranking fails
            print(f"    [Rerank failed: {e}, falling back to cosine]")
            return cosine_results[:top_k]
