"""BM25 keyword-based retrieval over memory entries."""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from memory_store import MemoryStore


class BM25Retriever:
    def __init__(self, store: MemoryStore):
        self.store = store
        self._index = None
        self._indexed_size = 0

    def _build_index(self):
        """Tokenize all memory entries and build BM25 index."""
        entries = self.store.get_all()
        tokenized = [entry.content.lower().split() for entry in entries]
        self._index = BM25Okapi(tokenized) if tokenized else None
        self._indexed_size = len(entries)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k by BM25 score."""
        entries = self.store.get_all()
        if not entries:
            return []

        # Rebuild if store changed since last index
        if self._index is None or len(entries) != self._indexed_size:
            self._build_index()
        if self._index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "entry": entries[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results
