"""
Simple in-memory vector store with embedding-based retrieval.
All three memory strategies write to and read from this same store.
"""

import uuid
import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from llm_client import get_embedding, get_embeddings_batch


@dataclass
class MemoryEntry:
    """A single memory entry in the store."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Tracking fields for probing
    source_session: Optional[str] = None  # which session this came from
    source_turns: Optional[List[str]] = None  # which dialogue turn IDs
    strategy: Optional[str] = None  # which strategy wrote this

    def __repr__(self):
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"MemoryEntry(id={self.id[:8]}, content='{preview}')"


class MemoryStore:
    """
    Vector store for memory entries.
    Supports: add, retrieve (top-k cosine), get_all, clear, stats.
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._dirty = True  # track when matrix needs rebuild

    def add(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        source_session: Optional[str] = None,
        source_turns: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Add a memory entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())[:8]
        if embedding is None:
            embedding = get_embedding(content)

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            source_session=source_session,
            source_turns=source_turns,
            strategy=strategy,
        )
        self.entries.append(entry)
        self._dirty = True
        return entry_id

    def add_batch(self, contents: List[str], **shared_kwargs) -> List[str]:
        """Add multiple entries at once (batches embedding calls)."""
        embeddings = get_embeddings_batch(contents)
        ids = []
        for content, emb in zip(contents, embeddings):
            entry_id = str(uuid.uuid4())[:8]
            entry = MemoryEntry(
                id=entry_id,
                content=content,
                embedding=emb,
                metadata=shared_kwargs.get("metadata", {}),
                source_session=shared_kwargs.get("source_session"),
                source_turns=shared_kwargs.get("source_turns"),
                strategy=shared_kwargs.get("strategy"),
            )
            self.entries.append(entry)
            ids.append(entry_id)
        self._dirty = True
        return ids

    def _rebuild_matrix(self):
        """Rebuild the numpy matrix for fast cosine similarity."""
        if not self.entries:
            self._embeddings_matrix = np.array([])
            self._dirty = False
            return
        self._embeddings_matrix = np.array([e.embedding for e in self.entries])
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings_matrix = self._embeddings_matrix / norms
        self._dirty = False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar memory entries for a query.
        Returns list of dicts with 'entry', 'score', 'rank'.
        """
        if not self.entries:
            return []
        if self._dirty:
            self._rebuild_matrix()

        query_emb = np.array(get_embedding(query))
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        scores = self._embeddings_matrix @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "entry": self.entries[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results

    def retrieve_with_query_embedding(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve using a pre-computed query embedding."""
        if not self.entries:
            return []
        if self._dirty:
            self._rebuild_matrix()

        query_emb = np.array(query_embedding)
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        scores = self._embeddings_matrix @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "entry": self.entries[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results

    def remove(self, entry_id: str) -> bool:
        """Remove a memory entry by ID. Returns True if found and removed."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(i)
                self._dirty = True
                return True
        return False

    def update(self, entry_id: str, new_content: str, new_metadata: Optional[Dict] = None) -> bool:
        """Update an existing memory entry's content and re-embed. Returns True if found."""
        for entry in self.entries:
            if entry.id == entry_id:
                entry.content = new_content
                entry.embedding = get_embedding(new_content)
                if new_metadata is not None:
                    entry.metadata.update(new_metadata)
                self._dirty = True
                return True
        return False

    def get_all(self) -> List[MemoryEntry]:
        return list(self.entries)

    def clear(self):
        self.entries = []
        self._embeddings_matrix = None
        self._dirty = True

    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict:
        return {
            "total_entries": len(self.entries),
            "sessions_covered": len(set(e.source_session for e in self.entries if e.source_session)),
            "strategy": self.entries[0].strategy if self.entries else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the store to a JSON-friendly dict."""
        return {
            "entries": [
                {
                    "id": e.id,
                    "content": e.content,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                    "source_session": e.source_session,
                    "source_turns": e.source_turns,
                    "strategy": e.strategy,
                }
                for e in self.entries
            ]
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        """Load entries from a dict produced by to_dict()."""
        entries = []
        for item in data.get("entries", []):
            entries.append(
                MemoryEntry(
                    id=item["id"],
                    content=item["content"],
                    embedding=item["embedding"],
                    metadata=item.get("metadata", {}) or {},
                    source_session=item.get("source_session"),
                    source_turns=item.get("source_turns"),
                    strategy=item.get("strategy"),
                )
            )
        self.entries = entries
        self._embeddings_matrix = None
        self._dirty = True

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.load_dict(data)
