"""
Strategy 1: Basic RAG

Stores raw conversation turns as-is in the memory store.
Each turn (or small group of turns) becomes one memory entry.
No LLM processing at write time — cheapest strategy.
"""

from typing import List
from .base import BaseStrategy
from data_loader import Session, format_session_as_text


class BasicRAGStrategy(BaseStrategy):
    name = "basic_rag"

    def __init__(self, store, chunk_size: int = 3):
        """
        Args:
            store: MemoryStore instance
            chunk_size: Number of turns per memory chunk
        """
        super().__init__(store)
        self.chunk_size = chunk_size

    def ingest_session(self, session: Session) -> List[str]:
        """Store raw conversation chunks."""
        entry_ids = []
        turns = session.turns

        # Chunk turns into groups
        for i in range(0, len(turns), self.chunk_size):
            chunk_turns = turns[i : i + self.chunk_size]
            # Format as text
            text_lines = []
            if session.timestamp:
                text_lines.append(f"[Date: {session.timestamp}]")
            for t in chunk_turns:
                text_lines.append(f"{t.speaker}: {t.text}")
            content = "\n".join(text_lines)

            dia_ids = [t.dia_id for t in chunk_turns]
            entry_id = self.store.add(
                content=content,
                source_session=session.session_id,
                source_turns=dia_ids,
                strategy=self.name,
                metadata={
                    "speakers": list(set(t.speaker for t in chunk_turns)),
                    "timestamp": session.timestamp,
                },
            )
            entry_ids.append(entry_id)

        return entry_ids
