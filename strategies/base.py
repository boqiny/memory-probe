"""
Base class for memory strategies.
Each strategy defines HOW conversation sessions get written to memory.
Retrieval is always the same (top-k cosine similarity) — only the write policy differs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import sys
sys.path.append("..")
from memory_store import MemoryStore
from data_loader import Session


class BaseStrategy(ABC):
    """
    A memory strategy controls what gets written to the memory store.
    
    - BasicRAG: stores raw conversation chunks
    - ExtractedFacts: uses LLM to extract structured facts (like A-MEM)
    - SummarizedEpisodes: uses LLM to summarize each session (like MemGPT)
    """

    name: str = "base"

    def __init__(self, store: MemoryStore):
        self.store = store

    @abstractmethod
    def ingest_session(self, session: Session) -> List[str]:
        """
        Process a conversation session and write entries to the memory store.
        
        Args:
            session: A conversation session with turns
            
        Returns:
            List of memory entry IDs that were created
        """
        pass

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve from memory. Same for all strategies."""
        return self.store.retrieve(query, top_k=top_k)

    def get_store(self) -> MemoryStore:
        return self.store
