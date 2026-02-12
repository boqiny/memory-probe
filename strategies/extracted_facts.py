"""
Strategy 2: Extracted Facts
Mimics: A-MEM / Mem0

For each session, uses LLM to extract structured facts:
- Key events, preferences, facts about each speaker
- Each fact stored as a separate memory entry with keywords/tags

This mirrors A-MEM's analyze_content + MemoryNote approach:
the LLM extracts keywords, context, and tags from content,
then stores enriched memory entries that can be evolved.

Includes conflict resolution: before adding a new fact, retrieve similar
existing memories and use an LLM to decide ADD / UPDATE / NOOP.
"""

from typing import List, Dict
from .base import BaseStrategy
from data_loader import Session, format_session_as_text
from llm_client import llm_call_json


EXTRACTION_PROMPT = """You are a memory extraction agent. Given a conversation session between two people, extract ALL important facts, events, preferences, and details mentioned.

For each fact, provide:
- "fact": A concise, self-contained statement (should make sense without the original context)
- "speakers": Which speaker(s) this fact is about
- "type": One of [event, preference, relationship, plan, personal_detail, opinion]

Conversation:
{conversation}

Return a JSON object with key "facts" containing a list of extracted facts.
Example format:
{{"facts": [{{"fact": "Alice adopted a golden retriever named Max in March", "speakers": ["Alice"], "type": "event"}}, ...]}}

Extract every meaningful piece of information. Be thorough — missing a fact means it's lost forever."""


CONFLICT_RESOLUTION_PROMPT = """You are a memory management system. A new fact has been extracted from a conversation. You must decide how to handle it relative to existing memories.

New fact: {new_fact}

Existing similar memories:
{existing_memories}

Decide one of:
- ADD: The new fact contains genuinely new information not covered by existing memories.
- UPDATE: The new fact updates/supersedes one of the existing memories (provide the ID to update).
- NOOP: The new fact is redundant — the information is already fully captured.

Return JSON: {{"action": "ADD" | "UPDATE" | "NOOP", "target_id": "<id of memory to update, only if UPDATE>", "reason": "brief explanation"}}"""


class ExtractedFactsStrategy(BaseStrategy):
    name = "extracted_facts"

    SIMILARITY_THRESHOLD = 0.7
    CONFLICT_TOP_K = 3

    def __init__(self, store):
        super().__init__(store)
        self._conflict_stats: Dict[str, int] = {
            "add": 0,
            "update": 0,
            "noop": 0,
            "no_conflict": 0,
        }

    def get_conflict_stats(self) -> Dict[str, int]:
        """Return conflict resolution statistics."""
        return dict(self._conflict_stats)

    def _resolve_conflict(self, new_fact: str, session: Session, dia_ids: List[str], speakers: List, fact_type: str) -> str:
        """
        Check new fact against existing memories. Returns the entry_id
        of the created/updated entry, or empty string for NOOP.
        """
        # No conflict possible if store is empty
        if self.store.size() == 0:
            self._conflict_stats["no_conflict"] += 1
            return self._add_fact(new_fact, session, dia_ids, speakers, fact_type)

        # Retrieve top-k similar existing memories
        similar = self.store.retrieve(new_fact, top_k=self.CONFLICT_TOP_K)

        # Check if any exceed the similarity threshold
        conflicts = [s for s in similar if s["score"] >= self.SIMILARITY_THRESHOLD]

        if not conflicts:
            self._conflict_stats["no_conflict"] += 1
            return self._add_fact(new_fact, session, dia_ids, speakers, fact_type)

        # Format existing memories for the LLM
        existing_lines = []
        for item in conflicts:
            entry = item["entry"]
            existing_lines.append(
                f"[ID: {entry.id}] (similarity: {item['score']:.3f}) {entry.content}"
            )
        existing_text = "\n".join(existing_lines)

        # LLM decides
        result = llm_call_json(
            CONFLICT_RESOLUTION_PROMPT.format(
                new_fact=new_fact,
                existing_memories=existing_text,
            ),
            system="You are a memory conflict resolver. Respond with JSON only.",
        )

        action = result.get("action", "ADD").upper()

        if action == "UPDATE":
            target_id = result.get("target_id", "")
            if target_id:
                updated = self.store.update(
                    target_id,
                    new_fact,
                    new_metadata={
                        "speakers": speakers,
                        "fact_type": fact_type,
                        "timestamp": session.timestamp,
                        "updated_from_session": session.session_id,
                    },
                )
                if updated:
                    self._conflict_stats["update"] += 1
                    return target_id
            # If update failed (bad ID), fall through to ADD
            self._conflict_stats["add"] += 1
            return self._add_fact(new_fact, session, dia_ids, speakers, fact_type)

        elif action == "NOOP":
            self._conflict_stats["noop"] += 1
            return ""

        else:  # ADD or unknown
            self._conflict_stats["add"] += 1
            return self._add_fact(new_fact, session, dia_ids, speakers, fact_type)

    def _add_fact(self, fact_text: str, session: Session, dia_ids: List[str], speakers: List, fact_type: str) -> str:
        """Add a single fact to the store."""
        return self.store.add(
            content=fact_text,
            source_session=session.session_id,
            source_turns=dia_ids,
            strategy=self.name,
            metadata={
                "speakers": speakers,
                "fact_type": fact_type,
                "timestamp": session.timestamp,
            },
        )

    def ingest_session(self, session: Session) -> List[str]:
        """Extract structured facts from session using LLM, with conflict resolution."""
        session_text = format_session_as_text(session)

        # LLM extracts facts
        result = llm_call_json(
            EXTRACTION_PROMPT.format(conversation=session_text),
            system="You are a precise information extraction system. Always respond with valid JSON.",
        )

        facts = result.get("facts", [])
        entry_ids = []

        for fact_obj in facts:
            fact_text = fact_obj.get("fact", "")
            if not fact_text.strip():
                continue

            speakers = fact_obj.get("speakers", [])
            fact_type = fact_obj.get("type", "unknown")
            dia_ids = [t.dia_id for t in session.turns]

            entry_id = self._resolve_conflict(
                fact_text, session, dia_ids, speakers, fact_type,
            )
            if entry_id:  # empty string means NOOP
                entry_ids.append(entry_id)

        return entry_ids
