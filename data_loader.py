"""
Loader for the LOCOMO dataset.
Download locomo10.json from: https://github.com/snap-research/locomo
Place it in data/locomo10.json
"""

import json
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single dialogue turn."""
    dia_id: str
    speaker: str
    text: str
    session_id: str
    timestamp: Optional[str] = None


@dataclass
class Session:
    """A single conversation session."""
    session_id: str
    turns: List[Turn]
    timestamp: Optional[str] = None


@dataclass
class QAItem:
    """A single QA annotation."""
    question: str
    answer: str
    category: str  # single_hop, multi_hop, temporal, open_domain, adversarial
    evidence_ids: List[str] = field(default_factory=list)  # dia_ids that contain the answer


@dataclass
class Conversation:
    """A complete LOCOMO conversation with sessions and QA annotations."""
    conv_id: int
    speaker_a: str
    speaker_b: str
    sessions: List[Session]
    qa_items: List[QAItem]

    @property
    def all_turns(self) -> List[Turn]:
        turns = []
        for s in self.sessions:
            turns.extend(s.turns)
        return turns

    @property
    def total_tokens_approx(self) -> int:
        return sum(len(t.text.split()) for t in self.all_turns)


CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal_reasoning",
    4: "open_domain",
    5: "adversarial",
}


def _natural_session_sort_key(key: str) -> int:
    """Extract the numeric part of session_N for natural sorting."""
    import re
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else 0


def load_locomo(path: str = "data/locomo10.json") -> List[Conversation]:
    """Load and parse the LOCOMO dataset."""
    with open(path, "r") as f:
        raw = json.load(f)

    conversations = []
    for conv_idx, conv_data in enumerate(raw):
        # Sessions and speakers are nested under "conversation"
        conv_obj = conv_data.get("conversation", conv_data)

        speaker_a = conv_obj.get("speaker_a", "Speaker A")
        speaker_b = conv_obj.get("speaker_b", "Speaker B")

        # Parse sessions
        sessions = []
        session_keys = sorted(
            [k for k in conv_obj.keys() if k.startswith("session_") and not k.endswith("date_time")],
            key=_natural_session_sort_key,
        )
        for sk in session_keys:
            session_id = sk
            timestamp_key = f"{sk}_date_time"
            timestamp = conv_obj.get(timestamp_key, None)
            turns = []
            for turn_data in conv_obj[sk]:
                speaker = turn_data.get("speaker", turn_data.get("name", "Unknown"))
                text = turn_data.get("text", turn_data.get("content", ""))
                dia_id = turn_data.get("dia_id", "")
                turns.append(Turn(
                    dia_id=str(dia_id),
                    speaker=speaker,
                    text=text,
                    session_id=session_id,
                    timestamp=timestamp,
                ))
            sessions.append(Session(session_id=session_id, turns=turns, timestamp=timestamp))

        # Parse QA annotations
        qa_items = []
        qa_data = conv_data.get("qa", [])
        for qa in qa_data:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            raw_cat = qa.get("category", "unknown")
            category = CATEGORY_MAP.get(raw_cat, str(raw_cat))
            evidence = qa.get("evidence", [])
            if isinstance(evidence, list):
                evidence_ids = [str(e) for e in evidence]
            else:
                evidence_ids = []
            qa_items.append(QAItem(
                question=question,
                answer=answer,
                category=category,
                evidence_ids=evidence_ids,
            ))

        conversations.append(Conversation(
            conv_id=conv_idx,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_items=qa_items,
        ))

    return conversations


def format_session_as_text(session: Session) -> str:
    """Format a session into readable text."""
    lines = []
    if session.timestamp:
        lines.append(f"[{session.timestamp}]")
    for turn in session.turns:
        lines.append(f"{turn.speaker}: {turn.text}")
    return "\n".join(lines)


def format_conversation_as_text(conversation: Conversation) -> str:
    """Format an entire conversation into readable text."""
    parts = []
    for session in conversation.sessions:
        parts.append(format_session_as_text(session))
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    convos = load_locomo()
    print(f"Loaded {len(convos)} conversations")
    for c in convos:
        print(f"  Conv {c.conv_id}: {len(c.sessions)} sessions, "
              f"{len(c.all_turns)} turns, {len(c.qa_items)} QA items, "
              f"~{c.total_tokens_approx} tokens")
        cats = {}
        for qa in c.qa_items:
            cats[qa.category] = cats.get(qa.category, 0) + 1
        print(f"    QA categories: {cats}")
