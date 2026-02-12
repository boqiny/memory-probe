"""
Strategy 3: Summarized Episodes
Mimics: MemGPT's summarize_messages / recursive summarization

For each session, generates a structured summary capturing:
- Key topics discussed
- Important events/updates for each speaker
- Timeline/temporal markers

This mirrors MemGPT's approach where conversation history is
periodically summarized to fit within the context window,
and summaries are stored in archival memory for later retrieval.
"""

from typing import List
from .base import BaseStrategy
from data_loader import Session, format_session_as_text
from llm_client import llm_call


SUMMARY_PROMPT = """You are a memory summarization agent. Summarize the following conversation session into a detailed but concise summary.

Your summary MUST capture:
1. All key events, updates, or changes in either speaker's life
2. Any plans, commitments, or intentions mentioned
3. Preferences, opinions, or emotional states expressed
4. Temporal markers (dates, "last week", "tomorrow", etc.)
5. Any references to previous conversations or shared history

Write the summary as a coherent paragraph. Include speaker names. Do NOT omit details — every piece of information matters for future recall.

Conversation session ({timestamp}):
{conversation}

Summary:"""


class SummarizedEpisodesStrategy(BaseStrategy):
    name = "summarized_episodes"

    def ingest_session(self, session: Session) -> List[str]:
        """Summarize the session and store as a single memory entry."""
        session_text = format_session_as_text(session)
        timestamp = session.timestamp or "unknown date"

        summary = llm_call(
            SUMMARY_PROMPT.format(conversation=session_text, timestamp=timestamp),
            system="You are a precise summarization system. Capture all important details.",
        )

        dia_ids = [t.dia_id for t in session.turns]
        speakers = list(set(t.speaker for t in session.turns))

        entry_id = self.store.add(
            content=summary.strip(),
            source_session=session.session_id,
            source_turns=dia_ids,
            strategy=self.name,
            metadata={
                "speakers": speakers,
                "timestamp": timestamp,
                "type": "session_summary",
            },
        )

        return [entry_id]
