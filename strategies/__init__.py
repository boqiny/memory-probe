from .verbatim_rag import BasicRAGStrategy
from .extracted_facts import ExtractedFactsStrategy
from .summarized_episodes import SummarizedEpisodesStrategy

ALL_STRATEGIES = {
    "basic_rag": BasicRAGStrategy,
    "extracted_facts": ExtractedFactsStrategy,
    "summarized_episodes": SummarizedEpisodesStrategy,
}
