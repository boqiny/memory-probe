"""
Configuration for memory-probe experiments.
All hyperparameters in one place.
"""

# LLM Settings
LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 1.0

# Memory Settings
RETRIEVAL_TOP_K = 5
EMBEDDING_DIM = 1536  # text-embedding-3-small
HYBRID_CANDIDATE_MULTIPLIER = 3  # fetch top_k * this from each method before reranking

# Experiment Settings
MAX_CONVERSATIONS = 10  # LOCOMO has 10
MAX_QUESTIONS_PER_CONVERSATION = None  # None = all

# Probe Settings
SIMILARITY_THRESHOLD = 0.85  # for redundancy detection
UTILIZATION_MATCH_THRESHOLD = 0.9  # cosine sim between with/without-memory answers to count as "same"

# Concurrency
NUM_WORKERS = 5

# Paths
LOCOMO_DATA_PATH = "data/locomo10.json"
RESULTS_DIR = "results"
