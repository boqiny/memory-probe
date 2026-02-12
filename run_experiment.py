"""
Main experiment runner for memory-probe.

For each memory strategy × each LOCOMO conversation × each QA question:
1. Ingest conversation sessions into memory store
2. Answer each question WITH memory
3. Answer each question WITHOUT memory (control)
4. Run all three probes
5. Save complete traces for human annotation
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from data_loader import load_locomo, Conversation
from memory_store import MemoryStore
from strategies import ALL_STRATEGIES
from qa_engine import answer_with_memory, answer_without_memory, compute_exact_match, compute_token_f1
from probes import RelevanceProbe, UtilizationProbe, FailureProbe
from configs.settings import (
    LOCOMO_DATA_PATH,
    RESULTS_DIR,
    RETRIEVAL_TOP_K,
    MAX_CONVERSATIONS,
    MAX_QUESTIONS_PER_CONVERSATION,
    EMBEDDING_MODEL,
    NUM_WORKERS,
)

VALID_RETRIEVAL_METHODS = ["cosine", "bm25", "hybrid"]


def _make_retriever(store, method: str):
    """Create a retriever for the given method, or None for cosine (default)."""
    if method == "cosine":
        return None  # answer_with_memory uses store.retrieve() directly
    elif method == "bm25":
        from retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(store)
    elif method == "hybrid":
        from retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(store)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

STRATEGY_VERSION = "2.0"  # Bump when strategy logic changes to invalidate caches


def _ingest_cache_path(strategy_name: str, conversation: Conversation, strategy_obj: Any) -> str:
    chunk_size = getattr(strategy_obj, "chunk_size", None)
    chunk_part = f"_chunk{chunk_size}" if chunk_size is not None else ""
    fname = f"memory_{strategy_name}_conv{conversation.conv_id}{chunk_part}.json"
    return os.path.join(RESULTS_DIR, "cache", fname)


def _ingest_conversation(
    strategy_name: str,
    conversation: Conversation,
) -> Tuple[MemoryStore, List[Dict[str, Any]]]:
    """
    Ingest a conversation into a memory store (with caching).
    Separated from QA so that top-k ablation can reuse the same store.
    """
    store = MemoryStore()
    StrategyClass = ALL_STRATEGIES[strategy_name]
    strategy = StrategyClass(store)

    cache_path = _ingest_cache_path(strategy_name, conversation, strategy)
    ingest_log: List[Dict[str, Any]] = []

    if os.path.exists(cache_path):
        print(f"    Loading cached memory store from {cache_path} ...", flush=True)
        with open(cache_path, "r") as f:
            cached = json.load(f)
        meta = cached.get("meta", {})
        if (
            meta.get("conv_id") == conversation.conv_id
            and meta.get("strategy") == strategy_name
            and meta.get("strategy_version") == STRATEGY_VERSION
        ):
            store.load_dict(cached.get("store", {}))
            ingest_log = cached.get("ingest_log", [])
            print(f"    Loaded cache: {store.size()} entries", flush=True)
        else:
            reason = "version mismatch" if meta.get("strategy_version") != STRATEGY_VERSION else "metadata mismatch"
            print(f"    Cache {reason}; rebuilding ingestion.", flush=True)

    if store.size() == 0:
        print(f"    Ingesting {len(conversation.sessions)} sessions into memory...", flush=True)
        for idx, session in enumerate(conversation.sessions, 1):
            print(
                f"      [{idx}/{len(conversation.sessions)}] Processing {session.session_id} ({len(session.turns)} turns)...",
                end=" ",
                flush=True,
            )
            entry_ids = strategy.ingest_session(session)
            print(f"created {len(entry_ids)} entries", flush=True)
            ingest_log.append(
                {
                    "session_id": session.session_id,
                    "entries_created": len(entry_ids),
                    "entry_ids": entry_ids,
                }
            )

        print(f"    Memory store: {store.size()} entries after ingestion", flush=True)
        # Capture conflict stats if the strategy supports it
        if hasattr(strategy, "get_conflict_stats"):
            conflict_stats = strategy.get_conflict_stats()
            for log_entry in ingest_log:
                log_entry["conflict_stats"] = conflict_stats

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "meta": {
                        "conv_id": conversation.conv_id,
                        "strategy": strategy_name,
                        "strategy_version": STRATEGY_VERSION,
                        "embedding_model": EMBEDDING_MODEL,
                        "created_at": datetime.now().isoformat(),
                    },
                    "store": store.to_dict(),
                    "ingest_log": ingest_log,
                },
                f,
            )
        print(f"    Saved ingestion cache to {cache_path}", flush=True)

    return store, ingest_log


def _without_memory_cache_path(conversation: Conversation) -> str:
    return os.path.join(RESULTS_DIR, "cache", f"without_memory_conv{conversation.conv_id}.json")


def _get_without_memory_answers(
    conversation: Conversation,
    max_questions: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
) -> Dict[int, Dict[str, Any]]:
    """
    Generate 'without memory' answers for each non-adversarial question.
    Returns a dict keyed by question index.
    Cached to disk so results persist across runs.
    """
    cache_path = _without_memory_cache_path(conversation)

    # Try loading from disk
    if os.path.exists(cache_path):
        print(f"    Loading cached without-memory answers from {cache_path}", flush=True)
        with open(cache_path, "r") as f:
            raw = json.load(f)
        # Keys are strings in JSON; convert back to int
        return {int(k): v for k, v in raw.items()}

    qa_items = conversation.qa_items
    limit = max_questions if max_questions is not None else MAX_QUESTIONS_PER_CONVERSATION
    if limit:
        qa_items = qa_items[:limit]

    work = [(qi, qa) for qi, qa in enumerate(qa_items) if qa.category != "adversarial"]

    cache: Dict[int, Dict[str, Any]] = {}

    def _do(qi_qa):
        qi, qa = qi_qa
        return qi, answer_without_memory(qa.question)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_do, item): item for item in work}
        with tqdm(total=len(work), desc="      Without-memory baseline", leave=False) as pbar:
            for future in as_completed(futures):
                qi, result = future.result()
                cache[qi] = result
                pbar.update(1)

    # Persist to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"    Saved without-memory cache to {cache_path}", flush=True)

    return cache


def _process_question(
    qi: int,
    qa: Any,
    store: MemoryStore,
    top_k: int,
    strategy_name: str,
    conversation: Conversation,
    without_memory_cache: Optional[Dict[int, Dict[str, Any]]],
    run_probes: bool,
    retrieval_method: str = "cosine",
    retriever=None,
) -> Dict[str, Any]:
    """Process a single question end-to-end. Thread-safe (store is read-only)."""

    # Answer WITH memory
    with_result = answer_with_memory(qa.question, store, top_k=top_k, retriever=retriever)

    # Answer WITHOUT memory (reuse cache if available)
    if without_memory_cache is not None and qi in without_memory_cache:
        without_result = without_memory_cache[qi]
    else:
        without_result = answer_without_memory(qa.question)

    # String-based metrics (EM + F1)
    em_with = compute_exact_match(with_result["answer"], qa.answer)
    f1_with = compute_token_f1(with_result["answer"], qa.answer)
    em_without = compute_exact_match(without_result["answer"], qa.answer)
    f1_without = compute_token_f1(without_result["answer"], qa.answer)

    record = {
        "conv_id": conversation.conv_id,
        "question_idx": qi,
        "question": qa.question,
        "gold_answer": qa.answer,
        "category": qa.category,
        "evidence_ids": qa.evidence_ids,
        "strategy": strategy_name,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        # Answers
        "answer_with_memory": with_result["answer"],
        "answer_without_memory": without_result["answer"],
        # String-based metrics
        "em_with": em_with,
        "f1_with": f1_with["f1"],
        "precision_with": f1_with["precision"],
        "recall_with": f1_with["recall"],
        "em_without": em_without,
        "f1_without": f1_without["f1"],
        # Retrieval trace
        "retrieved_texts": with_result["retrieved_texts"],
        "retrieval_scores": with_result["retrieval_scores"],
        "n_retrieved": len(with_result["retrieved_texts"]),
        # Memory store stats
        "memory_store_size": store.size(),
    }

    # Run probes
    if run_probes:
        relevance_probe = RelevanceProbe()
        utilization_probe = UtilizationProbe()
        failure_probe = FailureProbe()

        # Probe 1: Retrieval Relevance
        rel_result = relevance_probe.judge_batch(
            qa.question, qa.answer, with_result["retrieved_texts"]
        )
        record["retrieval_precision"] = rel_result["precision"]
        record["n_relevant_retrieved"] = rel_result["n_relevant"]
        record["relevance_judgments"] = rel_result["judgments"]

        # Probe 2: Utilization
        util_result = utilization_probe.judge(
            qa.question,
            qa.answer,
            with_result["answer"],
            without_result["answer"],
        )
        record["utilization_category"] = util_result["category"]
        record["utilization_same_answer"] = util_result["same_answer"]
        record["answer_with_correct"] = util_result["answer_with_correct"]
        record["answer_without_correct"] = util_result["answer_without_correct"]

        # Probe 3: Failure classification
        failure_result = failure_probe.classify(
            qa.question,
            qa.answer,
            with_result["answer"],
            util_result["answer_with_correct"],
            with_result["retrieved_texts"],
            rel_result["judgments"],
        )
        record["failure_category"] = failure_result["failure_category"]
        record["failure_explanation"] = failure_result["explanation"]
        record["failure_key_evidence"] = failure_result["key_evidence"]

    return record


def run_single_strategy(
    strategy_name: str,
    conversation: Conversation,
    run_probes: bool = True,
    max_questions: Optional[int] = None,
    top_k: int = RETRIEVAL_TOP_K,
    store: Optional[MemoryStore] = None,
    without_memory_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    num_workers: int = NUM_WORKERS,
    retrieval_method: str = "cosine",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run a single strategy on a single conversation.
    Returns a list of result dicts, one per QA question.
    If *store* is provided, skip ingestion (for top-k ablation reuse).
    If *without_memory_cache* is provided, reuse those answers instead of
    re-generating them (saves one LLM call per question per extra top-k).
    """
    # Phase 1: Ingest (or reuse provided store)
    if store is None:
        store, ingest_log = _ingest_conversation(strategy_name, conversation)
    else:
        ingest_log = []

    # Pre-build the embedding matrix so concurrent retrieve() calls don't race
    if store.size() > 0 and store._dirty:
        store._rebuild_matrix()

    # Build retriever once; reused across all threads (read-only)
    retriever = _make_retriever(store, retrieval_method)
    # Pre-build BM25 index before threads to avoid races
    if retriever is not None and hasattr(retriever, '_build_index'):
        retriever._build_index()
    if retriever is not None and hasattr(retriever, 'bm25'):
        retriever.bm25._build_index()

    # Phase 2: Collect non-adversarial questions
    qa_items = conversation.qa_items
    limit = max_questions if max_questions is not None else MAX_QUESTIONS_PER_CONVERSATION
    if limit:
        qa_items = qa_items[:limit]

    work = [(qi, qa) for qi, qa in enumerate(qa_items) if qa.category != "adversarial"]
    n_total = len(qa_items)
    n_work = len(work)
    print(f"\n    Processing {n_work} questions (top_k={top_k}, retrieval={retrieval_method}, workers={num_workers})...")

    # Phase 3: Process questions concurrently
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _process_question,
                qi, qa, store, top_k, strategy_name,
                conversation, without_memory_cache, run_probes,
                retrieval_method, retriever,
            ): qi
            for qi, qa in work
        }
        with tqdm(total=n_work, desc=f"      {strategy_name} k={top_k} {retrieval_method}", leave=False) as pbar:
            for future in as_completed(futures):
                record = future.result()
                results.append(record)
                pbar.update(1)

    # Maintain deterministic order
    results.sort(key=lambda r: r["question_idx"])

    return results, ingest_log


def run_experiment(
    strategies: List[str] = None,
    run_probes: bool = True,
    save_results: bool = True,
    top_k_values: Optional[List[int]] = None,
    num_workers: int = NUM_WORKERS,
    retrieval_methods: Optional[List[str]] = None,
    skip_convs: int = 0,
):
    """
    Run the full experiment: all strategies × all conversations × all retrieval methods × all top_k.
    Ingestion is done once per strategy × conversation and reused across retrieval methods and top_k.
    """
    if strategies is None:
        strategies = list(ALL_STRATEGIES.keys())
    if top_k_values is None:
        top_k_values = [RETRIEVAL_TOP_K]
    if retrieval_methods is None:
        retrieval_methods = ["cosine"]

    # Load data
    print("Loading LOCOMO dataset...")
    conversations = load_locomo(LOCOMO_DATA_PATH)
    if MAX_CONVERSATIONS:
        conversations = conversations[:MAX_CONVERSATIONS]
    if skip_convs:
        conversations = conversations[skip_convs:]
        print(f"Skipping first {skip_convs} conversations")
    print(f"Loaded {len(conversations)} conversations")
    print(f"Strategies: {strategies}")
    print(f"Retrieval methods: {retrieval_methods}")
    print(f"Top-k values: {top_k_values}")
    print(f"Workers: {num_workers}")

    all_results = []
    all_ingest_logs = {}

    # Create results file upfront so we can save incrementally
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    ingest_path = os.path.join(RESULTS_DIR, f"ingest_logs_{timestamp}.json")

    def _save_incremental():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        saveable_logs = {
            k: v for k, v in all_ingest_logs.items()
            if not k.startswith("_")
        }
        with open(ingest_path, "w") as f:
            json.dump(saveable_logs, f, indent=2, default=str)

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        for conv in conversations:
            print(f"\n  Conversation {conv.conv_id}: {len(conv.sessions)} sessions, {len(conv.qa_items)} questions")

            # Ingest once per strategy × conversation
            store, ingest_log = _ingest_conversation(strategy_name, conv)
            all_ingest_logs[f"{strategy_name}_conv{conv.conv_id}"] = ingest_log

            # Generate without-memory answers once per conversation
            # (independent of strategy, retrieval method, and top_k)
            # Disk-cached, so safe across runs
            without_cache = _get_without_memory_answers(conv, num_workers=num_workers)

            # Run QA for each top_k × retrieval_method (reusing store + without-memory cache)
            for k in top_k_values:
                for ret_method in retrieval_methods:
                    results, _ = run_single_strategy(
                        strategy_name, conv, run_probes=run_probes,
                        top_k=k, store=store,
                        without_memory_cache=without_cache,
                        num_workers=num_workers,
                        retrieval_method=ret_method,
                    )
                    all_results.extend(results)
                    print(f"    top_k={k}, retrieval={ret_method}: processed {len(results)} questions")

            # Save after each conversation
            if save_results:
                _save_incremental()
                print(f"    Results saved incrementally ({len(all_results)} total records)")

    if save_results:
        print(f"\nFinal results: {results_path}")

    return all_results


def run_pilot(strategy_name: str = "basic_rag", n_questions: int = 5,
              top_k: int = RETRIEVAL_TOP_K, num_workers: int = NUM_WORKERS,
              retrieval_method: str = "cosine"):
    """Quick pilot run to verify pipeline works."""
    print("=== PILOT RUN ===")
    conversations = load_locomo(LOCOMO_DATA_PATH)
    conv = conversations[0]
    print(f"Using conversation 0: {len(conv.sessions)} sessions, {len(conv.qa_items)} questions")

    results, ingest_log = run_single_strategy(
        strategy_name,
        conv,
        run_probes=True,
        max_questions=n_questions,
        top_k=top_k,
        num_workers=num_workers,
        retrieval_method=retrieval_method,
    )

    # Print summary
    print(f"\n{'='*40}")
    print(f"Pilot results ({strategy_name}, k={top_k}, retrieval={retrieval_method}, {len(results)} questions):")
    for r in results:
        print(f"\n  Q: {r['question'][:80]}...")
        gold = str(r['gold_answer'])[:60]
        with_mem = str(r['answer_with_memory'])[:60]
        print(f"  Gold: {gold}")
        print(f"  With memory: {with_mem}")
        print(f"  EM: {r.get('em_with', 'N/A')}  |  F1: {r.get('f1_with', 'N/A'):.3f}" if isinstance(r.get('f1_with'), float) else f"  EM: {r.get('em_with', 'N/A')}  |  F1: {r.get('f1_with', 'N/A')}")
        print(f"  Retrieval precision: {r.get('retrieval_precision', 'N/A')}")
        print(f"  Utilization: {r.get('utilization_category', 'N/A')}")
        print(f"  Failure mode: {r.get('failure_category', 'N/A')}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="memory-probe experiment runner")
    parser.add_argument("--pilot", action="store_true", help="Run a quick pilot test")
    parser.add_argument("--strategy", type=str, default=None, help="Run single strategy (basic_rag, extracted_facts, summarized_episodes)")
    parser.add_argument("--no-probes", action="store_true", help="Skip probes (faster, just collect answers)")
    parser.add_argument("--n-questions", type=int, default=5, help="Number of questions for pilot")
    parser.add_argument("--top-k", type=int, nargs="+", default=None,
                        help="Top-k values for retrieval ablation (e.g. --top-k 3 5 10). Default: 5")
    parser.add_argument("--retrieval-method", type=str, default="cosine",
                        choices=VALID_RETRIEVAL_METHODS,
                        help="Retrieval method for pilot runs (default: cosine)")
    parser.add_argument("--retrieval-methods", type=str, nargs="+", default=None,
                        choices=VALID_RETRIEVAL_METHODS,
                        help="Retrieval methods for full runs (e.g. --retrieval-methods bm25 hybrid)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Number of concurrent workers (default: {NUM_WORKERS})")
    parser.add_argument("--skip-convs", type=int, default=0,
                        help="Skip first N conversations (to resume partial runs)")
    args = parser.parse_args()

    if args.pilot:
        strategy = args.strategy or "basic_rag"
        k = args.top_k[0] if args.top_k else RETRIEVAL_TOP_K
        run_pilot(strategy_name=strategy, n_questions=args.n_questions, top_k=k,
                  num_workers=args.workers, retrieval_method=args.retrieval_method)
    else:
        strategies = [args.strategy] if args.strategy else None
        top_k_values = args.top_k if args.top_k else None
        ret_methods = args.retrieval_methods if args.retrieval_methods else None
        run_experiment(
            strategies=strategies,
            run_probes=not args.no_probes,
            top_k_values=top_k_values,
            num_workers=args.workers,
            retrieval_methods=ret_methods,
            skip_convs=args.skip_convs,
        )
