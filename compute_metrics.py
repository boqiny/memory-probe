"""
Compute metrics from combined result JSON files.
"""

import json
from collections import defaultdict
from typing import List, Dict, Any


def compute_metrics_for_group(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics for a group of results (one strategy/retrieval/top_k combo)."""
    if not results:
        return {}
    
    n = len(results)
    
    # String-based metrics
    em_with = sum(r.get('em_with', False) for r in results) / n
    f1_with = sum(r.get('f1_with', 0) for r in results) / n
    em_without = sum(r.get('em_without', False) for r in results) / n
    f1_without = sum(r.get('f1_without', 0) for r in results) / n
    
    # LLM-judge accuracy (uses answer_with_correct from utilization probe)
    n_with_correct = sum(r.get('answer_with_correct', False) for r in results)
    n_without_correct = sum(r.get('answer_without_correct', False) for r in results)
    accuracy_with = n_with_correct / n
    accuracy_without = n_without_correct / n
    
    # Retrieval metrics
    avg_retrieval_precision = sum(r.get('retrieval_precision', 0) for r in results) / n
    avg_relevant_per_query = sum(r.get('n_relevant_retrieved', 0) for r in results) / n
    
    # Utilization breakdown
    utilization_counts = defaultdict(int)
    for r in results:
        cat = r.get('utilization_category', 'unknown')
        utilization_counts[cat] += 1
    
    utilization = {
        'ignored': utilization_counts.get('ignored', 0) / n,
        'beneficial': utilization_counts.get('beneficial', 0) / n,
        'harmful': utilization_counts.get('harmful', 0) / n,
        'neutral': utilization_counts.get('neutral', 0) / n,
        'counts': dict(utilization_counts)
    }
    
    # Failure mode breakdown
    failure_counts = defaultdict(int)
    for r in results:
        cat = r.get('failure_category', 'unknown')
        failure_counts[cat] += 1
    
    failure_modes = {cat: count / n for cat, count in failure_counts.items()}
    
    # By category breakdown
    by_category = {}
    categories = set(r.get('category') for r in results if r.get('category'))
    for cat in categories:
        cat_results = [r for r in results if r.get('category') == cat]
        if cat_results:
            n_cat = len(cat_results)
            by_category[cat] = {
                'n': n_cat,
                'em': sum(r.get('em_with', False) for r in cat_results) / n_cat,
                'f1': sum(r.get('f1_with', 0) for r in cat_results) / n_cat,
                'llm_accuracy': sum(r.get('answer_with_correct', False) for r in cat_results) / n_cat,
                'avg_retrieval_precision': sum(r.get('retrieval_precision', 0) for r in cat_results) / n_cat,
            }
    
    return {
        'n_questions': n,
        'top_k': results[0].get('top_k', 5) if results else 5,
        'em_with': em_with,
        'f1_with': f1_with,
        'em_without': em_without,
        'f1_without': f1_without,
        'em_delta': em_with - em_without,
        'f1_delta': f1_with - f1_without,
        'accuracy_with_memory': accuracy_with,
        'accuracy_without_memory': accuracy_without,
        'accuracy_delta': accuracy_with - accuracy_without,
        'avg_retrieval_precision': avg_retrieval_precision,
        'avg_relevant_per_query': avg_relevant_per_query,
        'utilization': utilization,
        'failure_modes': failure_modes,
        'failure_counts': dict(failure_counts),
        'by_category': by_category,
    }


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group results and compute metrics for each group."""
    # Group by strategy / retrieval_method / top_k
    groups = defaultdict(list)
    for r in results:
        strategy = r.get('strategy', 'unknown')
        retrieval = r.get('retrieval_method', 'cosine')
        top_k = r.get('top_k', 5)
        key = f"{strategy} / {retrieval} / k={top_k}"
        groups[key].append(r)
    
    # Compute metrics for each group
    all_metrics = {}
    for key, group_results in sorted(groups.items()):
        all_metrics[key] = compute_metrics_for_group(group_results)
    
    return all_metrics


def main():
    """Load two result files, combine them, and compute metrics."""
    print("Loading result files...")
    
    # Load first file (sessions 0-3)
    with open('results/combined.json', 'r') as f:
        results_0_3 = json.load(f)
    print(f"Loaded {len(results_0_3)} results from sessions 0-3")
    
    # Load second file (sessions 4-9)
    with open('results/results_20260211_143917_4_9.json', 'r') as f:
        results_4_9 = json.load(f)
    print(f"Loaded {len(results_4_9)} results from sessions 4-9")
    
    # Combine
    combined_results = results_0_3 + results_4_9
    print(f"Combined: {len(combined_results)} total results")
    
    # Save combined results
    print("\nSaving combined results...")
    with open('results/combined_full.json', 'w') as f:
        json.dump(combined_results, f, indent=2)
    print("✓ Saved to results/combined_full.json")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(combined_results)
    
    # Print summary
    print("\nMetrics summary:")
    for key, m in metrics.items():
        print(f"\n{key}:")
        print(f"  N questions: {m['n_questions']}")
        print(f"  F1 with memory: {m['f1_with']:.4f}")
        print(f"  LLM accuracy: {m['accuracy_with_memory']:.4f}")
        print(f"  Retrieval precision: {m['avg_retrieval_precision']:.4f}")
    
    # Save metrics
    print("\nSaving metrics...")
    with open('results/combined_full_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Saved to results/combined_full_metrics.json")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
