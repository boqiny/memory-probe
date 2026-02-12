"""
Visualization script for memory probe results.
Creates publication-quality comparison figure for ICLR paper.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5


def load_metrics(path: str):
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_data(metrics):
    """Extract F1 and accuracy data organized by strategy and retrieval method."""
    strategies = ['basic_rag', 'extracted_facts', 'summarized_episodes']
    retrieval_methods = ['cosine', 'bm25', 'hybrid']
    
    f1_data = np.zeros((len(strategies), len(retrieval_methods)))
    acc_data = np.zeros((len(strategies), len(retrieval_methods)))
    
    for i, strategy in enumerate(strategies):
        for j, retrieval in enumerate(retrieval_methods):
            key = f"{strategy} / {retrieval} / k=5"
            if key in metrics:
                f1_data[i, j] = metrics[key]['f1_with']
                acc_data[i, j] = metrics[key]['accuracy_with_memory'] * 100
    
    return f1_data, acc_data


def create_figure(metrics_path, output_path='results/figure_comparison.png'):
    """Create publication-quality comparison figure."""
    metrics = load_metrics(metrics_path)
    f1_data, acc_data = extract_data(metrics)
    
    # Labels
    strategy_labels = ['Basic RAG', 'Extracted Facts', 'Summarized Episodes']
    retrieval_labels = ['Cosine', 'BM25', 'Hybrid']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.arange(len(strategy_labels))
    width = 0.25
    
    # Plot 1: F1 Scores
    for j, (label, color) in enumerate(zip(retrieval_labels, colors)):
        offset = (j - 1) * width
        bars = ax1.bar(x + offset, f1_data[:, j], width, 
                      label=label, color=color, alpha=0.85, 
                      edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('(a) F1 Score', fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_labels, rotation=15, ha='right')
    ax1.set_ylim(0, max(f1_data.max() * 1.2, 0.3))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: LLM-Judge Accuracy
    for j, (label, color) in enumerate(zip(retrieval_labels, colors)):
        offset = (j - 1) * width
        bars = ax2.bar(x + offset, acc_data[:, j], width,
                      label=label, color=color, alpha=0.85, 
                      edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('(b) LLM-Judge Accuracy', fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_labels, rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add single legend at the top center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Retrieval Method', loc='upper center', 
              bbox_to_anchor=(0.3, 1.02), ncol=3, frameon=True, fancybox=False, 
              edgecolor='black')
    
    # Add shared x-axis label
    fig.text(0.5, 0.02, 'Memory Strategy', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {output_path}")
    
    plt.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <metrics_file.json> [output_path]")
        print("\nExample:")
        print("  python visualize_results.py results/combined_0123_metrics.json")
        sys.exit(1)
    
    metrics_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'results/figure_comparison.png'
    
    create_figure(metrics_path, output_path)
