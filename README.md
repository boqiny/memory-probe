# Memory Probe

> **Accepted at the [ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems (MemAgents)](https://openreview.net/forum?id=cxYbqAtBIz)**

📄 **Paper:** [Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory](https://arxiv.org/abs/2603.02473)

Diagnostic framework that tests whether LLM memory agents actually *use* their retrieved memories. Evaluates three memory strategies on the LOCOMO dataset using LLM-as-judge probes for retrieval relevance, memory utilization, and failure analysis.

## Strategies

- **Default RAG** — stores raw conversation chunks (3 turns each), no LLM at write time
- **Extracted Facts** — LLM extracts structured facts per session with conflict resolution (A-MEM / Mem0 style)
- **Summarized Episodes** — LLM summarizes each session into one entry (MemGPT style)

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

## Usage

```bash
# Pilot run (5 questions, 1 strategy)
python run_experiment.py --pilot --strategy basic_rag

# Full experiment (all strategies, all conversations)
python run_experiment.py

# Top-k ablation
python run_experiment.py --top-k 3 5 10

# Single strategy with custom workers
python run_experiment.py --strategy extracted_facts --workers 10

# Analyze results
python analyze_results.py results/results_TIMESTAMP.json
```

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{yuan2026diagnosing,
  title={Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory},
  author={Yuan, Boqin and Su, Yue and Yao, Kun},
  booktitle={ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems}
}
}
