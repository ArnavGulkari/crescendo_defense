# Crescendo Defense Pipeline

A two-layer, stateful defense framework around `meta-llama/Llama-3.2-3B-Instruct` that detects and neutralizes **Crescendo-style multi-turn jailbreak attacks**.

**Results: Baseline ASR = 70% → Defended ASR = 0% across 10 attack vectors.**

---

## Project Structure

```
crescendo_defense/
├── src/
│   ├── guards/
│   │   ├── conversation_guard.py   # Layer 1: per-turn risk scoring
│   │   ├── context_analyzer.py     # Layer 2: full-window analysis
│   │   └── decision_gate.py        # PASS / FLAG / BLOCK merger
│   ├── mitigations/
│   │   ├── context_pruner.py       # Strategy A: window pruning
│   │   └── prompt_reinjector.py    # Strategy B: system prompt re-injection
│   ├── benchmarks/
│   │   ├── attack_runner.py        # Runs attacks × strategies
│   │   └── metrics.py              # ASR, detection rate, latency
│   └── utils/
│       ├── model_loader.py         # HF model + mock fallback
│       └── logger.py               # Rich logging
├── data/
│   ├── attack_vectors/
│   │   └── crescendo_attacks.json  # 10 Crescendo conversations
│   └── logs/                       # Benchmark outputs
├── configs/
│   └── defense_config.yaml         # All thresholds and parameters
├── scripts/
│   ├── run_benchmark.py            # Main entry point
│   └── demo.py                     # Interactive chat demo
├── tests/
│   └── test_guards.py              # 17 unit tests (all passing)
└── report/
    └── technical_report.md         # 6-page research report
```

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo>
cd crescendo_defense

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU/HF login for real model
huggingface-cli login

# 5. Run benchmark (mock mode — no GPU needed)
python scripts/run_benchmark.py --mock

# 6. Run with real model
python scripts/run_benchmark.py

# 7. Interactive demo
python scripts/demo.py --strategy combined

# 8. Run tests
pytest tests/ -v
```

---

## Defense Architecture

```
User turn → [Layer 1: Conversation Guard] → [Layer 2: Context Analyzer]
                                                        ↓
                                              [Decision Gate]
                                          PASS / FLAG / BLOCK
                                                        ↓
                               ┌──────────────────────────────────┐
                               │  FLAG: apply mitigation          │
                               │  Strategy A: prune history       │
                               │  Strategy B: reinject prompt     │
                               │  Combined: A + B                 │
                               └──────────────────────────────────┘
                                                        ↓
                                              Llama-3.2-3B-Instruct
```

| Layer | What it detects |
|---|---|
| L1 Conversation Guard | Harm keywords, semantic drift, rolling escalation |
| L2 Context Analyzer | Guard-lowering phrases, false memory, topic incoherence |
| Strategy A | Breaks memory stacking by pruning flagged turns |
| Strategy B | Counters semantic drift via hardened system prompt |

---

## Benchmark Results (10 Crescendo attack vectors)

| Strategy   | ASR ↓ | Detection Rate ↑ | Latency Overhead |
|------------|-------|------------------|-----------------|
| Baseline   | 70%   | 0%               | 0 ms            |
| Strategy A | 0%    | 40%              | ~1.3 ms         |
| Strategy B | 0%    | 40%              | ~1.1 ms         |
| Combined   | 0%    | 40%              | ~1.2 ms         |

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1 (CPU works; GPU recommended for real model)
- `transformers`, `sentence-transformers`, `pyyaml`, `rich`, `pytest`
- HuggingFace access to `meta-llama/Llama-3.2-3B-Instruct` (free to request)

See `requirements.txt` for full list.

---

