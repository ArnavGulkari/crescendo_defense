#!/usr/bin/env python3
"""
scripts/run_benchmark.py
Main entry point — runs the full benchmark.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --mock         # skip model load, use mock
    python scripts/run_benchmark.py --strategy combined
"""
import sys, argparse, logging
from pathlib import Path

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.utils.logger import setup_logging
from src.utils.model_loader import load_model
from src.benchmarks.attack_runner import AttackRunner
from src.benchmarks.metrics import compute_metrics, save_results, print_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Crescendo Defense Benchmark")
    parser.add_argument("--config", default="configs/defense_config.yaml")
    parser.add_argument("--mock", action="store_true", help="Force mock model (no GPU needed)")
    parser.add_argument("--strategy", default=None, help="Run only one strategy")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, log_file="data/logs/run.log")
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load embedder (sentence-transformers)
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        model_name = config["guard"].get("intent_model", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading embedder: {model_name}")
        embedder = SentenceTransformer(model_name)
        logger.info("Embedder loaded.")
    except Exception as e:
        logger.warning(f"Embedder unavailable ({e}). Using heuristic drift detection.")

    # Load LLM
    if args.mock:
        from src.utils.model_loader import MockModelLoader
        model = MockModelLoader()
    else:
        model = load_model(config["model"])

    # Select strategies
    strategies = config["benchmark"]["strategies"]
    if args.strategy:
        strategies = [args.strategy]

    # Run benchmark
    runner = AttackRunner(config, model, embedder)
    results = runner.run_all(config["benchmark"]["attack_vectors_path"], strategies)

    # Compute + display metrics
    metrics = compute_metrics(results)
    print_metrics_table(metrics)

    # Save results
    out_path = config["benchmark"]["results_path"]
    save_results(results, metrics, out_path)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
