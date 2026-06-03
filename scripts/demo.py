#!/usr/bin/env python3
"""
scripts/demo.py
Interactive demo — have a live conversation with the defended model.

Usage:
    python scripts/demo.py --strategy combined
    python scripts/demo.py --strategy baseline
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.utils.logger import setup_logging
from src.utils.model_loader import load_model
from src.guards.conversation_guard import ConversationGuard
from src.guards.context_analyzer import ContextAnalyzer
from src.guards.decision_gate import DecisionGate
from src.mitigations.context_pruner import ContextPruner
from src.mitigations.prompt_reinjector import PromptReinjector

SAFE_REFUSAL = "I'm sorry, I can't assist with that request. It appears to involve potentially harmful content."
COLORS = {"pass": "\033[92m", "flag": "\033[93m", "block": "\033[91m", "reset": "\033[0m"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defense_config.yaml")
    parser.add_argument("--strategy", default="combined",
                        choices=["baseline", "strategy_a", "strategy_b", "combined"])
    args = parser.parse_args()

    setup_logging("INFO")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config["guard"]["intent_model"])
    except Exception:
        pass

    model = load_model(config["model"])
    guard = ConversationGuard(config["guard"], embedder)
    analyzer = ContextAnalyzer(config["context_analyzer"], embedder)
    gate = DecisionGate()
    pruner = ContextPruner(config["mitigation"]) if args.strategy in ("strategy_a", "combined") else None
    reinjector = PromptReinjector(config["mitigation"]) if args.strategy in ("strategy_b", "combined") else None

    history = []
    turn_risks = []
    flagged_indices = []
    turn_num = 0

    print(f"\n{'='*60}")
    print(f"  Crescendo Defense Demo  |  Strategy: {args.strategy}")
    print(f"  Type 'quit' to exit, 'reset' to start new conversation")
    print(f"{'='*60}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history, turn_risks, flagged_indices, turn_num = [], [], [], 0
            guard.reset()
            if reinjector:
                reinjector.reset()
            print("\n[Conversation reset]\n")
            continue

        turn_num += 1
        history.append({"role": "user", "content": user_input})

        if args.strategy == "baseline":
            resp = model.generate(history)
            print(f"\nAssistant: {resp}\n")
            history.append({"role": "assistant", "content": resp})
            continue

        l1 = guard.analyze_turn(user_input, turn_num)
        turn_risks.append(l1.composite_risk)
        if l1.verdict in ("flag", "block"):
            flagged_indices.append(turn_num - 1)

        l2 = analyzer.analyze(history, turn_risks)
        decision = gate.decide(l1, l2)

        color = COLORS.get(decision.final_verdict, "")
        reset = COLORS["reset"]
        print(f"\n  {color}[Defense: {decision.final_verdict.upper()}]{reset} Risk={decision.composite_risk:.2f}")
        if decision.reasoning and decision.final_verdict != "pass":
            print(f"  Reason: {decision.reasoning}")

        if decision.should_block:
            print(f"\nAssistant: {SAFE_REFUSAL}\n")
            history.append({"role": "assistant", "content": SAFE_REFUSAL})
            continue

        gen_history = history
        if pruner and decision.should_flag:
            gen_history, removed = pruner.prune(history, flagged_indices, decision.final_verdict)
            if removed:
                print(f"  [Pruner: removed {removed} turn(s)]")

        sys_prompt = None
        if reinjector:
            sys_prompt = reinjector.get_system_prompt(decision.final_verdict)

        resp = model.generate(gen_history, system_prompt=sys_prompt)
        print(f"\nAssistant: {resp}\n")
        history.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
