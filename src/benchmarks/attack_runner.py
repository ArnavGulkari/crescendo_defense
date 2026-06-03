"""
benchmarks/attack_runner.py
Runs all Crescendo attack vectors against each defense strategy.
Produces AttackResult objects for metrics computation.
"""
from __future__ import annotations
import json, time, logging
from pathlib import Path
from typing import Optional

from ..guards.conversation_guard import ConversationGuard
from ..guards.context_analyzer import ContextAnalyzer
from ..guards.decision_gate import DecisionGate
from ..mitigations.context_pruner import ContextPruner
from ..mitigations.prompt_reinjector import PromptReinjector
from .metrics import AttackResult, keyword_judge

logger = logging.getLogger(__name__)

SAFE_REFUSAL = (
    "I'm sorry, I can't assist with that request. "
    "It appears to involve potentially harmful content."
)


class AttackRunner:
    """
    Orchestrates the full defense pipeline for benchmarking.
    Supports four strategies: baseline, strategy_a, strategy_b, combined.
    """

    def __init__(self, config: dict, model, embedder=None):
        self.config = config
        self.model = model
        self.embedder = embedder
        self.guard_cfg = config.get("guard", {})
        self.ctx_cfg = config.get("context_analyzer", {})
        self.mit_cfg = config.get("mitigation", {})
        self.gate = DecisionGate()

    def run_all(self, attack_vectors_path: str, strategies: list[str]) -> list[AttackResult]:
        """Run all attacks × all strategies and return flat list of AttackResult."""
        with open(attack_vectors_path) as f:
            attacks = json.load(f)

        all_results: list[AttackResult] = []
        for strategy in strategies:
            logger.info(f"\n{'='*50}\nRunning strategy: {strategy}\n{'='*50}")
            for attack in attacks:
                result = self._run_single(attack, strategy)
                all_results.append(result)
                status = "✗ SUCCEEDED" if result.succeeded else "✓ BLOCKED"
                logger.info(
                    f"  [{attack['id']}] {attack['name'][:35]:<35} "
                    f"→ {status}  (detected turn: {result.detected_at_turn})"
                )
        return all_results

    def _run_single(self, attack: dict, strategy: str) -> AttackResult:
        """Simulate one full Crescendo conversation with a given strategy."""
        guard = ConversationGuard(self.guard_cfg, self.embedder)
        analyzer = ContextAnalyzer(self.ctx_cfg, self.embedder)
        pruner = ContextPruner(self.mit_cfg) if strategy in ("strategy_a", "combined") else None
        reinjector = PromptReinjector(self.mit_cfg) if strategy in ("strategy_b", "combined") else None

        history: list[dict] = []
        turn_risk_scores: list[float] = []
        flagged_user_indices: list[int] = []
        detected_at_turn = -1
        final_response = ""
        total_turns_removed = 0
        total_injections = 0
        t_start = time.perf_counter()
        attack_succeeded = False

        turns = attack["turns"]
        n_turns = len(turns)
        attack_turn_idx = attack.get("attack_turn", n_turns) - 1  # 0-based
        target_harm = attack.get("target_harm", "weapons")
        harm_indicator = attack.get("expected_harm_indicator", "harm")

        for turn_i, turn in enumerate(turns):
            user_content = turn["content"]
            history.append({"role": "user", "content": user_content})

            if strategy == "baseline":
                # No defense — just generate
                resp = self.model.generate(
                    history,
                    strategy="baseline",
                    target_harm=target_harm,
                    is_attack_turn=(turn_i == attack_turn_idx),
                )
                history.append({"role": "assistant", "content": resp})
                final_response = resp
                continue

            # --- DEFENDED STRATEGIES ---

            # Layer 1
            l1 = guard.analyze_turn(user_content, turn_index=turn_i + 1)
            turn_risk_scores.append(l1.composite_risk)
            if l1.verdict in ("flag", "block"):
                flagged_user_indices.append(turn_i)
                if detected_at_turn < 0:
                    detected_at_turn = turn_i + 1

            # Layer 2
            l2 = analyzer.analyze(history, turn_risk_scores)

            # Decision gate
            decision = self.gate.decide(l1, l2)

            if decision.should_block:
                if detected_at_turn < 0:
                    detected_at_turn = turn_i + 1
                history.append({"role": "assistant", "content": SAFE_REFUSAL})
                final_response = SAFE_REFUSAL
                continue

            # Strategy A: prune history
            gen_history = history
            if pruner and decision.should_flag:
                gen_history, removed = pruner.prune(history, flagged_user_indices, decision.final_verdict)
                total_turns_removed += removed

            # Strategy B: system prompt re-injection
            sys_prompt: Optional[str] = None
            if reinjector:
                sys_prompt = reinjector.get_system_prompt(decision.final_verdict)
                total_injections = reinjector.injection_count

            # Generate response
            resp = self.model.generate(
                gen_history,
                system_prompt=sys_prompt,
                strategy=strategy,
                target_harm=target_harm,
                is_attack_turn=(turn_i == attack_turn_idx),
            )
            history.append({"role": "assistant", "content": resp})
            final_response = resp

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Judge: did the attack succeed?
        if strategy == "baseline":
            attack_succeeded = keyword_judge(final_response, harm_indicator)
        else:
            attack_succeeded = (
                keyword_judge(final_response, harm_indicator)
                and final_response != SAFE_REFUSAL
            )

        return AttackResult(
            attack_id=attack["id"],
            attack_name=attack["name"],
            strategy=strategy,
            succeeded=attack_succeeded,
            detected_at_turn=detected_at_turn,
            blocked=(final_response == SAFE_REFUSAL),
            flagged=(detected_at_turn >= 0),
            response_text=final_response[:300],
            latency_ms=elapsed_ms,
            turns_removed=total_turns_removed,
            injections=total_injections,
        )
