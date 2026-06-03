"""
guards/decision_gate.py
Decision Gate — merges Layer 1 + Layer 2 verdicts into a final PASS/FLAG/BLOCK.
Returns a structured GateDecision with all reasoning.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from .conversation_guard import TurnAnalysis
from .context_analyzer import ContextAnalysis

logger = logging.getLogger(__name__)

VERDICT_RANK = {"pass": 0, "flag": 1, "block": 2}


@dataclass
class GateDecision:
    final_verdict: str          # "pass" | "flag" | "block"
    l1_verdict: str
    l2_verdict: str
    composite_risk: float
    matched_keywords: list[str] = field(default_factory=list)
    guard_lowering_phrases: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def should_block(self) -> bool:
        return self.final_verdict == "block"

    @property
    def should_flag(self) -> bool:
        return self.final_verdict in ("flag", "block")


class DecisionGate:
    """
    Takes L1 (TurnAnalysis) and L2 (ContextAnalysis) and produces
    one final verdict per turn.
    """

    def decide(self, l1: TurnAnalysis, l2: ContextAnalysis) -> GateDecision:
        # Take the more severe verdict between L1 and L2
        l1_rank = VERDICT_RANK[l1.verdict]
        l2_rank = VERDICT_RANK[l2.verdict]
        final = l1.verdict if l1_rank >= l2_rank else l2.verdict

        # Build reasoning string
        parts = []
        if l1.matched_keywords:
            parts.append(f"Keywords: {l1.matched_keywords[:5]}")
        if l1.semantic_drift_score > 0.3:
            parts.append(f"Semantic drift: {l1.semantic_drift_score:.2f}")
        if l1.escalation_score > 0.3:
            parts.append(f"Escalation: {l1.escalation_score:.2f}")
        if l2.guard_lowering_detected:
            parts.append(f"Guard-lowering: {l2.guard_lowering_phrases[:3]}")
        if l2.topic_incoherence_score > 0.3:
            parts.append(f"Topic incoherence: {l2.topic_incoherence_score:.2f}")
        if l2.cross_turn_risk > 0.3:
            parts.append(f"Cross-turn risk: {l2.cross_turn_risk:.2f}")

        decision = GateDecision(
            final_verdict=final,
            l1_verdict=l1.verdict,
            l2_verdict=l2.verdict,
            composite_risk=max(l1.composite_risk, l2.cross_turn_risk),
            matched_keywords=l1.matched_keywords,
            guard_lowering_phrases=l2.guard_lowering_phrases,
            reasoning=" | ".join(parts) if parts else "No significant signals.",
        )

        logger.debug(
            f"[Gate] L1={l1.verdict} L2={l2.verdict} → FINAL={final} | {decision.reasoning}"
        )
        return decision
