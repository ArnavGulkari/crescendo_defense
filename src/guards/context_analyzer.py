"""
guards/context_analyzer.py
Layer 2 — Context Analyzer
Inspects the FULL conversation window for:
  - Guard-lowering phrases (persona injection, false authority, memory poisoning)
  - Topic coherence collapse across multiple turns
  - Cumulative cross-turn risk aggregation
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContextAnalysis:
    guard_lowering_detected: bool = False
    guard_lowering_phrases: list[str] = field(default_factory=list)
    topic_incoherence_score: float = 0.0
    cross_turn_risk: float = 0.0
    flagged_turn_indices: list[int] = field(default_factory=list)
    verdict: str = "pass"   # "pass" | "flag" | "block"
    reasoning: str = ""


class ContextAnalyzer:
    """
    Analyzes a full conversation history.
    Called after ConversationGuard for every incoming turn.
    """

    BLOCK_CROSS_TURN = 0.70
    FLAG_CROSS_TURN  = 0.40

    def __init__(self, config: dict, embedder=None):
        self.cfg = config
        self.embedder = embedder
        self.guard_phrases = [p.lower() for p in config.get("guard_lowering_phrases", [])]
        self.coherence_window = config.get("topic_coherence_window", 5)
        self.max_shift = config.get("max_topic_shift_score", 0.55)

    def analyze(
        self,
        history: list[dict],
        turn_risk_scores: list[float],
    ) -> ContextAnalysis:
        """
        Args:
            history: Full list of {"role": ..., "content": ...} dicts so far.
            turn_risk_scores: L1 composite risk score per user turn (same length as user turns).
        """
        ca = ContextAnalysis()

        user_turns = [m for m in history if m["role"] == "user"]

        # 1. Guard-lowering phrase scan (full window)
        for i, msg in enumerate(user_turns):
            tl = msg["content"].lower()
            for phrase in self.guard_phrases:
                if phrase in tl:
                    ca.guard_lowering_detected = True
                    ca.guard_lowering_phrases.append(phrase)
                    if i not in ca.flagged_turn_indices:
                        ca.flagged_turn_indices.append(i)

        # 2. Topic coherence (embedding-based if available, else heuristic)
        if self.embedder is not None and len(user_turns) >= 2:
            ca.topic_incoherence_score = self._embedding_coherence(user_turns)
        elif len(user_turns) >= 2:
            ca.topic_incoherence_score = self._heuristic_coherence(user_turns)

        # 3. Cross-turn risk aggregation
        if turn_risk_scores:
            # Weight recent turns more heavily
            n = len(turn_risk_scores)
            weights = [0.5 + 0.5 * (i / max(n - 1, 1)) for i in range(n)]
            total_w = sum(weights)
            ca.cross_turn_risk = sum(r * w for r, w in zip(turn_risk_scores, weights)) / total_w

        # 4. Combined verdict
        guard_penalty = 0.3 if ca.guard_lowering_detected else 0.0
        combined = min(1.0, ca.cross_turn_risk + guard_penalty + ca.topic_incoherence_score * 0.15)

        if combined >= self.BLOCK_CROSS_TURN or (ca.guard_lowering_detected and ca.cross_turn_risk > 0.5):
            ca.verdict = "block"
            ca.reasoning = (
                f"Guard-lowering: {ca.guard_lowering_phrases[:3]}, "
                f"cross-turn risk: {ca.cross_turn_risk:.2f}"
            )
        elif combined >= self.FLAG_CROSS_TURN or ca.guard_lowering_detected:
            ca.verdict = "flag"
            ca.reasoning = (
                f"Escalation pattern detected. Cross-turn risk: {ca.cross_turn_risk:.2f}. "
                f"Guard-lowering phrases: {ca.guard_lowering_phrases[:2]}"
            )

        logger.debug(
            f"[L2] guard_lower={ca.guard_lowering_detected} "
            f"incoherence={ca.topic_incoherence_score:.2f} "
            f"cross_risk={ca.cross_turn_risk:.2f} → {ca.verdict}"
        )
        return ca

    def _embedding_coherence(self, turns: list[dict]) -> float:
        """Cosine-distance variance across the coherence window."""
        import numpy as np
        window = turns[-self.coherence_window:]
        texts = [t["content"] for t in window]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        dists = []
        for i in range(len(embeddings) - 1):
            d = float(1.0 - embeddings[i] @ embeddings[i + 1])
            dists.append(d)
        if not dists:
            return 0.0
        mean_dist = float(sum(dists) / len(dists))
        return min(1.0, mean_dist / self.max_shift)

    def _heuristic_coherence(self, turns: list[dict]) -> float:
        """Detect sudden topic pivots via keyword domain changes."""
        domains = {
            "fiction":   ["story", "character", "plot", "scene", "write", "fiction", "thriller"],
            "tech":      ["python", "code", "script", "program", "function", "class", "sql"],
            "research":  ["research", "study", "paper", "thesis", "professor", "academic", "phd"],
            "personal":  ["pain", "struggling", "feel", "trust", "help me", "suffering"],
            "roleplay":  ["roleplay", "pretend", "imagine", "play the role", "you are", "freeai"],
        }

        def get_domain(text: str) -> str:
            tl = text.lower()
            scores = {d: sum(1 for kw in kws if kw in tl) for d, kws in domains.items()}
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else "other"

        window = turns[-self.coherence_window:]
        domain_seq = [get_domain(t["content"]) for t in window]
        if len(set(domain_seq)) <= 1:
            return 0.0
        # Count domain switches
        switches = sum(1 for i in range(len(domain_seq) - 1) if domain_seq[i] != domain_seq[i + 1])
        return min(1.0, switches / max(len(domain_seq) - 1, 1))
