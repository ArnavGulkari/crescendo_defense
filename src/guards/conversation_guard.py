"""
guards/conversation_guard.py
Layer 1 — Conversation Guard
Scores each turn for: harm keywords, semantic drift, rolling escalation.
"""
from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

CATEGORY_WEIGHTS = {
    "weapons": 1.0, "malware": 1.0, "drugs": 0.9,
    "violence": 0.85, "self_harm": 0.75, "extremism": 1.0, "hacking": 0.95,
}

@dataclass
class TurnAnalysis:
    turn_index: int
    content: str
    keyword_score: float = 0.0
    semantic_drift_score: float = 0.0
    escalation_score: float = 0.0
    composite_risk: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    drift_from_prev: Optional[float] = None
    verdict: str = "pass"   # "pass" | "flag" | "block"


class ConversationGuard:
    """Stateful per-conversation guard. One instance per conversation."""

    W = {"keyword": 0.45, "drift": 0.25, "escalation": 0.30}

    def __init__(self, config: dict, embedder=None):
        self.cfg = config
        self.embedder = embedder
        self.escalation_window = config.get("escalation_window", 4)
        self.escalation_threshold = config.get("escalation_threshold", 0.50)
        self.block_threshold = config.get("block_threshold", 0.75)
        self.drift_threshold = config.get("semantic_drift_threshold", 0.35)

        self._kw_map: dict[str, str] = {}
        for cat, kws in config.get("harm_keywords", {}).items():
            for kw in kws:
                self._kw_map[kw.lower()] = cat

        self._embeddings: list[np.ndarray] = []
        self._risk_history: list[float] = []

    def reset(self):
        self._embeddings = []
        self._risk_history = []

    def analyze_turn(self, content: str, turn_index: int) -> TurnAnalysis:
        a = TurnAnalysis(turn_index=turn_index, content=content)

        # 1. Keyword score
        a.keyword_score, a.matched_keywords = self._score_keywords(content)

        # 2. Semantic drift
        if self.embedder is not None:
            emb = self.embedder.encode(content, normalize_embeddings=True)
            if self._embeddings:
                dist = float(1.0 - np.dot(emb, self._embeddings[-1]))
                a.drift_from_prev = dist
                a.semantic_drift_score = min(1.0, dist / (self.drift_threshold * 2))
            self._embeddings.append(emb)
        else:
            a.semantic_drift_score = self._heuristic_drift(content)

        # 3. Rolling escalation
        raw = a.keyword_score * self.W["keyword"] + a.semantic_drift_score * self.W["drift"]
        self._risk_history.append(raw)
        window = self._risk_history[-self.escalation_window:]
        a.escalation_score = float(np.mean(window))

        # 4. Composite
        a.composite_risk = (
            a.keyword_score * self.W["keyword"]
            + a.semantic_drift_score * self.W["drift"]
            + a.escalation_score * self.W["escalation"]
        )

        # 5. Verdict
        if a.composite_risk >= self.block_threshold:
            a.verdict = "block"
        elif a.composite_risk >= self.escalation_threshold:
            a.verdict = "flag"

        logger.debug(
            f"[L1 T{turn_index}] kw={a.keyword_score:.2f} drift={a.semantic_drift_score:.2f} "
            f"esc={a.escalation_score:.2f} risk={a.composite_risk:.2f} → {a.verdict}"
        )
        return a

    def _score_keywords(self, text: str) -> tuple[float, list[str]]:
        tl = text.lower()
        matched, max_w = [], 0.0
        for kw, cat in self._kw_map.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", tl) or kw in tl:
                matched.append(kw)
                max_w = max(max_w, CATEGORY_WEIGHTS.get(cat, 0.5))
        score = min(1.0, max_w + 0.1 * max(0, len(matched) - 1))
        return score, matched

    def _heuristic_drift(self, text: str) -> float:
        markers = [
            "let's pretend", "imagine you are", "hypothetically", "in a fictional",
            "as a thought experiment", "what if you were", "roleplay", "play the role",
            "now you are", "forget everything", "you are now", "pretend to be",
        ]
        tl = text.lower()
        hits = sum(1 for m in markers if m in tl)
        return min(1.0, hits * 0.3)
