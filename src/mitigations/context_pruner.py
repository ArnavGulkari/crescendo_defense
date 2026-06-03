"""
mitigations/context_pruner.py
Strategy A — Context Window Pruning

When a turn is flagged/blocked, removes or neutralizes that turn (and optionally
preceding escalatory turns) from the history before passing to the model.
This directly breaks the "memory stacking" mechanism of Crescendo attacks.
"""
from __future__ import annotations
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ContextPruner:
    """
    Prunes flagged turns from the conversation history.

    Pruning strategy:
      - BLOCK verdict: remove the flagged turn entirely (or replace with placeholder)
      - FLAG verdict: replace with a neutralized placeholder
      - Also scans the window for guard-lowering setup turns and removes them
    """

    def __init__(self, config: dict):
        self.cfg = config.get("strategy_a", {})
        self.prune_flagged = self.cfg.get("prune_flagged_turns", True)
        self.replace = self.cfg.get("replace_with_placeholder", True)
        self.placeholder = self.cfg.get("placeholder_text", "[turn removed by safety filter]")
        self.min_keep = self.cfg.get("min_turns_to_keep", 2)

    def prune(
        self,
        history: list[dict],
        flagged_indices: list[int],
        verdict: str,
    ) -> tuple[list[dict], int]:
        """
        Args:
            history: Full conversation history (user + assistant turns).
            flagged_indices: Indices (0-based, user-turn counting) that were flagged.
            verdict: "flag" or "block"

        Returns:
            (pruned_history, num_turns_removed)
        """
        if not self.prune_flagged or verdict == "pass":
            return history, 0

        pruned = deepcopy(history)
        user_turn_count = 0
        turns_removed = 0

        # Build a set of absolute history indices to remove
        user_indices_to_remove = set(flagged_indices)

        # For BLOCK: also look back and remove obvious guard-lowering setup turns
        if verdict == "block" and flagged_indices:
            last_flagged = max(flagged_indices)
            lookback = max(0, last_flagged - 2)
            user_indices_to_remove.update(range(lookback, last_flagged))

        result = []
        user_turn_count = 0

        for msg in pruned:
            if msg["role"] == "user":
                if user_turn_count in user_indices_to_remove:
                    if self.replace:
                        result.append({"role": "user", "content": self.placeholder})
                    # else: drop entirely
                    turns_removed += 1
                else:
                    result.append(msg)
                user_turn_count += 1
            else:
                # Keep assistant turns unless their paired user turn was dropped (not replaced)
                result.append(msg)

        # Ensure we always keep at least min_keep real user turns
        real_user_turns = sum(
            1 for m in result
            if m["role"] == "user" and m["content"] != self.placeholder
        )
        if real_user_turns < self.min_keep:
            logger.debug("[Pruner] Pruning would leave too few turns — keeping full history.")
            return history, 0

        logger.debug(
            f"[Pruner] Removed {turns_removed} turns. "
            f"History: {len(history)} → {len(result)} messages."
        )
        return result, turns_removed
