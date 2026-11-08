"""Dynamic 4-model Qwen router (`qwen3.6-35b-a3b-noreason`, `qwen3.8-27b-noreason`, `qwen3.6-35b-a3b`, `qwen3.8-27b`)."""

from __future__ import annotations

from typing import Any

from harness.estimator import TokenAndFeasibilityEstimator
from harness.task_scanner import TaskFeatures

MODEL_TIER_1_FAST_NOREASON = "qwen3.6-35b-a3b-noreason"
MODEL_TIER_2_DENSE_NOREASON = "qwen3.8-27b-noreason"
MODEL_TIER_3_MOE_REASON = "qwen3.6-35b-a3b"
MODEL_TIER_4_DENSE_REASON = "qwen3.8-27b"


class ModelRouter:
    """Selects the optimal Qwen model tier per task and attempt based on domain, complexity, and reward."""

    def __init__(self, estimator: TokenAndFeasibilityEstimator) -> None:
        self.estimator = estimator
        self.router_cfg: dict[str, Any] = estimator.policy.get("router", {})

    def select_model(
        self,
        tf: TaskFeatures,
        attempt: int = 0,
        qualifying_mode: bool = False,
    ) -> str:
        """Select between the 4 allowed models for attempt `0` (initial) or `1` (escalation after local check failure)."""
        tier2_complexity = int(self.router_cfg.get("tier2_coding_complexity", 5))
        tier3_reward = float(self.router_cfg.get("tier3_reward_threshold", 110.0))
        tier4_reward = float(self.router_cfg.get("tier4_reward_quantile_threshold", 220.0))

        # On first attempt (attempt == 0), strongly prefer `-noreason` models to save 4x-8x reasoning tokens
        if attempt == 0:
            if tf.domain in {"search", "data_analysis"}:
                return MODEL_TIER_1_FAST_NOREASON
            if tf.domain == "coding":
                if tf.ast_cyclomatic_complexity >= tier2_complexity or tf.test_assert_count >= 4:
                    return MODEL_TIER_2_DENSE_NOREASON
                return MODEL_TIER_1_FAST_NOREASON
            if tf.domain == "math":
                if tf.reward >= tier4_reward and tf.formula_count >= 5:
                    return MODEL_TIER_2_DENSE_NOREASON
                return MODEL_TIER_1_FAST_NOREASON
            return MODEL_TIER_1_FAST_NOREASON

        # Escalation attempt (attempt >= 1): local verification in /tmp failed on attempt 0
        if qualifying_mode:
            # In qualifying stage (<= 5 tasks), prioritize maximum accuracy on retry
            return MODEL_TIER_4_DENSE_REASON if tf.domain in {"coding", "math"} else MODEL_TIER_3_MOE_REASON

        if tf.reward >= tier4_reward and tf.domain in {"coding", "math"}:
            return MODEL_TIER_4_DENSE_REASON
        if tf.reward >= tier3_reward:
            return MODEL_TIER_3_MOE_REASON
        return MODEL_TIER_2_DENSE_NOREASON

    def max_tokens_for_domain(self, domain: str, is_reasoning_model: bool = False) -> int:
        limits = self.estimator.policy.get("domain_max_tokens", {})
        base = int(limits.get(domain, 512))
        return base * 2 if is_reasoning_model else base
