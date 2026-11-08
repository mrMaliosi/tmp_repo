"""Classical ML predictor for pre-run token cost, p90 quantile budget, and task feasibility."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.task_scanner import TaskFeatures

ALLOWED_MODELS: tuple[str, ...] = (
    "qwen3.6-35b-a3b-noreason",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
)


@dataclass(frozen=True)
class TaskPrediction:
    """Predicted token metrics and feasibility probabilities for a task and model."""

    task_id: str
    model: str
    expected_tokens: float
    p90_tokens: float
    prob_feasible: float
    prob_success: float


class TokenAndFeasibilityEstimator:
    """Classical regression & logistic estimator whose coefficients are evolved offline by EBLO."""

    def __init__(self, policy_data: dict[str, Any] | None = None) -> None:
        self.policy = policy_data or self.default_policy()

    @staticmethod
    def default_policy() -> dict[str, Any]:
        return {
            "version": "1.0.0-eblo-evolved",
            "domain_base_cost": {
                "math": 170.0,
                "coding": 340.0,
                "data_analysis": 210.0,
                "search": 240.0,
            },
            "domain_p90_multiplier": {
                "math": 1.55,
                "coding": 1.75,
                "data_analysis": 1.45,
                "search": 1.50,
            },
            "model_cost_multipliers": {
                "qwen3.6-35b-a3b-noreason": 1.0,
                "qwen3.8-27b-noreason": 1.25,
                "qwen3.6-35b-a3b": 3.4,
                "qwen3.8-27b": 4.6,
            },
            "model_base_success": {
                "qwen3.6-35b-a3b-noreason": 0.84,
                "qwen3.8-27b-noreason": 0.90,
                "qwen3.6-35b-a3b": 0.94,
                "qwen3.8-27b": 0.97,
            },
            "complexity_weights": {
                "prompt_word_weight": 0.45,
                "ast_complexity_weight": 14.0,
                "test_assert_weight": 9.5,
                "csv_col_weight": 6.0,
                "kb_doc_weight": 3.2,
            },
            "feasibility_thresholds": {
                "min_prob_feasible": 0.32,
                "probe_reward_threshold": 80.0,
                "probe_suspicion_low": 0.25,
                "probe_suspicion_high": 0.65,
            },
            "scheduler": {
                "overhead_tokens": 45.0,
                "risk_exponent": 1.15,
            },
            "router": {
                "tier2_coding_complexity": 5,
                "tier3_reward_threshold": 110.0,
                "tier4_reward_quantile_threshold": 220.0,
            },
            "domain_max_tokens": {
                "math": 512,
                "coding": 768,
                "data_analysis": 512,
                "search": 384,
            },
            "system_prompts": {
                "math": (
                    "You are a precise mathematical solver. Write a compact, self-contained Python 3 script "
                    "that computes the exact answer and prints ONLY the final value to stdout."
                ),
                "coding": (
                    "You are an expert Python bug-repair engineer. Return ONLY the corrected Python source code "
                    "for the target module inside a ```python ... ``` block so all unit tests pass."
                ),
                "data_analysis": (
                    "You are a fast data analysis engineer. Given the CSV schema and sample rows, write a short "
                    "Python 3 script using the standard `csv` module that reads `CSV_PATH` and prints ONLY "
                    "the exact requested answer."
                ),
                "search": (
                    "You are a factual synthesis agent. Using ONLY the provided knowledge base snippets, "
                    "output the exact, concise final answer on a single line."
                ),
            },
        }

    @classmethod
    def from_json(cls, path: str | Path) -> TokenAndFeasibilityEstimator:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(policy_data=data)

    def predict_feasibility_prob(self, tf: TaskFeatures) -> float:
        """Estimate P(feasible | x_i) using static anomaly features and logistic scoring."""
        # Hard static contradictions immediately drive probability near 0
        if tf.has_syntax_error_in_tests or tf.contradictory_asserts:
            return 0.02
        if tf.missing_requested_columns:
            return 0.03
        if tf.domain == "data_analysis" and tf.csv_row_count == 0:
            return 0.02
        if tf.missing_query_entities:
            return 0.03
        if tf.kb_numeric_contradictions > 0:
            return 0.04
        if tf.math_paradox_markers > 0:
            return 0.02

        # Soft statistical anomaly scoring
        logit = 2.6
        if tf.domain == "data_analysis" and tf.csv_row_count > 0:
            corrupt_ratio = tf.csv_nan_or_corrupt_rows / max(1, tf.csv_row_count)
            if corrupt_ratio > 0.85:
                logit -= 2.4
            elif corrupt_ratio > 0.4:
                logit -= 0.8
        if tf.domain == "search" and tf.kb_doc_count == 0:
            logit -= 4.0
        if tf.constraint_markers > 6:
            logit -= 0.6

        prob = 1.0 / (1.0 + math.exp(-logit))
        return round(float(prob), 4)

    def estimate_for_model(self, tf: TaskFeatures, model: str) -> TaskPrediction:
        """Predict expected effective tokens, p90 token cap, P(feasible), and P(success | model)."""
        base_map = self.policy.get("domain_base_cost", {})
        base_cost = float(base_map.get(tf.domain, 220.0))

        cw = self.policy.get("complexity_weights", {})
        comp_add = (
            float(cw.get("prompt_word_weight", 0.45)) * tf.prompt_word_count
            + float(cw.get("ast_complexity_weight", 14.0)) * tf.ast_cyclomatic_complexity
            + float(cw.get("test_assert_weight", 9.5)) * tf.test_assert_count
            + float(cw.get("csv_col_weight", 6.0)) * tf.csv_col_count
            + float(cw.get("kb_doc_weight", 3.2)) * min(tf.kb_doc_count, 25)
        )

        model_mult = float(
            self.policy.get("model_cost_multipliers", {}).get(model, 1.0)
        )
        expected_tokens = (base_cost + comp_add) * model_mult

        p90_mult = float(
            self.policy.get("domain_p90_multiplier", {}).get(tf.domain, 1.55)
        )
        p90_tokens = max(120.0, expected_tokens * p90_mult)

        prob_feasible = self.predict_feasibility_prob(tf)

        base_succ = float(
            self.policy.get("model_base_success", {}).get(model, 0.85)
        )
        # Adjust success probability for task difficulty
        difficulty_penalty = min(
            0.35,
            0.012 * tf.ast_cyclomatic_complexity
            + 0.008 * tf.test_assert_count
            + 0.015 * tf.formula_count,
        )
        if model.endswith("-noreason"):
            prob_success = max(0.35, base_succ - difficulty_penalty)
        else:
            prob_success = max(0.55, base_succ - 0.4 * difficulty_penalty)

        return TaskPrediction(
            task_id=tf.task_id,
            model=model,
            expected_tokens=round(expected_tokens, 2),
            p90_tokens=round(p90_tokens, 2),
            prob_feasible=prob_feasible,
            prob_success=round(prob_success, 4),
        )
