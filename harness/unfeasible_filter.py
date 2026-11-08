"""Three-tier Unfeasible Task Detector (Tier 0 Static, Tier 1 Cheap Probe, Tier 2 In-Flight Degradation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.estimator import TokenAndFeasibilityEstimator
from harness.model_router import MODEL_TIER_1_FAST_NOREASON
from harness.task_scanner import TaskFeatures


@dataclass(frozen=True)
class FilterDecision:
    """Decision returned by the unfeasible task filter."""

    is_feasible: bool
    tier_triggered: int  # 0 = static, 1 = probe, 2 = in-flight, -1 = passed
    reason: str
    prob_feasible: float


class UnfeasibleTaskFilter:
    """Three-stage defense against the 15-25% unsolvable trap tasks in the competition."""

    def __init__(
        self,
        estimator: TokenAndFeasibilityEstimator,
        qualifying_mode: bool = False,
    ) -> None:
        self.estimator = estimator
        self.qualifying_mode = qualifying_mode
        self.thresholds: dict[str, Any] = estimator.policy.get("feasibility_thresholds", {})

    def evaluate_tier0_static(self, tf: TaskFeatures) -> FilterDecision:
        """Tier 0: Zero-token static workspace & prompt contradiction analysis."""
        prob = self.estimator.predict_feasibility_prob(tf)
        if self.qualifying_mode:
            # Stage 3 qualifying run has <= 5 solvable tasks; disable aggressive filtering
            return FilterDecision(
                is_feasible=True,
                tier_triggered=-1,
                reason="Qualifying mode active: static filter bypassed.",
                prob_feasible=max(0.95, prob),
            )

        if tf.has_syntax_error_in_tests:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason="Static Tier-0: SyntaxError detected in workspace unit test file.",
                prob_feasible=prob,
            )
        if tf.contradictory_asserts:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason="Static Tier-0: Mutually contradictory assertions detected in test_mylib.py.",
                prob_feasible=prob,
            )
        if tf.missing_requested_columns:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason=f"Static Tier-0: Requested column(s) {tf.missing_requested_columns} missing from CSV.",
                prob_feasible=prob,
            )
        if tf.domain == "data_analysis" and tf.csv_row_count == 0:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason="Static Tier-0: CSV file is empty (0 data rows).",
                prob_feasible=prob,
            )
        if tf.missing_query_entities:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason=f"Static Tier-0: Query entities {tf.missing_query_entities} absent from workspace/kb/*.md.",
                prob_feasible=prob,
            )
        if tf.kb_numeric_contradictions > 0:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason="Static Tier-0: Conflicting numeric facts for same entity across KB documents.",
                prob_feasible=prob,
            )
        if tf.math_paradox_markers > 0:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason="Static Tier-0: Unsatisfiable system or mathematical paradox detected in prompt.",
                prob_feasible=prob,
            )

        min_prob = float(self.thresholds.get("min_prob_feasible", 0.32))
        if prob < min_prob:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=0,
                reason=f"Static Tier-0: Feasibility probability {prob:.2f} below threshold {min_prob:.2f}.",
                prob_feasible=prob,
            )

        return FilterDecision(
            is_feasible=True,
            tier_triggered=-1,
            reason="Passed Tier-0 static checks.",
            prob_feasible=prob,
        )

    def should_run_tier1_probe(self, tf: TaskFeatures, prob_feasible: float) -> bool:
        """Check whether a suspicious high-reward task warrants a cheap 120-token probe."""
        if self.qualifying_mode:
            return False
        reward_thresh = float(self.thresholds.get("probe_reward_threshold", 80.0))
        low = float(self.thresholds.get("probe_suspicion_low", 0.25))
        high = float(self.thresholds.get("probe_suspicion_high", 0.65))
        return tf.reward >= reward_thresh and (low <= prob_feasible <= high)

    async def evaluate_tier1_probe(
        self,
        tf: TaskFeatures,
        agent_gateway: Any,
        workspace_summary: str,
    ) -> FilterDecision:
        """Tier 1: Cheap probe request using `qwen3.6-35b-a3b-noreason` with `max_tokens=120`."""
        prob = self.estimator.predict_feasibility_prob(tf)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict task feasibility verifier. Classify whether the task is "
                    "SOLVABLE or UNSOLVABLE_TRAP based on the prompt and workspace facts. "
                    "Reply with SOLVABLE or UNSOLVABLE_TRAP on the first line, followed by a 1-sentence reason."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task ID: {tf.task_id} (Domain: {tf.domain})\n"
                    f"Prompt: {tf.prompt[:600]}\n"
                    f"Workspace Summary: {workspace_summary[:500]}"
                ),
            },
        ]
        raw_reply = await agent_gateway.chat_completion(
            task_id=tf.task_id,
            model=MODEL_TIER_1_FAST_NOREASON,
            messages=messages,
            max_tokens=120,
            temperature=0.0,
        )
        if "UNSOLVABLE_TRAP" in raw_reply.upper():
            return FilterDecision(
                is_feasible=False,
                tier_triggered=1,
                reason=f"Tier-1 Probe classified as UNSOLVABLE_TRAP: {raw_reply.strip()[:120]}",
                prob_feasible=0.05,
            )
        return FilterDecision(
            is_feasible=True,
            tier_triggered=-1,
            reason="Passed Tier-1 probe.",
            prob_feasible=max(0.75, prob),
        )

    @staticmethod
    def evaluate_tier2_inflight(
        error_counts: list[int],
        missing_resource_detected: bool = False,
    ) -> FilterDecision:
        """Tier 2: In-flight degradation check.
        If 2 consecutive local test runs fail without reducing error count, or a missing workspace
        resource is requested, immediately abort the task as UNFEASIBLE.
        """
        if missing_resource_detected:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=2,
                reason="Tier-2 In-Flight: Solver requested missing workspace resource/column.",
                prob_feasible=0.0,
            )
        if len(error_counts) >= 2 and error_counts[-1] >= error_counts[-2] and error_counts[-1] > 0:
            return FilterDecision(
                is_feasible=False,
                tier_triggered=2,
                reason=(
                    f"Tier-2 In-Flight: 2 consecutive test failures without error reduction "
                    f"(errors: {error_counts[-2]} -> {error_counts[-1]})."
                ),
                prob_feasible=0.0,
            )
        return FilterDecision(
            is_feasible=True,
            tier_triggered=-1,
            reason="In-flight progress healthy.",
            prob_feasible=1.0,
        )
