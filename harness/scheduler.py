"""Portfolio Knapsack Scheduler ranking tasks by expected ROI per effective token."""

from __future__ import annotations

from dataclasses import dataclass

from harness.estimator import TaskPrediction, TokenAndFeasibilityEstimator
from harness.model_router import ModelRouter
from harness.task_scanner import TaskFeatures
from harness.unfeasible_filter import FilterDecision, UnfeasibleTaskFilter


@dataclass(frozen=True)
class ScheduledTaskItem:
    """Task entry prioritized by Expected ROI Score."""

    features: TaskFeatures
    initial_model: str
    prediction: TaskPrediction
    filter_decision: FilterDecision
    roi_score: float


class PortfolioKnapsackScheduler:
    """Ranks all feasible tasks by Expected ROI:
    Score_i = (reward_i * P_i(feasible) * P_i(success | m*)) / (C_{i, p90}(m*) + C_overhead)
    """

    def __init__(
        self,
        estimator: TokenAndFeasibilityEstimator,
        router: ModelRouter,
        unfeasible_filter: UnfeasibleTaskFilter,
    ) -> None:
        self.estimator = estimator
        self.router = router
        self.unfeasible_filter = unfeasible_filter
        sched_cfg = estimator.policy.get("scheduler", {})
        self.overhead_tokens = float(sched_cfg.get("overhead_tokens", 45.0))
        self.risk_exponent = float(sched_cfg.get("risk_exponent", 1.15))

    def build_schedule(
        self,
        tasks: list[TaskFeatures],
        qualifying_mode: bool = False,
    ) -> tuple[list[ScheduledTaskItem], list[ScheduledTaskItem]]:
        """Partition scanned tasks into `(prioritized_active_queue, rejected_unfeasible_list)`."""
        active_queue: list[ScheduledTaskItem] = []
        rejected: list[ScheduledTaskItem] = []

        for tf in tasks:
            decision = self.unfeasible_filter.evaluate_tier0_static(tf)
            model = self.router.select_model(tf, attempt=0, qualifying_mode=qualifying_mode)
            pred = self.estimator.estimate_for_model(tf, model)

            if not decision.is_feasible and not qualifying_mode:
                rejected.append(
                    ScheduledTaskItem(
                        features=tf,
                        initial_model=model,
                        prediction=pred,
                        filter_decision=decision,
                        roi_score=0.0,
                    )
                )
                continue

            prob_feas = 1.0 if qualifying_mode else decision.prob_feasible
            numerator = float(tf.reward) * (prob_feas**self.risk_exponent) * pred.prob_success
            denominator = pred.p90_tokens + self.overhead_tokens
            score = numerator / max(1.0, denominator)

            active_queue.append(
                ScheduledTaskItem(
                    features=tf,
                    initial_model=model,
                    prediction=pred,
                    filter_decision=decision,
                    roi_score=round(score, 6),
                )
            )

        # Sort descending by ROI score, tie-breaking by higher reward and lower expected tokens
        active_queue.sort(
            key=lambda item: (
                item.roi_score,
                item.features.reward,
                -item.prediction.expected_tokens,
            ),
            reverse=True,
        )
        return active_queue, rejected
