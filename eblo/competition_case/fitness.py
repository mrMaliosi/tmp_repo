"""Tournament fitness function for the Budget-Aware Token Estimation competition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentFitnessMetrics:
    """Detailed evaluation metrics for a single or multi-budget tournament simulation."""

    total_reward_earned: float
    total_possible_reward: float
    effective_tokens_used: float
    tasks_solved: int
    feasible_tasks_total: int
    traps_correctly_filtered: int
    traps_total: int
    tournament_fitness: float


def compute_tournament_fitness(
    total_reward_earned: float,
    effective_tokens_used: float,
    token_tiebreak_weight: float = 0.01,
) -> float:
    """Official tournament fitness with tie-break penalty:
    Fitness = sum_{i in Solved} reward_i - 0.01 * C_eff_used
    """
    return round(float(total_reward_earned) - token_tiebreak_weight * float(effective_tokens_used), 4)
