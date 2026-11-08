"""Offline EBLO evolutionary runner that evolves `CompetitionChromosome` policies and exports
the winning configuration to `harness/evolved_artifacts/best_policy.json` and `submission/evolved_policy.py`.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eblo.competition_case.fitness import TournamentFitnessMetrics
from eblo.competition_case.tournament_sim import TournamentSimulator
from harness.estimator import TokenAndFeasibilityEstimator


@dataclass(frozen=True)
class CompetitionChromosome:
    """Genotype encoding estimator weights, 4-model router thresholds, scheduler risk, and domain prompts."""

    id: str
    generation: int
    policy: dict[str, Any]


class CompetitionEvolutionOptimizer:
    """Runs EBLO evolutionary optimization over `CompetitionChromosome` individuals on 100-task tournaments."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self.simulator = TournamentSimulator(seed=2026)

    def _create_initial_population(self, pop_size: int) -> list[CompetitionChromosome]:
        base = TokenAndFeasibilityEstimator.default_policy()
        pop: list[CompetitionChromosome] = [
            CompetitionChromosome(id="gen0_baseline", generation=0, policy=base)
        ]
        for idx in range(1, pop_size):
            mutated = self._mutate_policy(base)
            pop.append(
                CompetitionChromosome(
                    id=f"gen0_cand_{idx}",
                    generation=0,
                    policy=mutated,
                )
            )
        return pop

    def _mutate_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = json.loads(json.dumps(policy))

        # Mutate domain base cost estimates
        for dom in ("math", "coding", "data_analysis", "search"):
            cur = float(cloned["domain_base_cost"][dom])
            cloned["domain_base_cost"][dom] = round(
                max(80.0, min(600.0, cur * self._rng.uniform(0.85, 1.15))),
                1,
            )

        # Mutate p90 multipliers
        for dom in ("math", "coding", "data_analysis", "search"):
            cur_m = float(cloned["domain_p90_multiplier"][dom])
            cloned["domain_p90_multiplier"][dom] = round(
                max(1.25, min(2.2, cur_m + self._rng.uniform(-0.12, 0.12))),
                2,
            )

        # Mutate scheduler parameters
        cloned["scheduler"]["overhead_tokens"] = round(
            max(15.0, min(120.0, float(cloned["scheduler"]["overhead_tokens"]) + self._rng.uniform(-15.0, 15.0))),
            1,
        )
        cloned["scheduler"]["risk_exponent"] = round(
            max(0.8, min(2.0, float(cloned["scheduler"]["risk_exponent"]) + self._rng.uniform(-0.2, 0.2))),
            2,
        )

        # Mutate router thresholds
        cloned["router"]["tier2_coding_complexity"] = int(
            max(3, min(10, int(cloned["router"]["tier2_coding_complexity"]) + self._rng.choice([-1, 0, 1])))
        )
        cloned["router"]["tier3_reward_threshold"] = round(
            max(50.0, min(250.0, float(cloned["router"]["tier3_reward_threshold"]) + self._rng.uniform(-25.0, 25.0))),
            1,
        )
        cloned["router"]["tier4_reward_quantile_threshold"] = round(
            max(
                150.0,
                min(400.0, float(cloned["router"]["tier4_reward_quantile_threshold"]) + self._rng.uniform(-30.0, 30.0)),
            ),
            1,
        )
        return cloned

    def _crossover_policies(self, pol_a: dict[str, Any], pol_b: dict[str, Any]) -> dict[str, Any]:
        child: dict[str, Any] = json.loads(json.dumps(pol_a))
        child["router"] = json.loads(json.dumps(pol_b["router"]))
        child["scheduler"] = json.loads(json.dumps(pol_b["scheduler"]))
        for dom in ("math", "coding", "data_analysis", "search"):
            if self._rng.random() < 0.5:
                child["domain_base_cost"][dom] = pol_b["domain_base_cost"][dom]
                child["domain_p90_multiplier"][dom] = pol_b["domain_p90_multiplier"][dom]
        return child

    async def evolve_and_export(
        self,
        generations: int = 3,
        pop_size: int = 6,
        num_tasks: int = 100,
        budgets: tuple[float, ...] = (6_000.0, 15_000.0, 40_000.0),
        artifact_path: str | Path | None = None,
        submission_py_path: str | Path | None = None,
    ) -> tuple[CompetitionChromosome, float, list[TournamentFitnessMetrics]]:
        pop = self._create_initial_population(pop_size)
        best_chromo = pop[0]
        best_score = -1e18
        best_metrics: list[TournamentFitnessMetrics] = []

        for gen in range(generations):
            scored: list[tuple[float, CompetitionChromosome, list[TournamentFitnessMetrics]]] = []
            for ch in pop:
                avg_fit, metrics = await self.simulator.evaluate_multi_budget(
                    ch.policy,
                    budgets=budgets,
                    num_tasks=num_tasks,
                )
                scored.append((avg_fit, ch, metrics))

            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best_chromo, best_metrics = scored[0]

            # Elitism + breeding for next generation
            elites = [scored[0][1], scored[1][1]] if len(scored) >= 2 else [scored[0][1]]
            next_pop: list[CompetitionChromosome] = list(elites)
            while len(next_pop) < pop_size:
                pa = self._rng.choice(elites)
                pb = self._rng.choice([s[1] for s in scored[: max(2, len(scored) // 2)]])
                crossed = self._crossover_policies(pa.policy, pb.policy)
                mutated = self._mutate_policy(crossed)
                next_pop.append(
                    CompetitionChromosome(
                        id=f"gen{gen + 1}_cand_{len(next_pop)}",
                        generation=gen + 1,
                        policy=mutated,
                    )
                )
            pop = next_pop

        # Export best policy JSON and Python module
        root_dir = Path(__file__).resolve().parent.parent.parent
        target_json = (
            Path(artifact_path)
            if artifact_path
            else root_dir / "harness" / "evolved_artifacts" / "best_policy.json"
        )
        target_json.parent.mkdir(parents=True, exist_ok=True)

        export_payload = {
            **best_chromo.policy,
            "eblo_metadata": {
                "winner_chromosome_id": best_chromo.id,
                "generation": best_chromo.generation,
                "mean_tournament_fitness": best_score,
                "budgets_evaluated": list(budgets),
                "metrics_per_budget": [
                    {
                        "budget": b,
                        "reward_earned": m.total_reward_earned,
                        "possible_reward": m.total_possible_reward,
                        "effective_tokens_used": m.effective_tokens_used,
                        "tasks_solved": m.tasks_solved,
                        "traps_filtered": m.traps_correctly_filtered,
                        "fitness": m.tournament_fitness,
                    }
                    for b, m in zip(budgets, best_metrics)
                ],
            },
        }
        target_json.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        target_py = (
            Path(submission_py_path)
            if submission_py_path
            else root_dir / "submission" / "evolved_policy.py"
        )
        target_py.parent.mkdir(parents=True, exist_ok=True)
        target_py.write_text(
            '"""Auto-exported evolved EBLO tournament policy for the openJiuwen harness."""\n\n'
            f"EVOLVED_POLICY = {repr(export_payload)}\n",
            encoding="utf-8",
        )

        return best_chromo, best_score, best_metrics


async def main() -> None:
    optimizer = CompetitionEvolutionOptimizer(seed=42)
    # Budget grid sits in the deficit zone of the calibrated simulator (the full
    # tournament costs ~5.7K C_eff): tight 6K, medium 15K, generous 40K.
    budgets = (6_000.0, 15_000.0, 40_000.0)
    chromo, score, metrics = await optimizer.evolve_and_export(generations=3, pop_size=6, num_tasks=100, budgets=budgets)
    print(f"Exported best EBLO policy ({chromo.id}) with mean multi-budget fitness: {score:.2f}")
    for b, m in zip(budgets, metrics):
        print(
            f"  Budget {int(b):>7d} -> Solved: {m.tasks_solved}/{m.feasible_tasks_total}, "
            f"Traps Filtered: {m.traps_correctly_filtered}/{m.traps_total}, "
            f"Reward: {m.total_reward_earned:.0f}/{m.total_possible_reward:.0f}, "
            f"C_eff: {m.effective_tokens_used:.1f}, Fitness: {m.tournament_fitness:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
