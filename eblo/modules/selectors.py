"""Selection operators: Tournament Selection, Elitism, and NSGA-II Pareto Front Selection."""

from __future__ import annotations

import random

from eblo.config.schema import SelectionConfig
from eblo.core.domain import Individual
from eblo.core.ports import SelectionOperatorPort


def sort_by_scalar_fitness(population: list[Individual]) -> list[Individual]:
    """Sort individuals descending by total_score, tie-breaking by pass_rate and faster runtime."""
    return sorted(
        population,
        key=lambda ind: (
            ind.fitness.total_score,
            ind.fitness.pass_rate,
            -ind.fitness.execution_time_ms,
        ),
        reverse=True,
    )


class TournamentSelector:
    """Tournament selection with configurable tournament_size and elitism preservation."""

    def __init__(self, config: SelectionConfig, seed: int | None = None) -> None:
        self.config = config
        self._rng = random.Random(seed)

    def select_elites(self, population: list[Individual], elite_count: int) -> list[Individual]:
        if elite_count <= 0 or not population:
            return []
        ranked = sort_by_scalar_fitness(population)
        return ranked[: min(elite_count, len(ranked))]

    def select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        if not population or count <= 0:
            return []
        k = min(self.config.tournament_size, len(population))
        selected: list[Individual] = []
        for _ in range(count):
            contenders = self._rng.sample(population, k=k)
            winner = sort_by_scalar_fitness(contenders)[0]
            selected.append(winner)
        return selected


class NSGA2Selector:
    """Multi-objective Pareto-front selector (NSGA-II) with crowding distance over:
    Objective 1 (maximize): pass_rate
    Objective 2 (minimize): execution_time_ms (effective only when pass_rate > 0)
    Objective 3 (minimize): memory_peak_kb
    """

    def __init__(self, config: SelectionConfig, seed: int | None = None) -> None:
        self.config = config
        self._rng = random.Random(seed)

    @staticmethod
    def _objectives(ind: Individual) -> tuple[float, float, float]:
        # Convert all objectives to maximization form:
        # 1. pass_rate (higher is better)
        # 2. negative execution_time_ms if syntax & smoke valid, else -1e9
        # 3. negative memory_peak_kb
        if ind.fitness.pass_rate <= 0.0:
            return (0.0, -1e9, -1e9)
        return (
            ind.fitness.pass_rate,
            -ind.fitness.execution_time_ms,
            -ind.fitness.memory_peak_kb,
        )

    @classmethod
    def _dominates(cls, ind_a: Individual, ind_b: Individual) -> bool:
        obj_a = cls._objectives(ind_a)
        obj_b = cls._objectives(ind_b)
        at_least_as_good = all(a >= b for a, b in zip(obj_a, obj_b))
        strictly_better = any(a > b for a, b in zip(obj_a, obj_b))
        return at_least_as_good and strictly_better

    @classmethod
    def fast_non_dominated_sort(cls, population: list[Individual]) -> list[list[Individual]]:
        n = len(population)
        if n == 0:
            return []

        domination_sets: list[list[int]] = [[] for _ in range(n)]
        dominated_counts: list[int] = [0] * n
        fronts: list[list[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if cls._dominates(population[p], population[q]):
                    domination_sets[p].append(q)
                elif cls._dominates(population[q], population[p]):
                    dominated_counts[p] += 1
            if dominated_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    dominated_counts[q] -= 1
                    if dominated_counts[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return [[population[idx] for idx in front] for front in fronts if front]

    @classmethod
    def compute_crowding_distance(cls, front: list[Individual]) -> dict[str, float]:
        size = len(front)
        if size == 0:
            return {}
        distances: dict[str, float] = {ind.id: 0.0 for ind in front}
        if size <= 2:
            for ind in front:
                distances[ind.id] = float("inf")
            return distances

        num_objectives = 3
        for m in range(num_objectives):
            sorted_front = sorted(front, key=lambda ind: cls._objectives(ind)[m])
            min_val = cls._objectives(sorted_front[0])[m]
            max_val = cls._objectives(sorted_front[-1])[m]
            distances[sorted_front[0].id] = float("inf")
            distances[sorted_front[-1].id] = float("inf")
            span = max_val - min_val
            if span <= 1e-12:
                continue
            for idx in range(1, size - 1):
                prev_val = cls._objectives(sorted_front[idx - 1])[m]
                next_val = cls._objectives(sorted_front[idx + 1])[m]
                distances[sorted_front[idx].id] += (next_val - prev_val) / span

        return distances

    def _rank_population(self, population: list[Individual]) -> list[Individual]:
        fronts = self.fast_non_dominated_sort(population)
        ranked: list[Individual] = []
        for front in fronts:
            crowding = self.compute_crowding_distance(front)
            sorted_front = sorted(
                front,
                key=lambda ind: (crowding.get(ind.id, 0.0), ind.fitness.total_score),
                reverse=True,
            )
            ranked.extend(sorted_front)
        return ranked

    def select_elites(self, population: list[Individual], elite_count: int) -> list[Individual]:
        if elite_count <= 0 or not population:
            return []
        # Always guarantee the single highest total_score individual is preserved first
        best_scalar = sort_by_scalar_fitness(population)[0]
        pareto_ranked = self._rank_population(population)
        elites = [best_scalar]
        for ind in pareto_ranked:
            if len(elites) >= elite_count:
                break
            if ind.id != best_scalar.id:
                elites.append(ind)
        return elites[: min(elite_count, len(population))]

    def select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        if not population or count <= 0:
            return []
        ranked = self._rank_population(population)
        rank_Index = {ind.id: idx for idx, ind in enumerate(ranked)}
        selected: list[Individual] = []
        k = min(max(2, self.config.tournament_size), len(population))
        for _ in range(count):
            contenders = self._rng.sample(population, k=k)
            winner = min(contenders, key=lambda ind: rank_Index[ind.id])
            selected.append(winner)
        return selected


def build_selector(config: SelectionConfig, seed: int | None = None) -> SelectionOperatorPort:
    """Factory function returning the configured selection strategy."""
    if config.type == "nsga2":
        return NSGA2Selector(config, seed=seed)
    return TournamentSelector(config, seed=seed)
