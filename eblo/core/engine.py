"""Main evolutionary orchestrator engine supporting Generational, Steady-State, and Island Model execution."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.table import Table

from eblo.config.schema import EBLOConfig
from eblo.core.domain import Chromosome, Individual, Phenotype, TaskSpec
from eblo.core.ports import (
    CrossoverOperatorPort,
    LLMClientPort,
    MutationOperatorPort,
    PhenotypeValidatorPort,
    PopulationInitializerPort,
    SelectionOperatorPort,
)
from eblo.llm.client import AsyncLLMClient
from eblo.llm.parser import parse_phenotype
from eblo.modules.initializers import build_initializer
from eblo.modules.operators import (
    HyperparameterJitterMutator,
    OrthogonalShiftMutator,
    ReflexionFeedbackMutator,
    SemanticSynthesisCrossover,
    StructuralBlockCrossover,
)
from eblo.modules.selectors import build_selector, sort_by_scalar_fitness
from eblo.modules.validators import CascadeValidator
from eblo.storage.lineage import ASTFitnessCache, LineageRecorder


@dataclass(frozen=True)
class GenerationMetrics:
    """Telemetry snapshot for a single evolutionary generation."""

    generation: int
    best_score: float
    best_pass_rate: float
    best_time_ms: float
    cache_hits: int
    active_mutation: str


@dataclass(frozen=True)
class EvolutionRunResult:
    """Final result of an EBLO evolutionary run."""

    run_id: str
    best_individual: Individual
    history: list[GenerationMetrics]
    final_population: list[Individual]
    lineage_jsonl_path: str
    best_solution_path: str


class EvolutionEngine:
    """High-performance asynchronous evolutionary orchestrator."""

    def __init__(
        self,
        config: EBLOConfig,
        llm_client: LLMClientPort | None = None,
        initializer: PopulationInitializerPort | None = None,
        validator: PhenotypeValidatorPort | None = None,
        selector: SelectionOperatorPort | None = None,
        seed: int | None = 42,
        console: Console | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._semaphore = asyncio.Semaphore(config.llm.max_concurrent)
        self.console = console or Console()

        self.llm_client: LLMClientPort = llm_client or AsyncLLMClient(config.llm)
        self.cache = ASTFitnessCache(
            backend=config.validator.cache_backend,
            db_path=config.validator.cache_db_path,
        )
        self.initializer: PopulationInitializerPort = initializer or build_initializer(
            config.initializer,
            self.llm_client,
            seed=seed,
        )
        self.validator: PhenotypeValidatorPort = validator or CascadeValidator(
            config.validator,
            cache=self.cache,
        )
        self.selector: SelectionOperatorPort = selector or build_selector(
            config.selection,
            seed=seed,
        )

        self.crossover_ops: dict[str, CrossoverOperatorPort] = {
            "structural_block": StructuralBlockCrossover(),
            "semantic_synthesis": SemanticSynthesisCrossover(self.llm_client),
        }
        self.mutation_ops: dict[str, MutationOperatorPort] = {
            "reflexion_feedback": ReflexionFeedbackMutator(self.llm_client),
            "orthogonal_shift": OrthogonalShiftMutator(self.llm_client),
            "hyperparameter_jitter": HyperparameterJitterMutator(
                constraints_pool=config.initializer.constraints_pool,
                seed=seed,
            ),
        }

    def _pick_crossover_op(self) -> CrossoverOperatorPort:
        specs = self.config.variation.crossovers
        weights = [s.weight for s in specs]
        chosen = self._rng.choices(specs, weights=weights, k=1)[0]
        return self.crossover_ops[chosen.type]

    def _pick_mutation_op(self) -> MutationOperatorPort:
        specs = self.config.variation.mutations
        weights = [s.weight for s in specs]
        chosen = self._rng.choices(specs, weights=weights, k=1)[0]
        return self.mutation_ops[chosen.type]

    async def _synthesize_and_evaluate(
        self,
        chromosome: Chromosome,
        task: TaskSpec,
        island_id: int = 0,
    ) -> Individual:
        """Render chromosome prompt, invoke LLM under concurrency semaphore, and run cascade validator."""
        messages = chromosome.render_prompt(task)
        async with self._semaphore:
            raw_output, latency_ms = await self.llm_client.complete(
                messages=messages,
                temperature=chromosome.temperature,
                json_mode=False,
            )
        phenotype: Phenotype = parse_phenotype(
            chromosome_id=chromosome.id,
            raw_response=raw_output,
            generation_latency_ms=latency_ms,
        )
        fitness = await self.validator.evaluate(phenotype, task)
        return Individual(
            chromosome=chromosome,
            phenotype=phenotype,
            fitness=fitness,
            island_id=island_id,
        )

    async def _evaluate_chromosomes(
        self,
        chromosomes: list[Chromosome],
        task: TaskSpec,
        island_ids: list[int] | None = None,
    ) -> list[Individual]:
        if island_ids is None:
            island_ids = [0] * len(chromosomes)
        coros = [
            self._synthesize_and_evaluate(ch, task, island_id=isl_id)
            for ch, isl_id in zip(chromosomes, island_ids)
        ]
        return list(await asyncio.gather(*coros))

    async def _breed_offspring_chromosome(
        self,
        population: list[Individual],
        task: TaskSpec,
        next_generation: int,
    ) -> tuple[Chromosome, str]:
        """Create one child chromosome via crossover and/or mutation according to variation rates."""
        total_rate = self.config.variation.crossover_rate + self.config.variation.mutation_rate
        cross_prob = (
            self.config.variation.crossover_rate / total_rate if total_rate > 0 else 0.5
        )

        if len(population) >= 2 and self._rng.random() < cross_prob:
            parents = self.selector.select_parents(population, 2)
            cross_op = self._pick_crossover_op()
            async with self._semaphore:
                child = await cross_op.crossover(parents[0], parents[1], task, next_generation)
            op_label = f"crossover:{cross_op.name}"
            # Optionally chain mutation if both rates are high and random trigger hits
            if self._rng.random() < (self.config.variation.mutation_rate * 0.35):
                mut_op = self._pick_mutation_op()
                temp_ind = Individual(
                    chromosome=child,
                    phenotype=parents[0].phenotype,
                    fitness=parents[0].fitness,
                    island_id=parents[0].island_id,
                )
                async with self._semaphore:
                    child = await mut_op.mutate(temp_ind, task, next_generation)
                op_label = f"{cross_op.name}+{mut_op.name}"
            return child, op_label

        parent = self.selector.select_parents(population, 1)[0]
        mut_op = self._pick_mutation_op()
        async with self._semaphore:
            child = await mut_op.mutate(parent, task, next_generation)
        return child, f"mutate:{mut_op.name}"

    async def _step_subpopulation_generational(
        self,
        subpop: list[Individual],
        task: TaskSpec,
        next_generation: int,
        island_id: int = 0,
    ) -> tuple[list[Individual], str]:
        """Advance a population or island by one generation with elitism and parallel offspring generation."""
        target_size = len(subpop)
        elite_count = min(self.config.selection.elitism_count, max(1, target_size - 1))
        elites_raw = self.selector.select_elites(subpop, elite_count)
        elites = [
            Individual(
                chromosome=e.chromosome,
                phenotype=e.phenotype,
                fitness=e.fitness,
                island_id=island_id,
            )
            for e in elites_raw
        ]

        offspring_needed = max(0, target_size - len(elites))
        if offspring_needed == 0:
            return elites, "elitism_only"

        breed_coros = [
            self._breed_offspring_chromosome(subpop, task, next_generation)
            for _ in range(offspring_needed)
        ]
        bred_results = await asyncio.gather(*breed_coros)
        offspring_chromosomes = [res[0] for res in bred_results]
        op_labels = [res[1] for res in bred_results]

        # Dominant active mutation/operator label for telemetry
        active_op = max(set(op_labels), key=op_labels.count) if op_labels else "none"
        offspring_individuals = await self._evaluate_chromosomes(
            offspring_chromosomes,
            task,
            island_ids=[island_id] * len(offspring_chromosomes),
        )
        return elites + offspring_individuals, active_op

    async def _step_steady_state(
        self,
        population: list[Individual],
        task: TaskSpec,
        next_generation: int,
    ) -> tuple[list[Individual], str]:
        """Steady-state evolutionary step: generate max_concurrent replacements and keep top individuals."""
        batch_size = min(max(2, self.config.llm.max_concurrent), max(1, len(population) // 2))
        breed_coros = [
            self._breed_offspring_chromosome(population, task, next_generation)
            for _ in range(batch_size)
        ]
        bred_results = await asyncio.gather(*breed_coros)
        offspring_chromosomes = [res[0] for res in bred_results]
        op_labels = [res[1] for res in bred_results]
        active_op = max(set(op_labels), key=op_labels.count) if op_labels else "steady_state"

        offspring_individuals = await self._evaluate_chromosomes(offspring_chromosomes, task)
        combined = sort_by_scalar_fitness(population + offspring_individuals)
        return combined[: len(population)], active_op

    @staticmethod
    def _migrate_ring_islands(islands: list[list[Individual]]) -> list[list[Individual]]:
        """Exchange the top individual between islands along a unidirectional ring topology."""
        num_islands = len(islands)
        if num_islands <= 1:
            return islands

        migrants = [sort_by_scalar_fitness(isl)[0] for isl in islands]
        updated_islands: list[list[Individual]] = []

        for idx in range(num_islands):
            source_idx = (idx - 1) % num_islands
            incoming = migrants[source_idx]
            migrated_ind = Individual(
                chromosome=Chromosome(
                    id=incoming.chromosome.id,
                    generation=incoming.chromosome.generation,
                    parent_ids=incoming.chromosome.parent_ids,
                    system_persona=incoming.chromosome.system_persona,
                    algorithmic_strategy=incoming.chromosome.algorithmic_strategy,
                    constraints=incoming.chromosome.constraints,
                    temperature=incoming.chromosome.temperature,
                    mutation_history=incoming.chromosome.mutation_history + (f"migrate:{source_idx}->{idx}",),
                ),
                phenotype=incoming.phenotype,
                fitness=incoming.fitness,
                island_id=idx,
            )
            ranked_current = sort_by_scalar_fitness(islands[idx])
            # Replace the weakest individual in the destination island with the incoming champion
            new_island = ranked_current[:-1] + [migrated_ind]
            updated_islands.append(new_island)

        return updated_islands

    async def run(
        self,
        task: TaskSpec,
        run_id: str | None = None,
        show_dashboard: bool = True,
        on_generation: Callable[[GenerationMetrics], None] | None = None,
    ) -> EvolutionRunResult:
        """Execute full evolutionary loop from Generation 0 to Generation N."""
        effective_run_id = run_id or f"{self.config.experiment.name}_{int(time.time())}"
        recorder = LineageRecorder(self.config.experiment.runs_dir, effective_run_id)

        pop_size = self.config.experiment.population_size
        mode = self.config.experiment.evolution_mode
        total_generations = self.config.experiment.generations

        # Generation 0 Initialization
        init_chromosomes = await self.initializer.initialize(task, pop_size)
        if mode == "island" and self.config.experiment.islands > 1:
            num_islands = min(self.config.experiment.islands, pop_size)
            island_ids = [idx % num_islands for idx in range(len(init_chromosomes))]
        else:
            num_islands = 1
            island_ids = [0] * len(init_chromosomes)

        population = await self._evaluate_chromosomes(init_chromosomes, task, island_ids=island_ids)

        history: list[GenerationMetrics] = []
        best_overall = sort_by_scalar_fitness(population)[0]

        gen0_metrics = GenerationMetrics(
            generation=0,
            best_score=best_overall.fitness.total_score,
            best_pass_rate=best_overall.fitness.pass_rate,
            best_time_ms=best_overall.fitness.execution_time_ms,
            cache_hits=self.cache.hits,
            active_mutation=f"init:{self.config.initializer.type}",
        )
        history.append(gen0_metrics)
        recorder.record_generation(0, population, gen0_metrics.active_mutation, self.cache.hits)
        if on_generation:
            on_generation(gen0_metrics)

        # Partition into islands if running in island mode
        islands: list[list[Individual]] = [[] for _ in range(num_islands)]
        for ind in population:
            islands[ind.island_id % num_islands].append(ind)

        # Evolution Loop: Generation 1 .. N
        for gen in range(1, total_generations + 1):
            if mode == "island" and num_islands > 1:
                island_tasks = [
                    self._step_subpopulation_generational(
                        islands[isl_idx],
                        task,
                        next_generation=gen,
                        island_id=isl_idx,
                    )
                    for isl_idx in range(num_islands)
                ]
                island_results = await asyncio.gather(*island_tasks)
                islands = [res[0] for res in island_results]
                island_ops = [res[1] for res in island_results]
                active_op = max(set(island_ops), key=island_ops.count)

                if gen % self.config.experiment.migration_interval == 0:
                    islands = self._migrate_ring_islands(islands)
                    active_op = f"{active_op}+ring_migration"

                population = [ind for isl in islands for ind in isl]
            elif mode == "steady_state":
                population, active_op = await self._step_steady_state(
                    population,
                    task,
                    next_generation=gen,
                )
            else:
                population, active_op = await self._step_subpopulation_generational(
                    population,
                    task,
                    next_generation=gen,
                    island_id=0,
                )

            # Enforce global monotonic elitism guarantee across generations
            current_best = sort_by_scalar_fitness(population)[0]
            if best_overall.fitness.total_score > current_best.fitness.total_score:
                ranked = sort_by_scalar_fitness(population)
                population = [best_overall] + ranked[:-1]
                current_best = best_overall
            else:
                best_overall = current_best

            metrics = GenerationMetrics(
                generation=gen,
                best_score=current_best.fitness.total_score,
                best_pass_rate=current_best.fitness.pass_rate,
                best_time_ms=current_best.fitness.execution_time_ms,
                cache_hits=self.cache.hits,
                active_mutation=active_op,
            )
            history.append(metrics)
            recorder.record_generation(gen, population, active_op, self.cache.hits)
            if on_generation:
                on_generation(metrics)

        best_code_path = recorder.save_best_solution(best_overall, task)

        if show_dashboard:
            self.render_dashboard(history, best_overall, effective_run_id)

        return EvolutionRunResult(
            run_id=effective_run_id,
            best_individual=best_overall,
            history=history,
            final_population=population,
            lineage_jsonl_path=str(recorder.jsonl_path),
            best_solution_path=str(best_code_path),
        )

    def render_dashboard(
        self,
        history: list[GenerationMetrics],
        best: Individual,
        run_id: str,
    ) -> None:
        """Render Rich terminal telemetry table for the evolutionary run."""
        table = Table(
            title=f"EBLO Evolutionary Progress — Run `{run_id}` (Mode: {self.config.experiment.evolution_mode})",
            show_lines=False,
        )
        table.add_column("Generation", justify="right", style="cyan", no_wrap=True)
        table.add_column("Best Score", justify="right", style="bold green")
        table.add_column("Best Pass Rate", justify="right", style="yellow")
        table.add_column("Best Time (ms)", justify="right", style="magenta")
        table.add_column("Cache Hits", justify="right", style="blue")
        table.add_column("Active Mutation", justify="left", style="white")

        for m in history:
            table.add_row(
                str(m.generation),
                f"{m.best_score:.4f}",
                f"{m.best_pass_rate * 100:.1f}%",
                f"{m.best_time_ms:.3f}",
                str(m.cache_hits),
                m.active_mutation,
            )
        self.console.print(table)
