"""Module 1: Population Initializers for generating P_0 chromosomes."""

from __future__ import annotations

import itertools
import random
from typing import Any
from uuid import uuid4

from eblo.config.schema import InitializerConfig
from eblo.core.domain import Chromosome, TaskSpec
from eblo.core.ports import LLMClientPort, PopulationInitializerPort
from eblo.llm.parser import extract_json_payload


class CombinatorialMatrixInitializer:
    """Zero-LLM initializer assembling chromosomes deterministically/stochastically from configured pools."""

    def __init__(self, config: InitializerConfig, seed: int | None = None) -> None:
        self.config = config
        self._rng = random.Random(seed)

    async def initialize(self, task: TaskSpec, pop_size: int) -> list[Chromosome]:
        personas = self.config.personas or ["Principal Algorithm Engineer"]
        strategies = self.config.strategies_pool or [
            "Design an optimal exact or heuristic algorithm tailored to the input constraints."
        ]
        constraints_pool = self.config.constraints_pool or [
            "Use strictly Python 3.11 standard library modules."
        ]

        # Generate diverse temperatures in [0.2, 0.9]
        temp_grid = [round(0.2 + (0.7 * i / max(1, pop_size - 1)), 2) for i in range(pop_size)]

        # Create Cartesian product combinations of persona x strategy
        pairs = list(itertools.product(personas, strategies))
        self._rng.shuffle(pairs)

        chromosomes: list[Chromosome] = []
        for idx in range(pop_size):
            persona, strategy = pairs[idx % len(pairs)]
            k = min(len(constraints_pool), self._rng.randint(2, min(4, max(2, len(constraints_pool)))))
            chosen_constraints = tuple(self._rng.sample(constraints_pool, k=k))
            temp = temp_grid[idx % len(temp_grid)]
            chromosomes.append(
                Chromosome(
                    id=str(uuid4()),
                    generation=0,
                    parent_ids=(),
                    system_persona=persona,
                    algorithmic_strategy=strategy,
                    constraints=chosen_constraints,
                    temperature=temp,
                    mutation_history=("init:combinatorial_matrix",),
                )
            )
        return chromosomes


class MetaPromptDivergentInitializer:
    """LLM-driven initializer generating pop_size orthogonal algorithmic strategies via JSON meta-prompt."""

    def __init__(
        self,
        config: InitializerConfig,
        llm_client: LLMClientPort,
        fallback_initializer: CombinatorialMatrixInitializer | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.fallback = fallback_initializer or CombinatorialMatrixInitializer(config)

    async def initialize(self, task: TaskSpec, pop_size: int) -> list[Chromosome]:
        if pop_size <= 0:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Principal Algorithm Research Director. "
                    "Given an algorithmic problem specification, generate a JSON object with key "
                    "`strategies` containing a list of distinct, orthogonal algorithmic approaches "
                    "(e.g., bitmask dynamic programming, branch-and-bound with admissible heuristics, "
                    "greedy construction + local search 2-opt, meet-in-the-middle, constraint propagation). "
                    "Each entry must contain: `system_persona` (str), `algorithmic_strategy` (str), "
                    "`constraints` (list of str), and `temperature` (float between 0.2 and 0.95)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task ID: {task.task_id}\n"
                    f"Task Description:\n{task.description}\n\n"
                    f"Generate {pop_size} orthogonal algorithmic strategies as a valid JSON object "
                    'with schema: {"strategies": [{"system_persona": "...", "algorithmic_strategy": "...", '
                    '"constraints": ["..."], "temperature": 0.7}]}.'
                ),
            },
        ]

        chromosomes: list[Chromosome] = []
        try:
            raw_text, _ = await self.llm_client.complete(
                messages=messages,
                temperature=self.config.default_temperature,
                json_mode=True,
            )
            payload = extract_json_payload(raw_text)
            items: list[dict[str, Any]]
            if isinstance(payload, list):
                items = [x for x in payload if isinstance(x, dict)]
            elif isinstance(payload, dict):
                raw_list = payload.get("strategies") or payload.get("approaches") or payload.get("items") or []
                items = [x for x in raw_list if isinstance(x, dict)]
            else:
                items = []

            for item in items[:pop_size]:
                persona = str(item.get("system_persona") or "Principal Algorithm Specialist").strip()
                strategy = str(
                    item.get("algorithmic_strategy")
                    or "Apply exact dynamic programming with aggressive state pruning."
                ).strip()
                raw_constraints = item.get("constraints") or []
                if isinstance(raw_constraints, list):
                    constraints = tuple(str(c).strip() for c in raw_constraints if str(c).strip())
                else:
                    constraints = (str(raw_constraints).strip(),)
                try:
                    temp = float(item.get("temperature", self.config.default_temperature))
                except (TypeError, ValueError):
                    temp = self.config.default_temperature
                temp = max(0.1, min(1.2, round(temp, 2)))

                chromosomes.append(
                    Chromosome(
                        id=str(uuid4()),
                        generation=0,
                        parent_ids=(),
                        system_persona=persona,
                        algorithmic_strategy=strategy,
                        constraints=constraints or tuple(self.config.constraints_pool[:2]),
                        temperature=temp,
                        mutation_history=("init:meta_prompt_divergent",),
                    )
                )
        except Exception:
            # Fall back to combinatorial matrix for any missing chromosomes
            pass

        if len(chromosomes) < pop_size:
            remainder = pop_size - len(chromosomes)
            fallback_chromos = await self.fallback.initialize(task, remainder)
            for idx, ch in enumerate(fallback_chromos):
                # If we already had some LLM strategies, diversify by cycling or mutating
                chromosomes.append(
                    Chromosome(
                        id=ch.id,
                        generation=0,
                        parent_ids=(),
                        system_persona=ch.system_persona,
                        algorithmic_strategy=ch.algorithmic_strategy,
                        constraints=ch.constraints,
                        temperature=ch.temperature,
                        mutation_history=("init:meta_prompt_fallback" if idx >= 0 else "init:combinatorial_matrix",),
                    )
                )
        return chromosomes[:pop_size]


class HybridPopulationInitializer:
    """Combines MetaPromptDivergentInitializer and CombinatorialMatrixInitializer by configurable ratio."""

    def __init__(
        self,
        config: InitializerConfig,
        llm_client: LLMClientPort,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.matrix_init = CombinatorialMatrixInitializer(config, seed=seed)
        self.meta_init = MetaPromptDivergentInitializer(
            config,
            llm_client=llm_client,
            fallback_initializer=self.matrix_init,
        )

    async def initialize(self, task: TaskSpec, pop_size: int) -> list[Chromosome]:
        if self.config.type == "combinatorial_matrix":
            return await self.matrix_init.initialize(task, pop_size)
        if self.config.type == "meta_prompt_divergent":
            return await self.meta_init.initialize(task, pop_size)

        meta_count = int(round(pop_size * self.config.meta_prompt_ratio))
        matrix_count = pop_size - meta_count
        meta_chromosomes = await self.meta_init.initialize(task, meta_count) if meta_count > 0 else []
        matrix_chromosomes = await self.matrix_init.initialize(task, matrix_count) if matrix_count > 0 else []
        return meta_chromosomes + matrix_chromosomes


def build_initializer(
    config: InitializerConfig,
    llm_client: LLMClientPort,
    seed: int | None = None,
) -> PopulationInitializerPort:
    """Factory function returning the configured population initializer strategy."""
    if config.type == "combinatorial_matrix":
        return CombinatorialMatrixInitializer(config, seed=seed)
    if config.type == "meta_prompt_divergent":
        return MetaPromptDivergentInitializer(config, llm_client=llm_client)
    return HybridPopulationInitializer(config, llm_client=llm_client, seed=seed)
