"""Configuration schema and YAML loader for EBLO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, validator


class ExperimentConfig(BaseModel):
    name: str = Field(default="eblo-default-run")
    evolution_mode: Literal["generational", "steady_state", "island"] = Field(default="island")
    generations: int = Field(default=15, ge=1)
    population_size: int = Field(default=24, ge=2)
    islands: int = Field(default=3, ge=1)
    migration_interval: int = Field(default=5, ge=1)
    runs_dir: str = Field(default="runs")


class LLMConfig(BaseModel):
    provider: Literal["openai_compatible", "mock"] = Field(default="openai_compatible")
    base_url: str = Field(default="http://127.0.0.1:8080/v1")
    api_key: str = Field(default="strata-local")
    model: str = Field(default="Qwen3.8-Flash-Next-Coder")
    max_concurrent: int = Field(default=4, ge=1)
    timeout_sec: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    stream: bool = Field(default=False)


class InitializerConfig(BaseModel):
    type: Literal["meta_prompt_divergent", "combinatorial_matrix", "hybrid"] = Field(default="hybrid")
    meta_prompt_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    default_temperature: float = Field(default=0.85, ge=0.1, le=1.5)
    personas: list[str] = Field(
        default_factory=lambda: [
            "Principal Algorithm Engineer focused on asymptotic time complexity and CPU cache locality.",
            "Competitive Programming Grandmaster specializing in dynamic programming, bitmasking, and branch-and-bound pruning.",
            "High-Performance Python Systems Architect utilizing precomputation and low-overhead data structures.",
            "Mathematical Optimization Specialist applying graph theory and exact search heuristics.",
        ]
    )
    constraints_pool: list[str] = Field(
        default_factory=lambda: [
            "Avoid recursion depth traps; prefer iterative state stacks or explicit queues.",
            "Use strictly standard library modules (math, heapq, collections, itertools, bisect, array).",
            "Optimize inner loops by hoisting invariant calculations and binding local variables.",
            "Guarantee worst-case termination within the time limit using early-exit pruning.",
            "Minimize memory allocations inside tight loops; reuse preallocated buffers or tuples.",
            "Use bit manipulation for state representation whenever the state space dimension allows.",
        ]
    )
    strategies_pool: list[str] = Field(
        default_factory=lambda: [
            "Apply exact dynamic programming with memoization and state-space pruning to eliminate dominated states.",
            "Implement a branch-and-bound search ordered by an admissible lower-bound heuristic with aggressive pruning.",
            "Construct an initial greedy solution followed by fast local search and exact verification.",
            "Use bidirectional search or meet-in-the-middle decomposition to reduce exponential complexity.",
        ]
    )


class ValidatorWeightsConfig(BaseModel):
    pass_rate: float = Field(default=0.75, ge=0.0)
    speed: float = Field(default=0.25, ge=0.0)


class ValidatorConfig(BaseModel):
    runner: Literal["local_subprocess", "in_process_fast"] = Field(default="local_subprocess")
    smoke_timeout_ms: int = Field(default=500, ge=50)
    benchmark_timeout_ms: int = Field(default=2500, ge=100)
    reference_time_ms: float = Field(default=100.0, gt=0.0)
    cache_backend: Literal["sqlite", "memory"] = Field(default="sqlite")
    cache_db_path: str = Field(default="runs/ast_cache.sqlite")
    weights: ValidatorWeightsConfig = Field(default_factory=ValidatorWeightsConfig)


class SelectionConfig(BaseModel):
    type: Literal["tournament", "nsga2"] = Field(default="tournament")
    tournament_size: int = Field(default=3, ge=2)
    elitism_count: int = Field(default=2, ge=0)


class WeightedCrossoverSpec(BaseModel):
    type: Literal["semantic_synthesis", "structural_block"]
    weight: float = Field(default=0.5, gt=0.0)


class WeightedMutationSpec(BaseModel):
    type: Literal["reflexion_feedback", "orthogonal_shift", "hyperparameter_jitter"]
    weight: float = Field(default=1.0 / 3.0, gt=0.0)


class VariationConfig(BaseModel):
    crossover_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    crossovers: list[WeightedCrossoverSpec] = Field(
        default_factory=lambda: [
            WeightedCrossoverSpec(type="semantic_synthesis", weight=0.7),
            WeightedCrossoverSpec(type="structural_block", weight=0.3),
        ]
    )
    mutations: list[WeightedMutationSpec] = Field(
        default_factory=lambda: [
            WeightedMutationSpec(type="reflexion_feedback", weight=0.5),
            WeightedMutationSpec(type="orthogonal_shift", weight=0.3),
            WeightedMutationSpec(type="hyperparameter_jitter", weight=0.2),
        ]
    )

    @validator("crossovers")
    def validate_non_empty_crossovers(cls, v: list[WeightedCrossoverSpec]) -> list[WeightedCrossoverSpec]:
        if not v:
            raise ValueError("At least one crossover operator must be configured.")
        return v

    @validator("mutations")
    def validate_non_empty_mutations(cls, v: list[WeightedMutationSpec]) -> list[WeightedMutationSpec]:
        if not v:
            raise ValueError("At least one mutation operator must be configured.")
        return v


class EBLOConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    initializer: InitializerConfig = Field(default_factory=InitializerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    variation: VariationConfig = Field(default_factory=VariationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EBLOConfig:
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        raw_data: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if hasattr(cls, "model_validate"):
            return cls.model_validate(raw_data)  # type: ignore[attr-defined]
        return cls.parse_obj(raw_data)

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()  # type: ignore[attr-defined]
        return self.dict()
