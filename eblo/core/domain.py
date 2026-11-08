"""Immutable core domain models for EBLO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TestCase:
    """Single unit or benchmark test case for candidate evaluation."""

    name: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    validator_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "expected": self.expected,
            "validator_code": self.validator_code,
        }


@dataclass(frozen=True)
class TaskSpec:
    """Algorithmic or engineering problem specification."""

    task_id: str
    description: str
    smoke_tests: list[TestCase]
    benchmark_tests: list[TestCase]
    entrypoint: str = "solve"
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "language": self.language,
            "smoke_tests": [t.to_dict() for t in self.smoke_tests],
            "benchmark_tests": [t.to_dict() for t in self.benchmark_tests],
        }


@dataclass(frozen=True)
class Chromosome:
    """Genotype representing a structured LLM prompt and sampling configuration."""

    id: str = field(default_factory=lambda: str(uuid4()))
    generation: int = 0
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    system_persona: str = "Principal Systems Architect and Algorithm Specialist."
    algorithmic_strategy: str = "Design an optimal, low-overhead exact or heuristic algorithm."
    constraints: tuple[str, ...] = field(default_factory=tuple)
    temperature: float = 0.7
    mutation_history: tuple[str, ...] = field(default_factory=tuple)

    def render_prompt(self, task: TaskSpec) -> list[dict[str, str]]:
        """Assemble structured OpenAI-compatible chat messages for phenotype code generation."""
        constraints_block = (
            "\n".join(f"- {c}" for c in self.constraints)
            if self.constraints
            else "- Use clean, self-contained standard library Python 3.11+ code."
        )
        system_content = (
            f"You are operating as: {self.system_persona}\n"
            "Your objective is to produce high-performance, bug-free executable code that solves "
            "the specified algorithmic problem. Think carefully inside <think>...</think> tags if needed, "
            f"and then output the complete {task.language} implementation inside a single "
            f"```{task.language} ... ``` markdown block."
        )
        user_content = (
            f"### Problem Specification (`{task.task_id}`)\n"
            f"{task.description.strip()}\n\n"
            f"### Required Entrypoint\n"
            f"Ensure the function `{task.entrypoint}` is defined at module top-level with the exact signature "
            "described above.\n\n"
            f"### Algorithmic Strategy to Implement\n"
            f"{self.algorithmic_strategy.strip()}\n\n"
            f"### Strict Engineering Constraints\n"
            f"{constraints_block}\n\n"
            "Provide only the reasoning block (optional) and the complete, self-contained code block."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phenotype:
    """Executable code artifact generated from a Chromosome."""

    chromosome_id: str
    code: str
    ast_hash: str
    reasoning_trace: str | None = None
    generation_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitnessReport:
    """Cascade evaluation report produced by the validator."""

    is_syntax_valid: bool
    smoke_passed: bool
    pass_rate: float
    execution_time_ms: float
    memory_peak_kb: float
    total_score: float
    error_trace: str | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Individual:
    """Evaluated evolutionary individual pairing Genotype, Phenotype, and FitnessReport."""

    chromosome: Chromosome
    phenotype: Phenotype
    fitness: FitnessReport
    island_id: int = 0

    @property
    def id(self) -> str:
        return self.chromosome.id

    @property
    def total_score(self) -> float:
        return self.fitness.total_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": self.island_id,
            "chromosome": self.chromosome.to_dict(),
            "phenotype": self.phenotype.to_dict(),
            "fitness": self.fitness.to_dict(),
        }
