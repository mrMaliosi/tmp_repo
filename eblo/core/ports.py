"""Abstract interfaces (Protocols and ABCs) for EBLO modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eblo.core.domain import Chromosome, FitnessReport, Individual, Phenotype, TaskSpec


@runtime_checkable
class LLMClientPort(Protocol):
    """Port for asynchronous LLM chat completion."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> tuple[str, float]:
        """Execute chat completion and return (raw_text, latency_ms)."""


@runtime_checkable
class PopulationInitializerPort(Protocol):
    """Module 1 interface: initializes initial population P_0 of Chromosomes."""

    async def initialize(self, task: TaskSpec, pop_size: int) -> list[Chromosome]:
        """Generate pop_size initial chromosomes for the given task."""


@runtime_checkable
class PhenotypeValidatorPort(Protocol):
    """Module 2 interface: evaluates a Phenotype via cascade validation and benchmarking."""

    async def evaluate(self, phenotype: Phenotype, task: TaskSpec) -> FitnessReport:
        """Evaluate phenotype and return FitnessReport."""


@runtime_checkable
class CrossoverOperatorPort(Protocol):
    """Module 3a interface: recombines two parent Individuals into a child Chromosome."""

    name: str

    async def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        """Produce a child chromosome from two parent individuals."""


@runtime_checkable
class MutationOperatorPort(Protocol):
    """Module 3b interface: mutates a parent Individual into a new Chromosome."""

    name: str

    async def mutate(
        self,
        parent: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        """Produce a mutated child chromosome from a parent individual."""


@runtime_checkable
class SelectionOperatorPort(Protocol):
    """Selection interface for choosing parents and survivors."""

    def select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        """Select parent individuals for breeding."""

    def select_elites(self, population: list[Individual], elite_count: int) -> list[Individual]:
        """Select top elite individuals to preserve unchanged."""
