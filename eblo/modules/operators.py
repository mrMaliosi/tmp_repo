"""Module 3: Genetic Crossover and Mutation Operators for EBLO."""

from __future__ import annotations

import random
from uuid import uuid4

from eblo.core.domain import Chromosome, Individual, TaskSpec
from eblo.core.ports import CrossoverOperatorPort, LLMClientPort, MutationOperatorPort
from eblo.llm.parser import extract_json_payload


class StructuralBlockCrossover:
    """Zero-LLM crossover (0 ms): combines system_persona & temperature from Parent A
    with algorithmic_strategy & constraints from Parent B.
    """

    name: str = "structural_block"

    async def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        ca = parent_a.chromosome
        cb = parent_b.chromosome
        merged_history = tuple(ca.mutation_history[-2:]) + ("crossover:structural_block",)
        return Chromosome(
            id=str(uuid4()),
            generation=next_generation,
            parent_ids=(ca.id, cb.id),
            system_persona=ca.system_persona,
            algorithmic_strategy=cb.algorithmic_strategy,
            constraints=cb.constraints,
            temperature=ca.temperature,
            mutation_history=merged_history,
        )


class SemanticSynthesisCrossover:
    """LLM-driven crossover: synthesizes an improved algorithmic_strategy from two parents'
    prompts, generated code, execution times, and error traces.
    """

    name: str = "semantic_synthesis"

    def __init__(self, llm_client: LLMClientPort) -> None:
        self.llm_client = llm_client

    async def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        ca, pa, fa = parent_a.chromosome, parent_a.phenotype, parent_a.fitness
        cb, pb, fb = parent_b.chromosome, parent_b.phenotype, parent_b.fitness

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Evolutionary Genetic Programming Synthesis Engine. "
                    "You receive two parent solutions (Genotype + Code + Fitness Report). "
                    "Synthesize a superior child genotype that combines the algorithmic strengths "
                    "of both parents while eliminating their bottlenecks or bugs. "
                    "Respond with a JSON object containing keys: "
                    "`system_persona` (str), `algorithmic_strategy` (str), and `constraints` (list of str)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"### Task: {task.task_id}\n{task.description}\n\n"
                    f"### Parent A (Score: {fa.total_score:.4f}, Pass Rate: {fa.pass_rate:.2f}, "
                    f"Time: {fa.execution_time_ms:.2f} ms)\n"
                    f"- Persona: {ca.system_persona}\n"
                    f"- Strategy: {ca.algorithmic_strategy}\n"
                    f"- Constraints: {list(ca.constraints)}\n"
                    f"- Error Trace: {fa.error_trace or 'None'}\n"
                    f"- Code Snippet:\n```python\n{pa.code[:1500]}\n```\n\n"
                    f"### Parent B (Score: {fb.total_score:.4f}, Pass Rate: {fb.pass_rate:.2f}, "
                    f"Time: {fb.execution_time_ms:.2f} ms)\n"
                    f"- Persona: {cb.system_persona}\n"
                    f"- Strategy: {cb.algorithmic_strategy}\n"
                    f"- Constraints: {list(cb.constraints)}\n"
                    f"- Error Trace: {fb.error_trace or 'None'}\n"
                    f"- Code Snippet:\n```python\n{pb.code[:1500]}\n```\n\n"
                    "Return JSON with `system_persona`, `algorithmic_strategy`, and `constraints`."
                ),
            },
        ]

        avg_temp = round((ca.temperature + cb.temperature) / 2.0, 2)
        try:
            raw_text, _ = await self.llm_client.complete(
                messages=messages,
                temperature=0.65,
                json_mode=True,
            )
            data = extract_json_payload(raw_text)
            if isinstance(data, dict):
                persona = str(data.get("system_persona") or ca.system_persona).strip()
                strategy = str(data.get("algorithmic_strategy") or cb.algorithmic_strategy).strip()
                raw_constraints = data.get("constraints")
                if isinstance(raw_constraints, list) and raw_constraints:
                    constraints = tuple(str(c).strip() for c in raw_constraints if str(c).strip())
                else:
                    constraints = tuple(dict.fromkeys(ca.constraints + cb.constraints))[:5]
                return Chromosome(
                    id=str(uuid4()),
                    generation=next_generation,
                    parent_ids=(ca.id, cb.id),
                    system_persona=persona,
                    algorithmic_strategy=strategy,
                    constraints=constraints,
                    temperature=avg_temp,
                    mutation_history=tuple(ca.mutation_history[-2:]) + ("crossover:semantic_synthesis",),
                )
        except Exception:
            pass

        # Deterministic synthesis fallback if LLM JSON parsing fails
        better = parent_a if fa.total_score >= fb.total_score else parent_b
        other = parent_b if better is parent_a else parent_a
        merged_constraints = tuple(dict.fromkeys(better.chromosome.constraints + other.chromosome.constraints))[:5]
        combined_strategy = (
            f"{better.chromosome.algorithmic_strategy} "
            f"Integrate complementary optimization from secondary parent: {other.chromosome.algorithmic_strategy}"
        )
        return Chromosome(
            id=str(uuid4()),
            generation=next_generation,
            parent_ids=(ca.id, cb.id),
            system_persona=better.chromosome.system_persona,
            algorithmic_strategy=combined_strategy,
            constraints=merged_constraints,
            temperature=avg_temp,
            mutation_history=tuple(better.chromosome.mutation_history[-2:]) + ("crossover:semantic_synthesis",),
        )


class ReflexionFeedbackMutator:
    """Error- and profiler-guided reflexive mutation operator.
    - If pass_rate < 1.0: sends prompt, generated code, and validator error_trace to fix the bug/timeout
      and appends a defensive constraint.
    - If pass_rate == 1.0: mutates strategy strictly to reduce execution_time_ms and memory overhead.
    """

    name: str = "reflexion_feedback"

    def __init__(self, llm_client: LLMClientPort) -> None:
        self.llm_client = llm_client

    async def mutate(
        self,
        parent: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        ch = parent.chromosome
        ph = parent.phenotype
        fit = parent.fitness

        if fit.pass_rate < 1.0:
            directive = (
                "The candidate solution FAILED validation tests or timed out.\n"
                f"Pass Rate: {fit.pass_rate:.2f}\n"
                f"Validator Error Trace:\n{fit.error_trace or 'Failed assertions on edge cases.'}\n\n"
                "Analyze the root cause of the failure in the code and strategy. "
                "Rewrite `algorithmic_strategy` to fix the flaw and add a specific rule to `constraints` "
                "that prevents this exact bug or TimeoutError."
            )
            tag = "mutate:reflexion_bugfix"
        else:
            directive = (
                "The candidate solution PASSED 100% of benchmark tests!\n"
                f"Current Execution Time: {fit.execution_time_ms:.3f} ms | "
                f"Peak Memory: {fit.memory_peak_kb:.2f} KB.\n\n"
                "Rewrite `algorithmic_strategy` strictly to accelerate runtime performance "
                "(e.g., tighter pruning bounds, bitmask operations, precomputed lookup tables, "
                "eliminating redundant allocations) while preserving 100% correctness."
            )
            tag = "mutate:reflexion_speedup"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Reflexive Program Repair and Performance Optimization Architect. "
                    "Respond with a valid JSON object containing: `system_persona` (str), "
                    "`algorithmic_strategy` (str), and `constraints` (list of str)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"### Task: {task.task_id}\n{task.description}\n\n"
                    f"### Current Genotype\n"
                    f"- Persona: {ch.system_persona}\n"
                    f"- Strategy: {ch.algorithmic_strategy}\n"
                    f"- Constraints: {list(ch.constraints)}\n\n"
                    f"### Generated Code\n```python\n{ph.code[:1800]}\n```\n\n"
                    f"### Reflexion Directive\n{directive}\n\n"
                    "Return JSON with updated `system_persona`, `algorithmic_strategy`, and `constraints`."
                ),
            },
        ]

        try:
            raw_text, _ = await self.llm_client.complete(
                messages=messages,
                temperature=0.6,
                json_mode=True,
            )
            data = extract_json_payload(raw_text)
            if isinstance(data, dict):
                persona = str(data.get("system_persona") or ch.system_persona).strip()
                strategy = str(data.get("algorithmic_strategy") or ch.algorithmic_strategy).strip()
                raw_constraints = data.get("constraints")
                if isinstance(raw_constraints, list) and raw_constraints:
                    constraints = tuple(str(c).strip() for c in raw_constraints if str(c).strip())
                else:
                    extra_rule = (
                        f"Guard against error: {(fit.error_trace or 'edge-case failure').splitlines()[0][:120]}"
                        if fit.pass_rate < 1.0
                        else "Optimize inner loop complexity and prune suboptimal states early."
                    )
                    constraints = ch.constraints + (extra_rule,)
                return Chromosome(
                    id=str(uuid4()),
                    generation=next_generation,
                    parent_ids=(ch.id,),
                    system_persona=persona,
                    algorithmic_strategy=strategy,
                    constraints=constraints[:6],
                    temperature=ch.temperature,
                    mutation_history=ch.mutation_history + (tag,),
                )
        except Exception:
            pass

        extra_rule = (
            f"Eliminate failure mode: {(fit.error_trace or 'assertion mismatch').splitlines()[0][:120]}"
            if fit.pass_rate < 1.0
            else "Minimize execution_time_ms via memoized state transitions and local variable binding."
        )
        updated_strategy = (
            f"{ch.algorithmic_strategy} [Reflexion Refinement: {extra_rule}]"
        )
        return Chromosome(
            id=str(uuid4()),
            generation=next_generation,
            parent_ids=(ch.id,),
            system_persona=ch.system_persona,
            algorithmic_strategy=updated_strategy,
            constraints=(ch.constraints + (extra_rule,))[:6],
            temperature=ch.temperature,
            mutation_history=ch.mutation_history + (tag,),
        )


class OrthogonalShiftMutator:
    """Exploratory LLM mutator: forbids the current approach in algorithmic_strategy
    and proposes a fundamentally different mathematical or algorithmic paradigm.
    """

    name: str = "orthogonal_shift"

    def __init__(self, llm_client: LLMClientPort) -> None:
        self.llm_client = llm_client

    async def mutate(
        self,
        parent: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        ch = parent.chromosome
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Radical Algorithmic Paradigm Shift Operator. "
                    "You must strictly FORBID the parent's current algorithmic strategy and invent "
                    "a mathematically or algorithmically orthogonal method to solve the task. "
                    "Return JSON with keys: `system_persona` (str), `algorithmic_strategy` (str), "
                    "and `constraints` (list of str)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"### Task: {task.task_id}\n{task.description}\n\n"
                    f"### Forbidden Strategy (DO NOT USE THIS PARADIGM)\n"
                    f"{ch.algorithmic_strategy}\n\n"
                    "Propose a completely different algorithmic paradigm in JSON format."
                ),
            },
        ]

        try:
            raw_text, _ = await self.llm_client.complete(
                messages=messages,
                temperature=0.9,
                json_mode=True,
            )
            data = extract_json_payload(raw_text)
            if isinstance(data, dict):
                persona = str(data.get("system_persona") or ch.system_persona).strip()
                strategy = str(data.get("algorithmic_strategy") or "").strip()
                raw_constraints = data.get("constraints")
                if strategy:
                    constraints = (
                        tuple(str(c).strip() for c in raw_constraints if str(c).strip())
                        if isinstance(raw_constraints, list) and raw_constraints
                        else ch.constraints
                    )
                    return Chromosome(
                        id=str(uuid4()),
                        generation=next_generation,
                        parent_ids=(ch.id,),
                        system_persona=persona,
                        algorithmic_strategy=strategy,
                        constraints=constraints[:6],
                        temperature=min(1.2, round(ch.temperature + 0.05, 2)),
                        mutation_history=ch.mutation_history + ("mutate:orthogonal_shift",),
                    )
        except Exception:
            pass

        orthogonal_paradigms = [
            "Switch paradigm to iterative Branch-and-Bound with priority-queue lower-bound relaxation.",
            "Switch paradigm to exact Bitmask Dynamic Programming over compressed state frontiers.",
            "Switch paradigm to Constraint Propagation combined with backtracking and earliest-deadline ordering.",
            "Switch paradigm to Meet-in-the-Middle bidirectional state search with hash-table joining.",
        ]
        chosen = next(
            (p for p in orthogonal_paradigms if p not in ch.algorithmic_strategy),
            orthogonal_paradigms[0],
        )
        return Chromosome(
            id=str(uuid4()),
            generation=next_generation,
            parent_ids=(ch.id,),
            system_persona="Orthogonal Algorithm Research Specialist",
            algorithmic_strategy=chosen,
            constraints=ch.constraints,
            temperature=min(1.2, round(ch.temperature + 0.05, 2)),
            mutation_history=ch.mutation_history + ("mutate:orthogonal_shift",),
        )


class HyperparameterJitterMutator:
    """Zero-LLM mutator: shifts temperature by +/- 0.15 within [0.1, 1.2]
    and randomly adds or removes one constraint from the constraint pool.
    """

    name: str = "hyperparameter_jitter"

    def __init__(self, constraints_pool: list[str] | None = None, seed: int | None = None) -> None:
        self.constraints_pool = constraints_pool or [
            "Avoid recursion depth traps; prefer iterative state stacks or explicit queues.",
            "Use strictly standard library modules (math, heapq, collections, itertools, bisect, array).",
            "Optimize inner loops by hoisting invariant calculations and binding local variables.",
            "Guarantee worst-case termination within the time limit using early-exit pruning.",
            "Minimize memory allocations inside tight loops; reuse preallocated buffers or tuples.",
            "Use bit manipulation for state representation whenever the state space dimension allows.",
        ]
        self._rng = random.Random(seed)

    async def mutate(
        self,
        parent: Individual,
        task: TaskSpec,
        next_generation: int,
    ) -> Chromosome:
        ch = parent.chromosome
        delta = self._rng.choice([-0.15, 0.15])
        new_temp = round(max(0.1, min(1.2, ch.temperature + delta)), 2)

        current_list = list(ch.constraints)
        can_remove = len(current_list) > 1
        available_to_add = [c for c in self.constraints_pool if c not in current_list]

        if can_remove and (not available_to_add or self._rng.random() < 0.5):
            drop_idx = self._rng.randrange(len(current_list))
            current_list.pop(drop_idx)
        elif available_to_add:
            current_list.append(self._rng.choice(available_to_add))

        return Chromosome(
            id=str(uuid4()),
            generation=next_generation,
            parent_ids=(ch.id,),
            system_persona=ch.system_persona,
            algorithmic_strategy=ch.algorithmic_strategy,
            constraints=tuple(current_list),
            temperature=new_temp,
            mutation_history=ch.mutation_history + ("mutate:hyperparameter_jitter",),
        )
