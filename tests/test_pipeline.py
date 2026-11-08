"""End-to-end unit and integration test suite for EBLO (Evolutionary Blazing-fast LLM Orchestrator).

Supports both standard `unittest` execution (`python3 -m unittest tests/test_pipeline.py`)
and `pytest` (if installed), plus `--live` mode for testing against a live Strata LLM server
at `http://127.0.0.1:8080/v1`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from eblo.config.schema import EBLOConfig
from eblo.core.domain import Chromosome, FitnessReport, Individual, Phenotype
from eblo.core.engine import EvolutionEngine
from eblo.examples.sample_task.solutions import (
    CANDIDATE_BITMASK_DP_FAST,
    CANDIDATE_DFS_SLOW,
    CANDIDATE_GREEDY_BUGGY,
)
from eblo.examples.sample_task.task import get_task_spec
from eblo.llm.client import AsyncLLMClient
from eblo.llm.parser import (
    compute_normalized_ast_hash,
    extract_json_payload,
    extract_thinking_and_code,
    parse_phenotype,
)
from eblo.modules.initializers import (
    CombinatorialMatrixInitializer,
    HybridPopulationInitializer,
    MetaPromptDivergentInitializer,
)
from eblo.modules.operators import (
    HyperparameterJitterMutator,
    OrthogonalShiftMutator,
    ReflexionFeedbackMutator,
    SemanticSynthesisCrossover,
    StructuralBlockCrossover,
)
from eblo.modules.selectors import NSGA2Selector, TournamentSelector
from eblo.modules.validators import CascadeValidator
from eblo.storage.lineage import ASTFitnessCache

LIVE_MODE: bool = False


class TestEBLOPipeline(unittest.IsolatedAsyncioTestCase):
    """Comprehensive verification of EBLO components and full evolutionary loops."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="eblo_test_")
        self.tmp_path = Path(self.tmp_dir.name)
        self.task = get_task_spec()

        default_yaml = Path(__file__).resolve().parent.parent / "eblo" / "config" / "default_config.yaml"
        self.config = EBLOConfig.from_yaml(default_yaml)
        self.config.experiment.runs_dir = str(self.tmp_path / "runs")
        self.config.validator.cache_db_path = str(self.tmp_path / "ast_cache.sqlite")
        if not LIVE_MODE:
            self.config.llm.provider = "mock"

    def tearDown(self) -> None:
        import gc; gc.collect()
        try:
            self.tmp_dir.cleanup()
        except PermissionError:
            pass

    def test_01_parser_thinking_and_ast_hash_normalization(self) -> None:
        """Verify <think> extraction and AST normalization stripping comments & docstrings."""
        raw_a = (
            "<think>\nStep 1: Analyze complexity.\nStep 2: Use bitmask DP.\n</think>\n"
            "```python\n"
            "def solve(x: int) -> int:\n"
            '    """Docstring that should be stripped by AST normalizer."""\n'
            "    # Comment that should be ignored\n"
            "    return x + 1\n"
            "```"
        )
        raw_b = (
            "```python\n"
            "def solve(x: int) -> int:\n"
            "    return x + 1\n"
            "```"
        )
        ph_a = parse_phenotype("chr-1", raw_a, 12.5)
        ph_b = parse_phenotype("chr-2", raw_b, 8.1)

        self.assertIsNotNone(ph_a.reasoning_trace)
        self.assertIn("Step 1: Analyze complexity.", ph_a.reasoning_trace or "")
        self.assertIsNone(ph_b.reasoning_trace)
        self.assertEqual(ph_a.ast_hash, ph_b.ast_hash)

    async def test_02_cascade_validator_stages_and_ast_cache(self) -> None:
        """Verify Stage 0 (AST cache), Stage 1 (Syntax gate), Stage 2 (Smoke gate), and Stage 3 (Full benchmark)."""
        cache = ASTFitnessCache(backend="sqlite", db_path=self.tmp_path / "cache.sqlite")
        validator = CascadeValidator(self.config.validator, cache=cache)

        # Stage 1: Syntax Error Gate
        bad_syntax = parse_phenotype("c-syn", "```python\ndef solve(:\n    pass\n```", 1.0)
        rep_syn = await validator.evaluate(bad_syntax, self.task)
        self.assertFalse(rep_syn.is_syntax_valid)
        self.assertFalse(rep_syn.smoke_passed)
        self.assertEqual(rep_syn.total_score, 0.0)
        self.assertIn("SyntaxError", rep_syn.error_trace or "")

        # Stage 2: Smoke Gate failure (wrong return value on smoke tests)
        wrong_smoke = parse_phenotype(
            "c-smoke",
            "```python\ndef solve(d, tw, s):\n    return -999\n```",
            1.0,
        )
        rep_smoke = await validator.evaluate(wrong_smoke, self.task)
        self.assertTrue(rep_smoke.is_syntax_valid)
        self.assertFalse(rep_smoke.smoke_passed)
        self.assertIn("AssertionError", rep_smoke.error_trace or "")

        # Stage 3: Full Benchmark pass with fast Bitmask DP
        fast_pheno = parse_phenotype(
            "c-fast",
            f"<think>Optimal DP</think>\n```python\n{CANDIDATE_BITMASK_DP_FAST}\n```",
            5.0,
        )
        rep_fast = await validator.evaluate(fast_pheno, self.task)
        self.assertTrue(rep_fast.is_syntax_valid)
        self.assertTrue(rep_fast.smoke_passed)
        self.assertEqual(rep_fast.pass_rate, 1.0)
        self.assertGreater(rep_fast.total_score, 0.90)
        self.assertFalse(rep_fast.cache_hit)

        # Stage 0: AST Lookup Cache Hit when re-evaluating equivalent code with extra comments/docstrings
        commented_code = "# Header comment\n" + CANDIDATE_BITMASK_DP_FAST + "\n# Footer comment\n"
        dup_pheno = parse_phenotype("c-dup", f"```python\n{commented_code}\n```", 2.0)
        self.assertEqual(fast_pheno.ast_hash, dup_pheno.ast_hash)

        rep_cached = await validator.evaluate(dup_pheno, self.task)
        self.assertTrue(rep_cached.cache_hit)
        self.assertEqual(rep_cached.total_score, rep_fast.total_score)
        self.assertGreaterEqual(cache.hits, 1)

    async def test_03_initializers_and_all_five_genetic_operators(self) -> None:
        """Verify both initializers, both crossovers, all three mutators, and both selectors."""
        llm_client = AsyncLLMClient(self.config.llm)
        try:
            # 1. Initializers
            comb_init = CombinatorialMatrixInitializer(self.config.initializer, seed=7)
            c_pop = await comb_init.initialize(self.task, 4)
            self.assertEqual(len(c_pop), 4)

            meta_init = MetaPromptDivergentInitializer(self.config.initializer, llm_client=llm_client)
            m_pop = await meta_init.initialize(self.task, 4)
            self.assertEqual(len(m_pop), 4)

            hybrid_init = HybridPopulationInitializer(self.config.initializer, llm_client=llm_client, seed=7)
            h_pop = await hybrid_init.initialize(self.task, 6)
            self.assertEqual(len(h_pop), 6)

            # Build two sample parent individuals (one failing, one passing)
            parent_a = Individual(
                chromosome=c_pop[0],
                phenotype=parse_phenotype(c_pop[0].id, CANDIDATE_GREEDY_BUGGY, 3.0),
                fitness=FitnessReport(
                    is_syntax_valid=True,
                    smoke_passed=False,
                    pass_rate=0.25,
                    execution_time_ms=1.2,
                    memory_peak_kb=12.0,
                    total_score=0.20,
                    error_trace="[smoke_waiting_required] AssertionError: expected 65, got -1",
                ),
            )
            parent_b = Individual(
                chromosome=m_pop[0],
                phenotype=parse_phenotype(m_pop[0].id, CANDIDATE_DFS_SLOW, 4.0),
                fitness=FitnessReport(
                    is_syntax_valid=True,
                    smoke_passed=True,
                    pass_rate=1.0,
                    execution_time_ms=25.0,
                    memory_peak_kb=24.0,
                    total_score=0.95,
                    error_trace=None,
                ),
            )

            # 2. Crossovers
            struct_cx = StructuralBlockCrossover()
            child_cx1 = await struct_cx.crossover(parent_a, parent_b, self.task, next_generation=1)
            self.assertEqual(child_cx1.system_persona, parent_a.chromosome.system_persona)
            self.assertEqual(child_cx1.algorithmic_strategy, parent_b.chromosome.algorithmic_strategy)

            sem_cx = SemanticSynthesisCrossover(llm_client)
            child_cx2 = await sem_cx.crossover(parent_a, parent_b, self.task, next_generation=1)
            self.assertEqual(child_cx2.generation, 1)
            self.assertEqual(len(child_cx2.parent_ids), 2)

            # 3. Mutators
            reflex_mut = ReflexionFeedbackMutator(llm_client)
            child_m1_bug = await reflex_mut.mutate(parent_a, self.task, next_generation=1)
            self.assertIn("mutate:reflexion_bugfix", child_m1_bug.mutation_history)

            child_m1_speed = await reflex_mut.mutate(parent_b, self.task, next_generation=1)
            self.assertIn("mutate:reflexion_speedup", child_m1_speed.mutation_history)

            orth_mut = OrthogonalShiftMutator(llm_client)
            child_m2 = await orth_mut.mutate(parent_a, self.task, next_generation=1)
            self.assertIn("mutate:orthogonal_shift", child_m2.mutation_history)

            jitter_mut = HyperparameterJitterMutator(seed=11)
            child_m3 = await jitter_mut.mutate(parent_a, self.task, next_generation=1)
            self.assertIn("mutate:hyperparameter_jitter", child_m3.mutation_history)
            self.assertNotEqual(child_m3.temperature, parent_a.chromosome.temperature)

            # 4. Selectors (Tournament + NSGA-II)
            t_sel = TournamentSelector(self.config.selection, seed=5)
            self.assertEqual(t_sel.select_elites([parent_a, parent_b], 1)[0].id, parent_b.id)

            nsga_sel = NSGA2Selector(self.config.selection, seed=5)
            elites = nsga_sel.select_elites([parent_a, parent_b], 1)
            self.assertEqual(elites[0].id, parent_b.id)
            parents = nsga_sel.select_parents([parent_a, parent_b], 2)
            self.assertEqual(len(parents), 2)
        finally:
            await llm_client.aclose()

    async def test_04_full_evolution_loop_island_and_generational(self) -> None:
        """Run full multi-generation evolution from Generation 0 to Generation N in Island mode."""
        self.config.experiment.evolution_mode = "island"
        self.config.experiment.generations = 4
        self.config.experiment.population_size = 6
        self.config.experiment.islands = 2
        self.config.experiment.migration_interval = 2
        self.config.selection.elitism_count = 1

        engine = EvolutionEngine(self.config, seed=42)
        try:
            result = await engine.run(
                task=self.task,
                run_id="test_island_run",
                show_dashboard=True,
            )
            # Generation 0 + 4 generations = 5 history records
            self.assertEqual(len(result.history), 5)

            # Verify monotonic non-decreasing best score guaranteed by elitism
            scores = [m.best_score for m in result.history]
            for prev, curr in zip(scores, scores[1:]):
                self.assertGreaterEqual(curr, prev - 1e-9)

            # Verify final best individual passes 100% of TSPTW benchmark tests
            self.assertEqual(result.best_individual.fitness.pass_rate, 1.0)
            self.assertGreater(result.best_individual.fitness.total_score, 0.90)

            # Verify lineage.jsonl and best solution artifact were persisted
            lineage_file = Path(result.lineage_jsonl_path)
            self.assertTrue(lineage_file.exists())
            lines = lineage_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5 * 6)  # 5 generations * 6 individuals
            sample_record = json.loads(lines[-1])
            self.assertIn("chromosome", sample_record)
            self.assertIn("phenotype", sample_record)
            self.assertIn("fitness", sample_record)

            best_code_file = Path(result.best_solution_path)
            self.assertTrue(best_code_file.exists())
        finally:
            if isinstance(engine.llm_client, AsyncLLMClient):
                await engine.llm_client.aclose()


def main() -> None:
    global LIVE_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--live", action="store_true", help="Run against live Strata endpoint at 127.0.0.1:8080")
    known, remaining = parser.parse_known_args()
    LIVE_MODE = known.live
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)


if __name__ == "__main__":
    main()
