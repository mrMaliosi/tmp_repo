"""Stage 4 Verification: Full 100-task tournament simulation (`tests/test_full_100_sim.py`).

Verifies:
1. Fast zero-token scan (< 1.5s) across 100 tasks (`math`, `coding`, `data_analysis`, `search`)
   while ignoring macOS `.DS_Store`, `__MACOSX`, and `._*` junk files.
2. Three-tier unfeasible trap filtering (20% unsolvable tasks detected at 0 tokens).
3. Exact effective token accounting (`C_eff = T_completion + T_reasoning + 0.1 * T_prompt`).
4. Atomic persistence of `/data/tasks/<task_id>/answer.txt` and `$OUT` across 5 concurrent workers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import tempfile
import unittest
from pathlib import Path

from eblo.competition_case.synthetic_generator import SyntheticTournamentGenerator
from eblo.competition_case.tournament_sim import build_realistic_mock_proxy
from harness.budget_ledger import compute_effective_tokens
from run_agent import TournamentHarnessRunner


class TestFull100TournamentStage4(unittest.IsolatedAsyncioTestCase):
    """Simulates Stage 4 of the competition (100 tasks across 4 domains + 20% unfeasible traps)."""

    def test_effective_token_formula_exactness(self) -> None:
        # C_eff = T_completion + T_reasoning + 0.1 * T_prompt
        cost = compute_effective_tokens(prompt_tokens=1000, completion_tokens=150, reasoning_tokens=300)
        self.assertAlmostEqual(cost, 150.0 + 300.0 + 100.0, places=5)

    async def test_full_100_tasks_simulation_and_atomic_persistence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eblo_stage4_100_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            tasks_dir = tmp_dir / "tasks"
            out_file = tmp_dir / "tournament_out.json"

            generator = SyntheticTournamentGenerator(seed=2026)
            gt_map = generator.generate_tournament_dir(
                tasks_dir,
                num_tasks=100,
                trap_ratio=0.20,
                inject_macos_junk=True,
            )
            self.assertEqual(len(gt_map), 100)

            mock_handler = build_realistic_mock_proxy(gt_map, accuracy_scale=1e9, noise_sigma=0.0)
            runner = TournamentHarnessRunner(
                tasks_dir=tasks_dir,
                out_path=out_file,
                global_budget_limit=40_000.0,
                max_workers=5,
                mock_handler=mock_handler,
            )
            summary = await runner.execute_tournament()

            # 1. Verify zero-token scanner speed (< 1.5s) and macOS junk immunity
            self.assertEqual(summary["scanned_tasks_total"], 100)
            self.assertLess(summary["scan_elapsed_sec"], 1.5)
            self.assertFalse(summary["qualifying_mode"])

            # 2. Verify all 20 unfeasible trap tasks were caught by the filter
            self.assertEqual(summary["static_rejected_traps"], 20)

            # 3. Verify high solve rate on the 80 feasible tasks and atomic answer.txt creation
            self.assertGreaterEqual(summary["solved_tasks"], 75)
            self.assertLess(summary["total_effective_tokens"], 15_000.0)

            verified_correct = 0
            for tid, gt in gt_map.items():
                ans_file = tasks_dir / tid / "answer.txt"
                if gt.is_feasible:
                    if ans_file.exists():
                        actual = ans_file.read_text(encoding="utf-8").strip()
                        self.assertEqual(actual, (gt.expected_answer or "").strip())
                        verified_correct += 1
                else:
                    self.assertFalse(ans_file.exists(), f"Trap task {tid} should not have answer.txt")

            self.assertGreaterEqual(verified_correct, 75)

            # 4. Verify $OUT file synchronization
            self.assertTrue(out_file.exists())
            out_data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(out_data["solved_count"], verified_correct)
            self.assertGreater(out_data["total_reward_earned"], 12_000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
