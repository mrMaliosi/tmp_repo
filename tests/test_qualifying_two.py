"""Stage 3 Verification: Qualifying round test on 2 feasible tasks (`tests/test_qualifying_two.py`).

Verifies that when `<= 5` tasks are present in `--tasks-dir`, the harness automatically
activates `qualifying_mode=True`, disables aggressive trap rejection, solves 100% of tasks
(2/2), and atomically writes `answer.txt` in each task directory as well as `$OUT`.
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
from run_agent import TournamentHarnessRunner


class TestQualifyingTwoStage3(unittest.IsolatedAsyncioTestCase):
    """Simulates Stage 3 of the competition (Qualifying run on 2 feasible tasks)."""

    async def test_qualifying_two_tasks_100_percent_solved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eblo_stage3_qual_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            tasks_dir = tmp_dir / "tasks"
            out_file = tmp_dir / "qualifying_out.json"

            generator = SyntheticTournamentGenerator(seed=101)
            gt_map = generator.generate_tournament_dir(
                tasks_dir,
                num_tasks=2,
                trap_ratio=0.0,
                inject_macos_junk=True,
            )
            self.assertEqual(len(gt_map), 2)

            mock_handler = build_realistic_mock_proxy(gt_map, accuracy_scale=1e9)
            runner = TournamentHarnessRunner(
                tasks_dir=tasks_dir,
                out_path=out_file,
                global_budget_limit=10_000.0,
                max_workers=2,
                mock_handler=mock_handler,
            )
            summary = await runner.execute_tournament()

            self.assertTrue(summary["qualifying_mode"])
            self.assertEqual(summary["scanned_tasks_total"], 2)
            self.assertEqual(summary["solved_tasks"], 2)
            self.assertEqual(summary["static_rejected_traps"], 0)

            for tid, gt in gt_map.items():
                ans_file = tasks_dir / tid / "answer.txt"
                self.assertTrue(ans_file.exists(), f"Missing answer.txt for {tid}")
                actual = ans_file.read_text(encoding="utf-8").strip()
                self.assertEqual(actual, (gt.expected_answer or "").strip())

            self.assertTrue(out_file.exists())
            out_data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(out_data["solved_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
