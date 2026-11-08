"""Demo: run the harness against the REAL organizer sample tasks (math_0000, search_0001)
using a mock LLM proxy built from competition_tasks/ground_truth.json.
Simulates Stage 3 (qualifying run on 2 feasible tasks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eblo.competition_case.synthetic_generator import SyntheticTaskGroundTruth
from eblo.competition_case.tournament_sim import build_realistic_mock_proxy
from run_agent import TournamentHarnessRunner

TASKS_DIR = PROJECT_ROOT / "competition_tasks"


def load_ground_truth() -> dict[str, SyntheticTaskGroundTruth]:
    raw = json.loads((TASKS_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    gt = {}
    for tid, spec in raw.items():
        gt[tid] = SyntheticTaskGroundTruth(
            task_id=tid,
            domain=spec["domain"],
            reward=10.0,
            is_feasible=True,
            expected_answer=spec["expected"],
            solution_code=None,
        )
    return gt


async def main() -> None:
    gt_map = load_ground_truth()
    mock = build_realistic_mock_proxy(gt_map)

    runner = TournamentHarnessRunner(
        tasks_dir=TASKS_DIR,
        out_path=Path("/tmp") / "demo_out.json" if sys.platform != "win32" else Path.cwd() / "demo_out.json",
        global_budget_limit=10_000.0,
        max_workers=2,
        mock_handler=mock,
    )
    summary = await runner.execute_tournament()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    for tid, g in gt_map.items():
        ans = TASKS_DIR / tid / "answer.txt"
        got = ans.read_text(encoding="utf-8").strip() if ans.exists() else "<missing>"
        status = "OK " if got == (g.expected_answer or "").strip() else "FAIL"
        print(f"[{status}] {tid}: expected={g.expected_answer!r} got={got!r}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
