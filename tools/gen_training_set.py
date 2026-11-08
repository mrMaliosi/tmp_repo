"""Generate a local 100-task training tournament in the organizer's format.

Creates `train_tasks_100/` (task folders with task.json + workspace, macOS junk
included) and `train_ground_truth.json` for post-run calibration of the
evolution simulator (per-domain solve rates, token usage).

Usage:
    python tools/gen_training_set.py [--num 100] [--trap-ratio 0.2] [--seed 2026]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eblo.competition_case.synthetic_generator import SyntheticTournamentGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--trap-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default=str(_PROJECT_ROOT / "train_tasks_100"))
    parser.add_argument("--gt-file", type=str, default=str(_PROJECT_ROOT / "train_ground_truth.json"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    gt_file = Path(args.gt_file)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"WARNING: {out_dir} is not empty; answer.txt files from previous runs remain.")

    gen = SyntheticTournamentGenerator(seed=args.seed)
    gt_map = gen.generate_tournament_dir(
        out_dir,
        num_tasks=args.num,
        trap_ratio=args.trap_ratio,
        inject_macos_junk=True,
    )

    payload = {
        tid: {
            "domain": gt.domain,
            "reward": gt.reward,
            "is_feasible": gt.is_feasible,
            "expected": gt.expected_answer,
        }
        for tid, gt in sorted(gt_map.items())
    }
    gt_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    feasible = sum(1 for g in gt_map.values() if g.is_feasible)
    print(f"Generated {len(gt_map)} tasks ({feasible} feasible, {len(gt_map) - feasible} traps).")
    print(f"  tasks dir : {out_dir}")
    print(f"  ground truth: {gt_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
