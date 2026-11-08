"""Calibrate the tournament simulator against a real API run.

Inputs:
  - run log  (default `real_run_out.json.log`, JSONL with per-call `llm_call` events;
              produced by the harness after the per-call logging patch)
  - run summary (`real_run_out.json`, aggregate C_eff for cross-checking)
  - ground truth (`train_ground_truth.json`)

Outputs `eblo/competition_case/sim_calibration.json`:
  - completion_token_scale : scales the simulator's per-call completion tokens so
    the simulated tournament matches the real run's effective-token total
  - difficulty_accuracy    : per-domain (easy, medium, hard) first-attempt solve
    probabilities for the tier-1 model. `easy` is measured (Jeffreys-smoothed
    real solve rate); medium/hard are kept conservative since the training set
    is too easy to estimate them from data.

Usage:
    python tools/calibrate_sim.py [--log real_run_out.json.log] [--gt train_ground_truth.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Deterministic simulator measurement of the same tournament (seed 2026, the
# training set) with the shipped evolved policy: the completion-token scale is
# the ratio of real total C_eff to the simulated one (prompt tokens are
# generated identically by both, so the scale acts on the completion part).
_SIM_TARGET_NOTE = "scale = (real_C_eff - 0.1*sim_prompt - sim_reasoning) / sim_completion"


def _jeffreys(solved: int, total: int) -> float:
    """Posterior-mean Beta(0.5, 0.5) estimate of a solve rate (avoids 0 and 1)."""
    return (solved + 0.5) / (total + 1.0)


def _difficulty_of(task_id: str) -> int:
    digest = sum(ord(ch) for ch in task_id)
    if digest % 10 < 5:
        return 0
    if digest % 10 < 8:
        return 1
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=str, default=str(_PROJECT_ROOT / "real_run_out.json.log"))
    parser.add_argument("--summary", type=str, default=str(_PROJECT_ROOT / "real_run_out.json"))
    parser.add_argument("--gt", type=str, default=str(_PROJECT_ROOT / "train_ground_truth.json"))
    parser.add_argument(
        "--out",
        type=str,
        default=str(_PROJECT_ROOT / "eblo" / "competition_case" / "sim_calibration.json"),
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    summary_path = Path(args.summary)
    gt_path = Path(args.gt)
    out_path = Path(args.out)

    # ---- 1. Real run aggregate + per-call events --------------------------------
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    real_c_eff = float(summary["total_effective_tokens"])
    real_solved = int(summary["solved_count"])

    llm_calls = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "llm_call":
                llm_calls.append(rec)

    has_per_call = bool(llm_calls)
    print(f"Real run: solved {real_solved}, C_eff {real_c_eff:.1f}, "
          f"{len(llm_calls)} per-call events ({'yes' if has_per_call else 'NO -> falling back to aggregate-only'})")

    # ---- 2. Solve-rate calibration (difficulty_accuracy) ------------------------
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    cells: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])  # (domain, diff) -> [solved, total]
    answered = set(summary.get("answers", {}).keys())
    for tid, g in gt.items():
        if not g.get("is_feasible"):
            continue
        cell = cells[(g["domain"], _difficulty_of(tid))]
        cell[1] += 1
        if tid in answered:
            cell[0] += 1

    easy_rates: dict[str, float] = {}
    print("\nPer-domain solve rates (easy tier, Jeffreys-smoothed):")
    for dom in ("math", "search", "data_analysis", "coding"):
        solved, total = cells.get((dom, 0), [0, 0])
        if total > 0:
            easy_rates[dom] = round(_jeffreys(solved, total), 3)
        else:
            easy_rates[dom] = 0.95
        print(f"  {dom:>14}: {solved}/{total} -> {easy_rates[dom]}")

    # medium/hard stay conservative (training set is mostly easy-tier tasks;
    # real medium/hard solve rates cannot be estimated from 80/80-solved data)
    default_ladder = {
        "math": (None, 0.85, 0.68),
        "search": (None, 0.80, 0.62),
        "data_analysis": (None, 0.80, 0.58),
        "coding": (None, 0.74, 0.50),
    }
    difficulty_accuracy = {}
    for dom, (easy, med, hard) in default_ladder.items():
        difficulty_accuracy[dom] = [easy_rates[dom], med, hard]

    # ---- 3. Completion-token scale ----------------------------------------------
    # Deterministic simulation of the same tournament with the shipped policy.
    # Force the raw simulator tables (no previously written calibration file)
    # so repeated invocations remain idempotent.
    os.environ["EBLO_SIM_UNCALIBRATED"] = "1"
    import asyncio
    import tempfile

    from eblo.competition_case.synthetic_generator import SyntheticTournamentGenerator
    from eblo.competition_case.tournament_sim import build_realistic_mock_proxy
    from run_agent import TournamentHarnessRunner

    async def _measure_sim() -> tuple[int, int, int]:
        with tempfile.TemporaryDirectory(prefix="calib_sim_") as td:
            tasks = Path(td) / "tasks"
            gen = SyntheticTournamentGenerator(seed=2026)
            gt_map = gen.generate_tournament_dir(
                tasks, num_tasks=100, trap_ratio=0.20, inject_macos_junk=True
            )
            mock = build_realistic_mock_proxy(gt_map, accuracy_scale=1e9, noise_sigma=0.0)
            runner = TournamentHarnessRunner(
                tasks_dir=tasks,
                out_path=Path(td) / "out.json",
                global_budget_limit=1_000_000.0,
                max_workers=5,
                mock_handler=mock,
            )
            s = await runner.execute_tournament()
            return (
                int(s["total_prompt_tokens"]),
                int(s["total_completion_tokens"]),
                int(s["total_reasoning_tokens"]),
            )

    sim_prompt, sim_completion, sim_reasoning = asyncio.run(_measure_sim())
    os.environ.pop("EBLO_SIM_UNCALIBRATED", None)
    target_completion_c_eff = real_c_eff - 0.1 * sim_prompt - sim_reasoning
    if target_completion_c_eff <= 0:
        raise SystemExit("Real run C_eff is smaller than the simulated prompt overhead; cannot calibrate.")
    scale = target_completion_c_eff / sim_completion
    print(f"\nDeterministic sim: prompt={sim_prompt}, completion={sim_completion}, reasoning={sim_reasoning}")
    print(f"completion_token_scale = ({real_c_eff:.1f} - {0.1 * sim_prompt:.1f} - {sim_reasoning}) / {sim_completion} = {scale:.4f}")

    # ---- 4. Persist --------------------------------------------------------------
    payload = {
        "source": {
            "run_summary": str(summary_path.name),
            "run_log": str(log_path.name) if log_path.exists() else None,
            "ground_truth": str(gt_path.name),
            "real_c_eff": real_c_eff,
            "real_solved": real_solved,
            "note": _SIM_TARGET_NOTE,
        },
        "completion_token_scale": round(scale, 4),
        "difficulty_accuracy": difficulty_accuracy,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
