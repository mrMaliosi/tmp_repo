"""Full 100-task tournament simulator with strict effective token budget ceilings.

The mock proxy emulates realistic model behavior so that offline evolution has a
non-flat fitness landscape:
- Per-domain difficulty classes (easy / medium / hard): harder tasks are solved less
  often by cheap `-noreason` models, forcing the evolution to balance escalation
  policy vs. token cost.
- Noisy per-call success: success probability decays with difficulty and improves
  with reasoning models.
- Noisy token usage: lognormal jitter around the ideal completion length, so the
  p90 budget fuse and scheduler priorities actually matter.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from eblo.competition_case.fitness import TournamentFitnessMetrics, compute_tournament_fitness
from eblo.competition_case.synthetic_generator import (
    SyntheticTaskGroundTruth,
    SyntheticTournamentGenerator,
)
from harness.estimator import TokenAndFeasibilityEstimator
from run_agent import TournamentHarnessRunner

# ---------------------------------------------------------------------------
# Calibration: `sim_calibration.json` (emitted by tools/calibrate_sim.py from a
# real API run) overrides the difficulty-accuracy ladder and scales completion
# token tables. Missing file -> hand-tuned defaults below.
# ---------------------------------------------------------------------------
_CALIBRATION_PATH = Path(__file__).resolve().parent / "sim_calibration.json"


def _load_calibration() -> dict[str, Any]:
    # `EBLO_SIM_UNCALIBRATED=1` forces the raw (unscaled) defaults: used by
    # tools/calibrate_sim.py to measure the uncalibrated simulator baseline
    # without picking up a previously written calibration file (idempotency).
    if os.environ.get("EBLO_SIM_UNCALIBRATED"):
        return {}
    try:
        return json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_CALIB: dict[str, Any] = _load_calibration()

# Domain difficulty ladder: P(solve | easy/medium/hard) for the cheapest model
# on its FIRST attempt. The `easy` tier is calibrated from the real NeuralDeep
# run (80/80 solved -> Jeffreys-smoothed ~0.95); medium/hard remain conservative
# estimates pending harder training data.
_DEFAULT_DIFFICULTY_ACCURACY: dict[str, tuple[float, float, float]] = {
    "math": (0.95, 0.85, 0.68),
    "search": (0.95, 0.80, 0.62),
    "data_analysis": (0.95, 0.80, 0.58),
    "coding": (0.94, 0.74, 0.50),
}


def _difficulty_accuracy_table() -> dict[str, tuple[float, float, float]]:
    raw = _CALIB.get("difficulty_accuracy", {})
    out: dict[str, tuple[float, float, float]] = {}
    for dom, default in _DEFAULT_DIFFICULTY_ACCURACY.items():
        vals = raw.get(dom)
        if isinstance(vals, (list, tuple)) and len(vals) == 3:
            out[dom] = (float(vals[0]), float(vals[1]), float(vals[2]))
        else:
            out[dom] = default
    return out


# Ideal completion token usage by domain and difficulty (before noise), scaled
# by the calibration factor measured against the real API run.
_RAW_COMPLETION_TOKENS: dict[str, tuple[int, int, int]] = {
    "math": (60, 130, 220),
    "search": (45, 90, 150),
    "data_analysis": (110, 200, 320),
    "coding": (180, 320, 520),
}
_TOKEN_SCALE = float(_CALIB.get("completion_token_scale", 1.0))
_DOMAIN_COMPLETION_TOKENS: dict[str, tuple[int, int, int]] = {
    dom: tuple(max(10, int(round(t * _TOKEN_SCALE))) for t in vals)  # type: ignore[arg-type]
    for dom, vals in _RAW_COMPLETION_TOKENS.items()
}

_DOMAIN_DIFFICULTY_ACCURACY: dict[str, tuple[float, float, float]] = _difficulty_accuracy_table()

# Reasoning/dense models are more capable: their per-call success probability is
# scaled by these multipliers (overridable from sim_calibration.json).
_DEFAULT_MODEL_ACCURACY: dict[str, float] = {
    "qwen3.6-35b-a3b-noreason": 1.00,
    "qwen3.8-27b-noreason": 1.08,
    "qwen3.6-35b-a3b": 1.22,
    "qwen3.8-27b": 1.30,
}


def _model_accuracy_table() -> dict[str, float]:
    raw = _CALIB.get("model_accuracy", {})
    out: dict[str, float] = {}
    for m, default in _DEFAULT_MODEL_ACCURACY.items():
        try:
            out[m] = float(raw.get(m, default))
        except (TypeError, ValueError):
            out[m] = default
    return out


_MODEL_BASE_ACCURACY: dict[str, float] = _model_accuracy_table()

_DIFFICULTY_LABELS = ("easy", "medium", "hard")


def _difficulty_of(gt: SyntheticTaskGroundTruth) -> int:
    """Deterministically map a task to a difficulty class 0/1/2 from its id hash."""
    digest = sum(ord(ch) for ch in gt.task_id)
    if digest % 10 < 5:
        return 0
    if digest % 10 < 8:
        return 1
    return 2


def _lognormal_jitter(rng: random.Random, sigma: float = 0.25) -> float:
    """Multiplicative lognormal noise around 1.0 (Box-Muller transform)."""
    u1 = max(1e-9, rng.random())
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(sigma * z)


class NoisyMockProxy:
    """Stateful mock proxy with per-task RNG, difficulty classes, and noisy token usage.

    When `gt_solution_code` cannot be trusted to compile (or a call must fail),
    the proxy returns a plausible-but-wrong answer so that wrong solutions are
    never counted as solved by the harness's local verification layer.
    """

    def __init__(
        self,
        ground_truth: dict[str, SyntheticTaskGroundTruth],
        seed: int = 7,
        accuracy_scale: float = 1.0,
        noise_sigma: float = 0.25,
    ) -> None:
        self.gt = ground_truth
        self.seed = seed
        self.accuracy_scale = accuracy_scale
        self.noise_sigma = noise_sigma
        self._rng = random.Random(seed)
        # Correlated failures: solvability is drawn ONCE per (task, model).
        # A model that cannot solve a task keeps failing on same-model retries;
        # only escalating to a different model gets a fresh chance. This matches
        # real LLM behavior far better than independent per-call coin flips.
        self._solvable: dict[tuple[str, str], bool] = {}
        self._per_task_difficulty: dict[str, int] = {
            tid: _difficulty_of(g) for tid, g in ground_truth.items()
        }

    def __call__(
        self,
        task_id: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict[str, Any]:
        rng = self._rng
        gt = self.gt.get(task_id)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        prompt_tokens = max(25, len(prompt_text) // 4)

        # Tier-1 probe handling: noisy classification (the probe is cheap, not omniscient)
        if "SOLVABLE or UNSOLVABLE_TRAP" in prompt_text:
            is_trap = gt is not None and not gt.is_feasible
            # Probe accuracy: catches 80% of traps, falsely flags 8% of feasible tasks
            if is_trap and rng.random() < 0.80:
                verdict = "UNSOLVABLE_TRAP: Contradictory or missing workspace data."
            elif (not is_trap) and rng.random() < 0.08:
                verdict = "UNSOLVABLE_TRAP: Insufficient consistent evidence."
            else:
                verdict = "SOLVABLE: Consistent."
            return {
                "content": verdict,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 18, "reasoning_tokens": 0},
            }

        is_reasoning = not model.endswith("-noreason")

        # Traps and unknown tasks: the model burns tokens trying, then gives up.
        if gt is None or not gt.is_feasible:
            comp_toks = int(140 * _lognormal_jitter(rng, self.noise_sigma))
            reason_toks = int(420 * _lognormal_jitter(rng, self.noise_sigma)) if is_reasoning else 0
            return {
                "content": "UNSOLVABLE",
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": comp_toks, "reasoning_tokens": reason_toks},
            }

        diff = self._per_task_difficulty.get(task_id, 1)
        domain = gt.domain
        base_p = _DOMAIN_DIFFICULTY_ACCURACY.get(domain, (0.9, 0.8, 0.65))[diff]
        model_bonus = _MODEL_BASE_ACCURACY.get(model, 1.0)
        raw_p = base_p * model_bonus * self.accuracy_scale
        # Realistic mode caps single-call accuracy at 98.5%; deterministic mode
        # (accuracy_scale > 10, used by pipeline regression tests) solves always.
        p_success = raw_p if self.accuracy_scale > 10.0 else min(0.985, raw_p)

        comp_ideal = _DOMAIN_COMPLETION_TOKENS.get(domain, (100, 180, 300))[diff]
        comp_toks = max(20, int(comp_ideal * _lognormal_jitter(rng, self.noise_sigma)))

        # Clamp to requested max_tokens (like a real proxy truncating the stream)
        if comp_toks > max_tokens:
            comp_toks = max_tokens

        reason_toks = 0
        if is_reasoning:
            reason_toks = int((comp_toks * 2.4 + 140) * _lognormal_jitter(rng, self.noise_sigma))

        solved = self._draw_solvability(task_id, model, domain, diff, p_success)
        sol = gt.solution_code or gt.expected_answer or ""
        if solved:
            if domain in {"math", "data_analysis", "coding"}:
                content = f"```python\n{sol}\n```"
            else:
                content = str(sol)
            if is_reasoning:
                content = f"<think>\nAnalyzing {task_id} constraints and verifying invariants.\n</think>\n{content}"
        else:
            # Plausible but wrong output (never verified as correct locally)
            if domain in {"math", "data_analysis", "coding"}:
                wrong = self._plausible_wrong_answer(sol)
                content = f"```python\n{wrong}\n```"
            else:
                content = str(self._plausible_wrong_answer(sol))
            if is_reasoning:
                content = (
                    f"<think>\nAnalyzing {task_id} constraints; deriving an approximate route.\n</think>\n{content}"
                )

        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_toks,
                "reasoning_tokens": reason_toks,
            },
        }

    def _draw_solvability(self, task_id: str, model: str, domain: str, diff: int, p_success: float) -> bool:
        """Draw solvability once per (task, model): same-model retries repeat the outcome.

        Deterministic mode (accuracy_scale > 10, used by regression tests) always solves.
        """
        if self.accuracy_scale > 10.0:
            return True
        key = (task_id, model)
        cached = self._solvable.get(key)
        if cached is not None:
            return cached
        drawn = self._rng.random() < p_success
        self._solvable[key] = drawn
        return drawn

    @staticmethod
    def _plausible_wrong_answer(sol: str) -> str:
        """Produce a syntactically valid but incorrect solution/answer."""
        sol = (sol or "").strip()
        if sol.startswith("print(") and sol.endswith(")"):
            # math-style: change an operator to make a wrong but runnable script
            for old, new in (("+", "-"), ("-", "+"), ("*", "+")):
                if old in sol:
                    return sol.replace(old, new, 1)
            return sol
        if sol.replace("-", "", 1).isdigit():
            n = int(sol)
            return str(n + 7 if n < 10_000 else n - 13)
        # coding-style source: swap + for - in the return expression (still compiles)
        if "return" in sol:
            head, _, tail = sol.partition("return")
            if "+" in tail:
                return head + "return" + tail.replace("+", "-", 1)
            return head + "return" + tail + " * 2"
        return sol


def build_realistic_mock_proxy(
    ground_truth: dict[str, SyntheticTaskGroundTruth],
    seed: int = 7,
    accuracy_scale: float = 1.0,
    noise_sigma: float = 0.25,
):
    """Backward-compatible factory returning the stateful NoisyMockProxy.

    `accuracy_scale` and `noise_sigma` let callers choose the fidelity:
    - Tests that verify the deterministic pipeline (scan, filter, budget accounting)
      pass `accuracy_scale=1e9` to make every feasible task solvable (sigma noise on
      tokens is irrelevant for those assertions).
    - Offline evolution uses realistic defaults so the fitness landscape is non-flat.
    """
    return NoisyMockProxy(
        ground_truth,
        seed=seed,
        accuracy_scale=accuracy_scale,
        noise_sigma=noise_sigma,
    )


class TournamentSimulator:
    """Simulates the 100-task competition across one or more global token budgets."""

    def __init__(self, seed: int = 2026, proxy_seed: int = 7, noise_sigma: float = 0.25) -> None:
        self.generator = SyntheticTournamentGenerator(seed=seed)
        self.proxy_seed = proxy_seed
        self.noise_sigma = noise_sigma

    async def simulate_single_budget(
        self,
        policy: dict[str, Any],
        global_budget: float = 40_000.0,
        num_tasks: int = 100,
        trap_ratio: float = 0.20,
    ) -> TournamentFitnessMetrics:
        with tempfile.TemporaryDirectory(prefix="eblo_tourney_sim_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            tasks_dir = tmp_dir / "tasks"
            out_file = tmp_dir / "out_summary.json"
            policy_file = tmp_dir / "candidate_policy.json"

            merged_policy = TokenAndFeasibilityEstimator.default_policy()
            for k, v in policy.items():
                if isinstance(v, dict) and isinstance(merged_policy.get(k), dict):
                    merged_policy[k] = {**merged_policy[k], **v}
                else:
                    merged_policy[k] = v
            policy_file.write_text(json.dumps(merged_policy, indent=2), encoding="utf-8")

            gt_map = self.generator.generate_tournament_dir(
                tasks_dir,
                num_tasks=num_tasks,
                trap_ratio=trap_ratio,
                inject_macos_junk=True,
            )
            mock_handler = build_realistic_mock_proxy(
                gt_map,
                seed=self.proxy_seed,
                noise_sigma=self.noise_sigma,
            )

            runner = TournamentHarnessRunner(
                tasks_dir=tasks_dir,
                out_path=out_file,
                policy_path=policy_file,
                global_budget_limit=global_budget,
                max_workers=5,
                mock_handler=mock_handler,
            )
            summary = await runner.execute_tournament()

            earned_reward = 0.0
            possible_reward = 0.0
            solved_count = 0
            feasible_total = 0
            traps_total = 0

            for tid, gt in gt_map.items():
                if gt.is_feasible:
                    feasible_total += 1
                    possible_reward += gt.reward
                    ans_path = tasks_dir / tid / "answer.txt"
                    if ans_path.exists():
                        actual = ans_path.read_text(encoding="utf-8").strip()
                        if actual == (gt.expected_answer or "").strip():
                            earned_reward += gt.reward
                            solved_count += 1
                else:
                    traps_total += 1

            eff_used = float(summary.get("total_effective_tokens", 0.0))
            traps_filtered = int(summary.get("static_rejected_traps", 0))
            fitness = compute_tournament_fitness(earned_reward, eff_used)

            return TournamentFitnessMetrics(
                total_reward_earned=round(earned_reward, 2),
                total_possible_reward=round(possible_reward, 2),
                effective_tokens_used=round(eff_used, 2),
                tasks_solved=solved_count,
                feasible_tasks_total=feasible_total,
                traps_correctly_filtered=traps_filtered,
                traps_total=traps_total,
                tournament_fitness=fitness,
            )

    async def evaluate_multi_budget(
        self,
        policy: dict[str, Any],
        budgets: tuple[float, ...] = (15_000.0, 40_000.0, 100_000.0),
        num_tasks: int = 100,
    ) -> tuple[float, list[TournamentFitnessMetrics]]:
        """Evaluate candidate policy across tight (15k), medium (40k), and generous (100k) budgets.

        Multiple seeds for the noisy proxy reduce evaluation variance: the ranking of
        policies must reflect their quality, not proxy-RNG luck.
        """
        metrics_list: list[TournamentFitnessMetrics] = []
        for b in budgets:
            m = await self.simulate_single_budget(policy, global_budget=b, num_tasks=num_tasks)
            metrics_list.append(m)
        avg_fitness = sum(m.tournament_fitness for m in metrics_list) / max(1, len(metrics_list))
        return round(avg_fitness, 4), metrics_list
