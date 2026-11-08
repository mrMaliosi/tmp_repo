"""Synthetic 100-task generator across 4 competition domains (`math`, `coding`, `data_analysis`, `search`)
with configurable 20% unfeasible trap injection and macOS archive noise.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheticTaskGroundTruth:
    """Ground truth specification for a generated synthetic task."""

    task_id: str
    domain: str
    reward: float
    is_feasible: bool
    expected_answer: str | None
    solution_code: str | None = None


class SyntheticTournamentGenerator:
    """Generates realistic 100-task competition directories (`math_XXXX`, `coding_XXXX`,
    `data_analysis_XXXX`, `search_XXXX`) with random rewards (10..500) and 20% unfeasible traps.
    """

    def __init__(self, seed: int = 2026) -> None:
        self.seed = seed

    def generate_tournament_dir(
        self,
        target_dir: str | Path,
        num_tasks: int = 100,
        trap_ratio: float = 0.20,
        inject_macos_junk: bool = True,
    ) -> dict[str, SyntheticTaskGroundTruth]:
        root = Path(target_dir)
        root.mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.seed)

        if inject_macos_junk:
            (root / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
            macosx_dir = root / "__MACOSX"
            macosx_dir.mkdir(exist_ok=True)
            (macosx_dir / "._task_0000").write_text("junk", encoding="utf-8")
            (root / "._metadata").write_text("junk", encoding="utf-8")

        domains = ["math", "search", "data_analysis", "coding"]
        num_traps = int(round(num_tasks * trap_ratio))
        trap_indices = set(rng.sample(range(num_tasks), k=min(num_traps, num_tasks)))

        ground_truth: dict[str, SyntheticTaskGroundTruth] = {}

        for idx in range(num_tasks):
            domain = domains[idx % len(domains)]
            task_id = f"{domain}_{idx:04d}"
            is_trap = idx in trap_indices
            reward = float(rng.choice([10, 25, 50, 80, 120, 180, 250, 350, 500]))
            task_dir = root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            if inject_macos_junk and idx % 10 == 0:
                (task_dir / ".DS_Store").write_text("macos_ds", encoding="utf-8")
                (task_dir / "._task.json").write_text("macos_appledouble", encoding="utf-8")

            if domain == "math":
                gt = self._build_math_task(task_dir, task_id, reward, is_trap, rng)
            elif domain == "search":
                gt = self._build_search_task(task_dir, task_id, reward, is_trap, rng)
            elif domain == "data_analysis":
                gt = self._build_data_task(task_dir, task_id, reward, is_trap, rng)
            else:
                gt = self._build_coding_task(task_dir, task_id, reward, is_trap, rng)

            ground_truth[task_id] = gt

        return ground_truth

    @staticmethod
    def _build_math_task(
        task_dir: Path,
        task_id: str,
        reward: float,
        is_trap: bool,
        rng: random.Random,
    ) -> SyntheticTaskGroundTruth:
        a = rng.randint(12, 95)
        b = rng.randint(15, 85)
        c = rng.randint(3, 19)
        ans = a * b + c * c

        if is_trap:
            prompt = (
                f"Find two even integers x and y whose sum is {2 * a + 1}, "
                f"where x + y = {2 * a} and x + y = {2 * a + 14}. "
                "Note: this system has an impossible invariant and no solution exists."
            )
            payload = {
                "task_id": task_id,
                "domain": "math",
                "reward": reward,
                "question": prompt,
            }
            (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return SyntheticTaskGroundTruth(
                task_id=task_id,
                domain="math",
                reward=reward,
                is_feasible=False,
                expected_answer=None,
            )

        prompt = f"Compute the exact integer value of ({a} * {b}) + ({c} ** 2)."
        script = f"print(({a} * {b}) + ({c} ** 2))"
        payload = {
            "task_id": task_id,
            "domain": "math",
            "reward": reward,
            "question": prompt,
        }
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return SyntheticTaskGroundTruth(
            task_id=task_id,
            domain="math",
            reward=reward,
            is_feasible=True,
            expected_answer=str(ans),
            solution_code=script,
        )

    @staticmethod
    def _build_search_task(
        task_dir: Path,
        task_id: str,
        reward: float,
        is_trap: bool,
        rng: random.Random,
    ) -> SyntheticTaskGroundTruth:
        kb_dir = task_dir / "workspace" / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)

        target_person_id = f"person_{rng.randint(10, 29)}"
        target_company_id = f"company_{rng.randint(10, 29)}"
        revenue_val = rng.randint(120, 980)

        # Create 12 markdown files in kb/
        for i in range(10, 16):
            p_name = f"person_{i}"
            c_name = f"company_{i}"
            (kb_dir / f"{p_name}.md").write_text(
                f"# Profile: {p_name}\n{p_name} is Lead Architect at {c_name}.\n",
                encoding="utf-8",
            )
            (kb_dir / f"{c_name}.md").write_text(
                f"# Organization: {c_name}\n{c_name} recorded revenue: {100 + i} million.\n",
                encoding="utf-8",
            )

        if is_trap:
            # Query references person_9999 which is completely absent from workspace/kb/
            prompt = "What is the annual revenue of the company where person_9999 works?"
            payload = {
                "task_id": task_id,
                "domain": "search",
                "reward": reward,
                "question": prompt,
            }
            (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return SyntheticTaskGroundTruth(
                task_id=task_id,
                domain="search",
                reward=reward,
                is_feasible=False,
                expected_answer=None,
            )

        # Ensure target person and company exist with exact facts
        (kb_dir / f"{target_person_id}.md").write_text(
            f"# Profile: {target_person_id}\n{target_person_id} serves as Chief Scientist at {target_company_id}.\n",
            encoding="utf-8",
        )
        (kb_dir / f"{target_company_id}.md").write_text(
            f"# Organization: {target_company_id}\n{target_company_id} reported annual revenue {revenue_val}.\n",
            encoding="utf-8",
        )
        prompt = f"What is the numeric annual revenue of {target_company_id} where {target_person_id} works?"
        payload = {
            "task_id": task_id,
            "domain": "search",
            "reward": reward,
            "question": prompt,
        }
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return SyntheticTaskGroundTruth(
            task_id=task_id,
            domain="search",
            reward=reward,
            is_feasible=True,
            expected_answer=str(revenue_val),
            solution_code=str(revenue_val),
        )

    @staticmethod
    def _build_data_task(
        task_dir: Path,
        task_id: str,
        reward: float,
        is_trap: bool,
        rng: random.Random,
    ) -> SyntheticTaskGroundTruth:
        ws_dir = task_dir / "workspace"
        ws_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ws_dir / "sales.csv"

        vals = [rng.randint(10, 90) for _ in range(8)]
        total_revenue = sum(vals)
        rows = ["id,region,revenue_usd"] + [
            f"{idx},EMEA,{v}" for idx, v in enumerate(vals, start=1)
        ]
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        if is_trap:
            # Prompt asks for a non-existent column `non_existent_margin_col`
            prompt = "Calculate the sum of column `non_existent_margin_col` in `sales.csv`."
            payload = {
                "task_id": task_id,
                "domain": "data_analysis",
                "reward": reward,
                "question": prompt,
            }
            (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return SyntheticTaskGroundTruth(
                task_id=task_id,
                domain="data_analysis",
                reward=reward,
                is_feasible=False,
                expected_answer=None,
            )

        prompt = "Compute the exact integer sum of `revenue_usd` across all rows in `sales.csv`."
        script = (
            "import csv\n"
            "with open(CSV_PATH, 'r', encoding='utf-8') as f:\n"
            "    r = csv.DictReader(f)\n"
            "    print(sum(int(row['revenue_usd']) for row in r))\n"
        )
        payload = {
            "task_id": task_id,
            "domain": "data_analysis",
            "reward": reward,
            "question": prompt,
        }
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return SyntheticTaskGroundTruth(
            task_id=task_id,
            domain="data_analysis",
            reward=reward,
            is_feasible=True,
            expected_answer=str(total_revenue),
            solution_code=script,
        )

    @staticmethod
    def _build_coding_task(
        task_dir: Path,
        task_id: str,
        reward: float,
        is_trap: bool,
        rng: random.Random,
    ) -> SyntheticTaskGroundTruth:
        ws_dir = task_dir / "workspace"
        ws_dir.mkdir(parents=True, exist_ok=True)
        factor = rng.randint(2, 9)

        # Buggy implementation subtracts instead of multiplying
        mylib_buggy = (
            "def compute_metric(x: int, y: int) -> int:\n"
            f"    return (x - y) * {factor}\n"
        )
        mylib_fixed = (
            "def compute_metric(x: int, y: int) -> int:\n"
            f"    return (x + y) * {factor}\n"
        )
        (ws_dir / "mylib.py").write_text(mylib_buggy, encoding="utf-8")

        if is_trap:
            # Contradictory assertions in test_mylib.py: compute_metric(2, 3) == 10 and == 999
            test_contradictory = (
                "from mylib import compute_metric\n\n"
                "def test_contradiction():\n"
                "    assert compute_metric(2, 3) == 10\n"
                "    assert compute_metric(2, 3) == 999\n"
            )
            (ws_dir / "test_mylib.py").write_text(test_contradictory, encoding="utf-8")
            payload = {
                "task_id": task_id,
                "domain": "coding",
                "reward": reward,
                "question": "Fix `compute_metric` in `mylib.py` so that `test_mylib.py` passes.",
            }
            (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return SyntheticTaskGroundTruth(
                task_id=task_id,
                domain="coding",
                reward=reward,
                is_feasible=False,
                expected_answer=None,
            )

        test_valid = (
            "from mylib import compute_metric\n\n"
            "def test_basic():\n"
            f"    assert compute_metric(2, 3) == {5 * factor}\n"
            f"    assert compute_metric(10, 5) == {15 * factor}\n"
        )
        (ws_dir / "test_mylib.py").write_text(test_valid, encoding="utf-8")
        payload = {
            "task_id": task_id,
            "domain": "coding",
            "reward": reward,
            "question": f"Fix `compute_metric(x, y)` in `mylib.py` so it returns `(x + y) * {factor}` and passes `test_mylib.py`.",
        }
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return SyntheticTaskGroundTruth(
            task_id=task_id,
            domain="coding",
            reward=reward,
            is_feasible=True,
            expected_answer=mylib_fixed.strip(),
            solution_code=mylib_fixed.strip(),
        )
