"""Domain solver for `data_analysis_XXXX` tasks using local CSV profiling in `/tmp`."""

from __future__ import annotations

import asyncio
import csv
import sys
import tempfile
from pathlib import Path

from harness.jiuwen_agent import JiuwenAgentGateway, extract_code_or_text
from harness.model_router import ModelRouter
from harness.task_scanner import TaskFeatures, is_ignored_fs_entry
from harness.unfeasible_filter import UnfeasibleTaskFilter


class DataAnalysisDomainSolver:
    """Profiles CSV files locally in Python and executes a compact aggregation script in `/tmp`."""

    def __init__(
        self,
        gateway: JiuwenAgentGateway,
        router: ModelRouter,
        unfeasible_filter: UnfeasibleTaskFilter,
        system_prompt: str,
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.unfeasible_filter = unfeasible_filter
        self.system_prompt = system_prompt

    async def solve(self, tf: TaskFeatures, qualifying_mode: bool = False) -> str | None:
        ws_dir = tf.task_dir / "workspace"
        csv_files = (
            [p for p in sorted(ws_dir.glob("*.csv")) if not is_ignored_fs_entry(p.name)]
            if ws_dir.exists()
            else []
        )
        if not csv_files:
            return None

        csv_path = csv_files[0].resolve()
        schema_preview = self._build_compact_csv_profile(csv_path)

        error_counts: list[int] = []
        last_error = ""

        for attempt in range(2):
            model = self.router.select_model(tf, attempt=attempt, qualifying_mode=qualifying_mode)
            is_reason = not model.endswith("-noreason")
            max_toks = self.router.max_tokens_for_domain("data_analysis", is_reasoning_model=is_reason)

            user_content = (
                f"Task ({tf.task_id}): {tf.prompt}\n"
                f"CSV_PATH = {str(csv_path)!r}\n"
                f"{schema_preview}\n"
                "Write a Python 3 script that reads `CSV_PATH` and prints ONLY the exact final answer."
            )
            if last_error:
                user_content += f"\nPrevious attempt error:\n{last_error}"

            raw_resp = await self.gateway.chat_completion(
                task_id=tf.task_id,
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_toks,
                temperature=0.0 if attempt == 0 else 0.2,
            )

            code = extract_code_or_text(raw_resp)
            if "CSV_PATH" not in code and (".csv" in code or "open(" in code):
                code = f"CSV_PATH = {str(csv_path)!r}\n" + code
            elif "CSV_PATH" in code and f"{str(csv_path)!r}" not in code:
                code = f"CSV_PATH = {str(csv_path)!r}\n" + code

            ok, output = await self._execute_in_tmp(code, tf.task_id)
            if ok and output.strip():
                last_line = output.strip().splitlines()[-1].strip()
                if last_line and last_line not in {"None", "nan", "NaN", "ERROR"}:
                    return last_line

            last_error = output or "Empty output"
            error_counts.append(1)
            inflight = self.unfeasible_filter.evaluate_tier2_inflight(
                error_counts=error_counts,
                missing_resource_detected=("KeyError" in last_error and not qualifying_mode),
            )
            if not inflight.is_feasible and not qualifying_mode:
                return None

        return None

    @staticmethod
    def _build_compact_csv_profile(csv_path: Path) -> str:
        headers: list[str] = []
        sample_rows: list[list[str]] = []
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
            reader = csv.reader(fp)
            try:
                headers = next(reader)
            except StopIteration:
                return "Empty CSV"
            for _ in range(3):
                try:
                    sample_rows.append(next(reader))
                except StopIteration:
                    break
        lines = [f"Columns ({len(headers)}): {headers}"]
        for idx, row in enumerate(sample_rows, start=1):
            lines.append(f"Sample row {idx}: {row}")
        return "\n".join(lines)

    @staticmethod
    async def _execute_in_tmp(code: str, task_id: str) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix=f"data_{task_id}_") as tmp_dir:
            script_path = Path(tmp_dir) / "analyze.py"
            script_path.write_text(code, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=6.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "TimeoutError"

            if proc.returncode == 0:
                return True, stdout_b.decode("utf-8", errors="replace").strip()
            return False, stderr_b.decode("utf-8", errors="replace").strip()
