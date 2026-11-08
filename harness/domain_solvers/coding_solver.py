"""Domain solver for `coding_XXXX` tasks with local `/tmp` test execution and verification."""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from harness.jiuwen_agent import JiuwenAgentGateway, extract_code_or_text
from harness.model_router import ModelRouter
from harness.task_scanner import TaskFeatures, is_ignored_fs_entry
from harness.unfeasible_filter import UnfeasibleTaskFilter


class CodingDomainSolver:
    """Copies `workspace/` into `/tmp/work_<task_id>`, runs local tests to localize failures,
    applies LLM patches, and strictly verifies all tests pass before returning the solution.
    """

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
        if not ws_dir.exists():
            return None

        with tempfile.TemporaryDirectory(prefix=f"work_{tf.task_id}_") as tmp_dir_str:
            work_dir = Path(tmp_dir_str)
            self._copy_clean_workspace(ws_dir, work_dir)

            target_py, test_py = self._locate_target_and_test_files(work_dir)
            if target_py is None or test_py is None:
                return None

            passed, err_count, failure_output = await self._run_tests(work_dir, test_py)
            if passed:
                return target_py.read_text(encoding="utf-8").strip()

            error_history: list[int] = [max(1, err_count)]
            original_code = target_py.read_text(encoding="utf-8", errors="replace")

            for attempt in range(2):
                model = self.router.select_model(tf, attempt=attempt, qualifying_mode=qualifying_mode)
                is_reason = not model.endswith("-noreason")
                max_toks = self.router.max_tokens_for_domain("coding", is_reasoning_model=is_reason)

                test_code = test_py.read_text(encoding="utf-8", errors="replace")
                user_prompt = (
                    f"Task ({tf.task_id}): {tf.prompt}\n\n"
                    f"### Current `{target_py.name}`:\n```python\n{original_code[:1600]}\n```\n\n"
                    f"### Unit Tests (`{test_py.name}`):\n```python\n{test_code[:1200]}\n```\n\n"
                    f"### Test Failure Output:\n{failure_output[:900]}\n\n"
                    f"Return the complete fixed Python code for `{target_py.name}` inside a ```python ... ``` block."
                )

                raw_resp = await self.gateway.chat_completion(
                    task_id=tf.task_id,
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_toks,
                    temperature=0.0 if attempt == 0 else 0.2,
                )

                patched_code = extract_code_or_text(raw_resp).strip()
                if not patched_code:
                    continue

                pycache_dir = work_dir / "__pycache__"
                if pycache_dir.exists():
                    shutil.rmtree(pycache_dir, ignore_errors=True)
                target_py.write_text(patched_code + "\n", encoding="utf-8")

                passed, new_err_count, failure_output = await self._run_tests(work_dir, test_py)
                if passed:
                    expected_answer_expr = tf.raw_metadata.get("answer_expr")
                    if isinstance(expected_answer_expr, str) and expected_answer_expr.strip():
                        val = await self._eval_answer_expr(work_dir, target_py.stem, expected_answer_expr)
                        if val is not None:
                            return val
                    return patched_code

                error_history.append(max(1, new_err_count))
                inflight = self.unfeasible_filter.evaluate_tier2_inflight(error_history)
                if not inflight.is_feasible and not qualifying_mode:
                    return None

        return None

    @staticmethod
    def _copy_clean_workspace(src: Path, dst: Path) -> None:
        for item in src.iterdir():
            if is_ignored_fs_entry(item.name):
                continue
            if item.is_file():
                shutil.copy2(item, dst / item.name)

    @staticmethod
    def _locate_target_and_test_files(work_dir: Path) -> tuple[Path | None, Path | None]:
        py_files = [
            p
            for p in sorted(work_dir.glob("*.py"))
            if not is_ignored_fs_entry(p.name) and not p.name.startswith("_")
        ]
        test_files = [p for p in py_files if p.name.startswith("test_") or p.name.endswith("_test.py")]
        impl_files = [p for p in py_files if p not in test_files]
        return (impl_files[0] if impl_files else None), (test_files[0] if test_files else None)

    @staticmethod
    async def _run_tests(work_dir: Path, test_py: Path) -> tuple[bool, int, str]:
        runner_code = (
            "import importlib, sys, traceback\n"
            "sys.dont_write_bytecode = True\n"
            f"sys.path.insert(0, {str(work_dir)!r})\n"
            "importlib.invalidate_caches()\n"
            f"import {test_py.stem} as tm\n"
            "failures = 0\n"
            "traces = []\n"
            "for name in sorted(dir(tm)):\n"
            "    if name.startswith('test_'):\n"
            "        fn = getattr(tm, name)\n"
            "        if callable(fn):\n"
            "            try:\n"
            "                fn()\n"
            "            except Exception:\n"
            "                failures += 1\n"
            "                traces.append(traceback.format_exc())\n"
            "print(f'FAILURES={failures}')\n"
            "if traces:\n"
            "    print('\\n'.join(traces[:3]))\n"
            "sys.exit(0 if failures == 0 else 1)\n"
        )
        runner_file = work_dir / "_run_check.py"
        runner_file.write_text(runner_code, encoding="utf-8")

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-I",
            str(runner_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, 99, "TimeoutError while running tests."

        out = (stdout_b.decode("utf-8", errors="replace") + "\n" + stderr_b.decode("utf-8", errors="replace")).strip()
        if proc.returncode == 0:
            return True, 0, out

        err_count = 1
        for line in out.splitlines():
            if line.startswith("FAILURES="):
                try:
                    err_count = int(line.split("=", 1)[1])
                except ValueError:
                    err_count = 1
        return False, max(1, err_count), out

    @staticmethod
    async def _eval_answer_expr(work_dir: Path, module_name: str, expr: str) -> str | None:
        eval_file = work_dir / "_eval_ans.py"
        eval_file.write_text(
            f"import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, {str(work_dir)!r})\nimport {module_name}\nfrom {module_name} import *\nprint({expr})\n",
            encoding="utf-8",
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-I",
            str(eval_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
        )
        try:
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode == 0:
                return stdout_b.decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return None
