"""Module 2: Cascade Phenotype Validator with AST Cache, Subprocess Sandbox, and In-Process Runner."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
import time
import tracemalloc
import traceback
from pathlib import Path
from typing import Any

from eblo.config.schema import ValidatorConfig
from eblo.core.domain import FitnessReport, Phenotype, TaskSpec, TestCase
from eblo.storage.lineage import ASTFitnessCache

_SUBPROCESS_HARNESS_CODE = '''import json
import sys
import time
import traceback
import tracemalloc

def _run_suite():
    payload_path = sys.argv[1]
    with open(payload_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    code = payload["code"]
    entrypoint = payload["entrypoint"]
    tests = payload["tests"]

    namespace = {"__name__": "__eblo_candidate__"}
    try:
        compiled = compile(code, "<phenotype>", "exec")
        exec(compiled, namespace)
    except Exception:
        print(json.dumps({
            "passed": 0,
            "total": len(tests),
            "execution_time_ms": 0.0,
            "memory_peak_kb": 0.0,
            "error_trace": traceback.format_exc(),
        }))
        return

    if entrypoint not in namespace or not callable(namespace[entrypoint]):
        print(json.dumps({
            "passed": 0,
            "total": len(tests),
            "execution_time_ms": 0.0,
            "memory_peak_kb": 0.0,
            "error_trace": f"Entrypoint '{entrypoint}' not found or not callable in phenotype.",
        }))
        return

    fn = namespace[entrypoint]
    passed = 0
    errors = []

    tracemalloc.start()
    total_ns = 0

    for tc in tests:
        name = tc["name"]
        args = tc.get("args", [])
        kwargs = tc.get("kwargs", {})
        expected = tc.get("expected", None)
        validator_code = tc.get("validator_code", None)

        try:
            t0 = time.perf_counter_ns()
            actual = fn(*args, **kwargs)
            t1 = time.perf_counter_ns()
            total_ns += (t1 - t0)

            if validator_code:
                val_ns = {"actual": actual, "expected": expected, "args": args, "kwargs": kwargs}
                exec(validator_code, val_ns)
                is_ok = bool(val_ns.get("is_valid", True))
                if not is_ok:
                    reason = val_ns.get("error_message", f"Custom validator failed on {name}: got {actual!r}")
                    errors.append(f"[{name}] {reason}")
                else:
                    passed += 1
            else:
                if actual == expected:
                    passed += 1
                else:
                    errors.append(f"[{name}] AssertionError: expected {expected!r}, got {actual!r}")
        except Exception:
            errors.append(f"[{name}] Exception:\\n{traceback.format_exc()}")

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(json.dumps({
        "passed": passed,
        "total": len(tests),
        "execution_time_ms": total_ns / 1_000_000.0,
        "memory_peak_kb": peak_bytes / 1024.0,
        "error_trace": "\\n".join(errors[:3]) if errors else None,
    }))

if __name__ == "__main__":
    _run_suite()
'''


class SubprocessSandboxRunner:
    """Executes test suites in an isolated Python child process with strict timeout enforcement."""

    async def run_tests(
        self,
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_ms: int,
    ) -> dict[str, Any]:
        if not tests:
            return {
                "passed": 0,
                "total": 0,
                "execution_time_ms": 0.0,
                "memory_peak_kb": 0.0,
                "error_trace": None,
            }

        with tempfile.TemporaryDirectory(prefix="eblo_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            harness_file = tmp_path / "harness.py"
            payload_file = tmp_path / "payload.json"

            harness_file.write_text(_SUBPROCESS_HARNESS_CODE, encoding="utf-8")
            payload_data = {
                "code": code,
                "entrypoint": entrypoint,
                "tests": [t.to_dict() for t in tests],
            }
            payload_file.write_text(json.dumps(payload_data), encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(harness_file),
                str(payload_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
            )

            timeout_sec = max(0.05, timeout_ms / 1000.0)
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return {
                    "passed": 0,
                    "total": len(tests),
                    "execution_time_ms": float(timeout_ms),
                    "memory_peak_kb": 0.0,
                    "error_trace": f"TimeoutError: Execution exceeded {timeout_ms} ms limit.",
                }

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0 or not stdout_text:
                return {
                    "passed": 0,
                    "total": len(tests),
                    "execution_time_ms": float(timeout_ms),
                    "memory_peak_kb": 0.0,
                    "error_trace": stderr_text or f"Subprocess exited with code {proc.returncode}",
                }

            last_line = stdout_text.splitlines()[-1]
            try:
                return json.loads(last_line)
            except json.JSONDecodeError:
                return {
                    "passed": 0,
                    "total": len(tests),
                    "execution_time_ms": float(timeout_ms),
                    "memory_peak_kb": 0.0,
                    "error_trace": f"Invalid runner output: {stdout_text[:300]} | stderr: {stderr_text[:300]}",
                }


class InProcessFastRunner:
    """Alternative in-process test runner strategy for trusted low-latency benchmarking."""

    async def run_tests(
        self,
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_ms: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync_run, code, entrypoint, tests, timeout_ms)

    @staticmethod
    def _sync_run(
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_ms: int,
    ) -> dict[str, Any]:
        namespace: dict[str, Any] = {"__name__": "__eblo_inproc__"}
        try:
            compiled = compile(code, "<phenotype>", "exec")
            exec(compiled, namespace)
        except Exception:
            return {
                "passed": 0,
                "total": len(tests),
                "execution_time_ms": 0.0,
                "memory_peak_kb": 0.0,
                "error_trace": traceback.format_exc(),
            }

        fn = namespace.get(entrypoint)
        if not callable(fn):
            return {
                "passed": 0,
                "total": len(tests),
                "execution_time_ms": 0.0,
                "memory_peak_kb": 0.0,
                "error_trace": f"Entrypoint '{entrypoint}' not found or not callable in phenotype.",
            }

        passed = 0
        errors: list[str] = []
        total_ns = 0
        limit_ns = int(timeout_ms * 1_000_000)

        tracemalloc.start()
        for tc in tests:
            try:
                t0 = time.perf_counter_ns()
                actual = fn(*tc.args, **tc.kwargs)
                t1 = time.perf_counter_ns()
                total_ns += t1 - t0
                if total_ns > limit_ns:
                    errors.append(f"[{tc.name}] TimeoutError: cumulative runtime exceeded {timeout_ms} ms")
                    break
                if tc.validator_code:
                    val_ns: dict[str, Any] = {
                        "actual": actual,
                        "expected": tc.expected,
                        "args": tc.args,
                        "kwargs": tc.kwargs,
                    }
                    exec(tc.validator_code, val_ns)
                    if bool(val_ns.get("is_valid", True)):
                        passed += 1
                    else:
                        errors.append(
                            f"[{tc.name}] {val_ns.get('error_message', f'Validation failed: {actual!r}')}"
                        )
                else:
                    if actual == tc.expected:
                        passed += 1
                    else:
                        errors.append(f"[{tc.name}] AssertionError: expected {tc.expected!r}, got {actual!r}")
            except Exception:
                errors.append(f"[{tc.name}] Exception:\n{traceback.format_exc()}")

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "passed": passed,
            "total": len(tests),
            "execution_time_ms": total_ns / 1_000_000.0,
            "memory_peak_kb": peak_bytes / 1024.0,
            "error_trace": "\n".join(errors[:3]) if errors else None,
        }


class CascadeValidator:
    """Four-stage cascade validator with AST hash lookup, syntax gate, smoke gate, and full benchmark."""

    def __init__(self, config: ValidatorConfig, cache: ASTFitnessCache | None = None) -> None:
        self.config = config
        self.cache = cache or ASTFitnessCache(
            backend=config.cache_backend,
            db_path=config.cache_db_path,
        )
        if config.runner == "in_process_fast":
            self.runner: SubprocessSandboxRunner | InProcessFastRunner = InProcessFastRunner()
        else:
            self.runner = SubprocessSandboxRunner()

    def compute_total_score(self, pass_rate: float, execution_time_ms: float) -> float:
        """Calculate weighted scalar fitness score:
        Score = w_pass * pass_rate + w_speed * pass_rate * (1 / (1 + execution_time_ms / T_ref))
        """
        if pass_rate <= 0.0:
            return 0.0
        w_pass = self.config.weights.pass_rate
        w_speed = self.config.weights.speed
        t_ref = self.config.reference_time_ms
        speed_factor = 1.0 / (1.0 + max(0.0, execution_time_ms) / t_ref)
        score = (w_pass * pass_rate) + (w_speed * pass_rate * speed_factor)
        return round(score, 6)

    async def evaluate(self, phenotype: Phenotype, task: TaskSpec) -> FitnessReport:
        """Execute 4-stage cascade evaluation with early exit."""
        # Stage 0: AST Lookup
        cached_report = self.cache.get(task.task_id, phenotype.ast_hash)
        if cached_report is not None:
            return cached_report

        code = phenotype.code.strip()
        if not code:
            empty_report = FitnessReport(
                is_syntax_valid=False,
                smoke_passed=False,
                pass_rate=0.0,
                execution_time_ms=0.0,
                memory_peak_kb=0.0,
                total_score=0.0,
                error_trace="SyntaxError: Empty code generated by phenotype.",
                cache_hit=False,
            )
            self.cache.put(task.task_id, phenotype.ast_hash, empty_report)
            return empty_report

        # Stage 1: Syntax & Static Gate
        try:
            tree = ast.parse(code)
            compile(tree, filename="<phenotype>", mode="exec")
        except SyntaxError as exc:
            err_msg = f"SyntaxError at line {exc.lineno}, offset {exc.offset}: {exc.msg}\n{exc.text or ''}"
            syntax_report = FitnessReport(
                is_syntax_valid=False,
                smoke_passed=False,
                pass_rate=0.0,
                execution_time_ms=0.0,
                memory_peak_kb=0.0,
                total_score=0.0,
                error_trace=err_msg.strip(),
                cache_hit=False,
            )
            self.cache.put(task.task_id, phenotype.ast_hash, syntax_report)
            return syntax_report

        # Stage 2: Smoke Gate
        smoke_res = await self.runner.run_tests(
            code=code,
            entrypoint=task.entrypoint,
            tests=task.smoke_tests,
            timeout_ms=self.config.smoke_timeout_ms,
        )
        smoke_total = max(1, int(smoke_res.get("total", len(task.smoke_tests))))
        smoke_passed_count = int(smoke_res.get("passed", 0))
        smoke_passed = smoke_passed_count == smoke_total and smoke_res.get("error_trace") is None

        if not smoke_passed:
            # Partial smoke score scaled down so failed smoke candidates still rank by partial progress
            smoke_ratio = smoke_passed_count / smoke_total
            partial_pass_rate = round(0.25 * smoke_ratio, 4)
            exec_ms = float(smoke_res.get("execution_time_ms", 0.0))
            mem_kb = float(smoke_res.get("memory_peak_kb", 0.0))
            smoke_fail_report = FitnessReport(
                is_syntax_valid=True,
                smoke_passed=False,
                pass_rate=partial_pass_rate,
                execution_time_ms=round(exec_ms, 4),
                memory_peak_kb=round(mem_kb, 3),
                total_score=self.compute_total_score(partial_pass_rate, exec_ms),
                error_trace=smoke_res.get("error_trace") or "Smoke gate failed.",
                cache_hit=False,
            )
            self.cache.put(task.task_id, phenotype.ast_hash, smoke_fail_report)
            return smoke_fail_report

        # Stage 3: Full Benchmark
        bench_res = await self.runner.run_tests(
            code=code,
            entrypoint=task.entrypoint,
            tests=task.benchmark_tests,
            timeout_ms=self.config.benchmark_timeout_ms,
        )
        bench_total = max(1, int(bench_res.get("total", len(task.benchmark_tests))))
        bench_passed = int(bench_res.get("passed", 0))
        pass_rate = round(bench_passed / bench_total, 4)
        exec_ms = round(float(bench_res.get("execution_time_ms", 0.0)), 4)
        mem_kb = round(float(bench_res.get("memory_peak_kb", 0.0)), 3)
        total_score = self.compute_total_score(pass_rate, exec_ms)

        full_report = FitnessReport(
            is_syntax_valid=True,
            smoke_passed=True,
            pass_rate=pass_rate,
            execution_time_ms=exec_ms,
            memory_peak_kb=mem_kb,
            total_score=total_score,
            error_trace=bench_res.get("error_trace"),
            cache_hit=False,
        )
        self.cache.put(task.task_id, phenotype.ast_hash, full_report)
        return full_report
