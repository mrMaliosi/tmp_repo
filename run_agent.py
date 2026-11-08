"""Main container entrypoint (`/app/run_agent.py`) for the Budget-Aware Token Estimation Competition."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from harness.budget_ledger import (
    BudgetLedger,
    EarlyStopException,
    GlobalBudgetExhaustedException,
)
from harness.domain_solvers import (
    CodingDomainSolver,
    DataAnalysisDomainSolver,
    MathDomainSolver,
    SearchDomainSolver,
)
from harness.estimator import TokenAndFeasibilityEstimator
from harness.jiuwen_agent import JiuwenAgentGateway
from harness.run_log import RunLogger
from harness.model_router import ModelRouter
from harness.scheduler import PortfolioKnapsackScheduler, ScheduledTaskItem
from harness.task_scanner import ZeroTokenTaskScanner
from harness.unfeasible_filter import UnfeasibleTaskFilter

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "harness" / "evolved_artifacts" / "best_policy.json"


class TournamentHarnessRunner:
    """Orchestrates up to 5 concurrent agent sessions across the prioritized task queue."""

    def __init__(
        self,
        tasks_dir: str | Path,
        out_path: str | Path | None = None,
        policy_path: str | Path | None = None,
        global_budget_limit: float | None = None,
        max_workers: int = 5,
        global_timeout_sec: float = 3480.0,  # 58 minutes out of 60
        mock_handler: Callable[[str, str, list[dict[str, str]], int], dict[str, Any]] | None = None,
    ) -> None:
        self.tasks_dir = Path(tasks_dir)
        self.out_path = self._resolve_out_path(out_path)
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
        self.max_workers = max(1, min(5, int(max_workers)))
        self.global_timeout_sec = global_timeout_sec

        self.ledger = BudgetLedger(
            global_budget_limit=global_budget_limit,
            out_file=self.out_path,
        )
        self.scanner = ZeroTokenTaskScanner()
        self.estimator = TokenAndFeasibilityEstimator.from_json(self.policy_path)
        self.router = ModelRouter(self.estimator)
        self.gateway = JiuwenAgentGateway(
            ledger=self.ledger,
            mock_handler=mock_handler,
        )
        self.logger = RunLogger(Path(str(self.out_path) + ".log") if self.out_path else None)
        self.gateway.logger = self.logger
        self.stop_reason: str = "queue_completed"

    async def execute_tournament(self) -> dict[str, Any]:
        """Scan all tasks in zero-token mode, schedule by ROI, and run up to 5 concurrent workers."""
        scan_t0 = time.perf_counter()
        scanned_tasks = self.scanner.scan_tasks_dir(self.tasks_dir)
        scan_elapsed_sec = time.perf_counter() - scan_t0

        # Stage 3 qualifying run check: if <= 5 tasks found, enable qualifying mode (100% completion focus)
        qualifying_mode = len(scanned_tasks) <= 5
        unfeasible_filter = UnfeasibleTaskFilter(
            estimator=self.estimator,
            qualifying_mode=qualifying_mode,
        )
        scheduler = PortfolioKnapsackScheduler(
            estimator=self.estimator,
            router=self.router,
            unfeasible_filter=unfeasible_filter,
        )

        active_queue, rejected_items = scheduler.build_schedule(
            scanned_tasks,
            qualifying_mode=qualifying_mode,
        )

        for rej in rejected_items:
            rec = self.ledger.register_task(rej.features.task_id, rej.prediction.p90_tokens)
            rec.unfeasible = True

        sys_prompts = self.estimator.policy.get("system_prompts", {})
        math_solver = MathDomainSolver(
            gateway=self.gateway,
            router=self.router,
            system_prompt=sys_prompts.get("math", ""),
        )
        data_solver = DataAnalysisDomainSolver(
            gateway=self.gateway,
            router=self.router,
            unfeasible_filter=unfeasible_filter,
            system_prompt=sys_prompts.get("data_analysis", ""),
        )
        search_solver = SearchDomainSolver(
            gateway=self.gateway,
            router=self.router,
            system_prompt=sys_prompts.get("search", ""),
        )
        coding_solver = CodingDomainSolver(
            gateway=self.gateway,
            router=self.router,
            unfeasible_filter=unfeasible_filter,
            system_prompt=sys_prompts.get("coding", ""),
        )

        queue: asyncio.Queue[ScheduledTaskItem] = asyncio.Queue()
        for item in active_queue:
            p90_cap = item.prediction.p90_tokens * (2.5 if qualifying_mode else 1.35)
            self.ledger.register_task(item.features.task_id, allocated_p90_budget=p90_cap)
            queue.put_nowait(item)

        worker_semaphore = asyncio.Semaphore(self.max_workers)

        async def _worker() -> None:
            while not queue.empty() and not self.ledger.proxy_exhausted:
                try:
                    sched_item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                tf = sched_item.features
                async with worker_semaphore:
                    try:
                        # Optional Tier-1 cheap probe if suspicious high-reward task
                        if unfeasible_filter.should_run_tier1_probe(
                            tf, sched_item.filter_decision.prob_feasible
                        ):
                            probe_dec = await unfeasible_filter.evaluate_tier1_probe(
                                tf=tf,
                                agent_gateway=self.gateway,
                                workspace_summary=f"files={tf.workspace_files_count}, bytes={tf.workspace_total_bytes}",
                            )
                            if not probe_dec.is_feasible:
                                self.ledger.get_task_record(tf.task_id).unfeasible = True
                                continue

                        if tf.domain == "math":
                            answer = await math_solver.solve(tf, qualifying_mode=qualifying_mode)
                        elif tf.domain == "data_analysis":
                            answer = await data_solver.solve(tf, qualifying_mode=qualifying_mode)
                        elif tf.domain == "search":
                            answer = await search_solver.solve(tf, qualifying_mode=qualifying_mode)
                        elif tf.domain == "coding":
                            answer = await coding_solver.solve(tf, qualifying_mode=qualifying_mode)
                        else:
                            answer = await math_solver.solve(tf, qualifying_mode=qualifying_mode)

                        if answer is not None and str(answer).strip():
                            await self.ledger.persist_verified_answer(
                                task_dir=tf.task_dir,
                                task_id=tf.task_id,
                                answer=str(answer).strip(),
                                reward=tf.reward,
                            )
                            self.logger.event("task_solved", task_id=tf.task_id, reward=tf.reward)
                        else:
                            self.ledger.get_task_record(tf.task_id).unfeasible = True
                            self.logger.event("task_unsolved", task_id=tf.task_id, reason="empty_answer")
                    except EarlyStopException:
                        self.ledger.get_task_record(tf.task_id).unfeasible = True
                        self.logger.event("task_unsolved", task_id=tf.task_id, reason="p90_budget_exceeded")
                    except GlobalBudgetExhaustedException:
                        self.stop_reason = "global_budget_exhausted"
                        self.logger.event("stop", reason=self.stop_reason)
                        break
                    except Exception as exc:
                        # Never die silently: record the failure and move on.
                        self.logger.event(
                            "task_error",
                            task_id=tf.task_id,
                            error=type(exc).__name__,
                            detail=str(exc)[:300],
                        )
                    finally:
                        queue.task_done()

        try:
            workers = [asyncio.create_task(_worker()) for _ in range(self.max_workers)]
            await asyncio.wait_for(asyncio.gather(*workers), timeout=self.global_timeout_sec)
        except asyncio.TimeoutError:
            pass
        finally:
            await self.gateway.aclose()

        if not self.stop_reason and self.ledger.proxy_exhausted:
            self.stop_reason = "proxy_exhausted"
        self.logger.event(
            "run_finished",
            reason=self.stop_reason,
            solved=self.ledger.summary()["solved_tasks"],
            effective_tokens=self.ledger.summary()["total_effective_tokens"],
        )

        summary = self.ledger.summary()
        summary["scanned_tasks_total"] = len(scanned_tasks)
        summary["static_rejected_traps"] = len(rejected_items)
        summary["qualifying_mode"] = qualifying_mode
        summary["scan_elapsed_sec"] = round(scan_elapsed_sec, 4)
        summary["stop_reason"] = self.stop_reason
        return summary


    @staticmethod
    def _resolve_out_path(out_path: str | Path | None) -> Path | None:
        """Resolve the `$OUT` summary file path.

        Resolution order:
        1. Explicit `out_path` argument / `--out` CLI flag.
        2. `$OUT` environment variable (may be a file path OR a directory —
           the organizers' launcher creates it with `mkdir -p "$OUT"`),
           in which case the summary file becomes `<OUT>/summary_out.json`.
        3. Fallback: `<tasks_dir>/summary_out.json` (the organizers' mount point is rw).
        """
        candidates: list[Path] = []
        if out_path:
            candidates.append(Path(out_path))
        elif os.environ.get("OUT"):
            out_env = Path(os.environ["OUT"])
            if out_env.is_dir():
                candidates.append(out_env / "summary_out.json")
            else:
                candidates.append(out_env)
                candidates.append(out_env / "summary_out.json")
        tasks_fallback = Path(os.environ.get("TASKS", "/data/tasks"))
        candidates.append(tasks_fallback / "summary_out.json")

        for cand in candidates:
            try:
                if cand.is_dir():
                    continue
                parent = cand.parent
                parent.mkdir(parents=True, exist_ok=True)
                if os.access(parent, os.W_OK):
                    return cand
            except OSError:
                continue
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_agent.py",
        description="Budget-Aware Token Estimation Tournament Agent Harness (openJiuwen)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=os.environ.get("TASKS", "/data/tasks"),
        help="Path to directory containing task folders (`/data/tasks`).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.environ.get("OUT", None),
        help="Optional summary output file path (`$OUT`).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=str(DEFAULT_POLICY_PATH),
        help="Path to evolved policy JSON artifact.",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Optional explicit global effective token ceiling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Maximum concurrent task sessions (1..5).",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    runner = TournamentHarnessRunner(
        tasks_dir=args.tasks_dir,
        out_path=args.out,
        policy_path=args.policy,
        global_budget_limit=args.budget_limit,
        max_workers=args.workers,
    )
    await runner.execute_tournament()
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
