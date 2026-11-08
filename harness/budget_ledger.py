"""Exact competition token budget ledger and atomic answer persistence."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EarlyStopException(RuntimeError):
    """Raised when a task exceeds its allocated p90 effective token budget."""


class GlobalBudgetExhaustedException(RuntimeError):
    """Raised when the global effective token budget or proxy connection is exhausted."""


def compute_effective_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    """Compute competition effective token cost:
    C_eff = T_completion + T_reasoning + 0.1 * T_prompt
    """
    return float(completion_tokens) + float(reasoning_tokens) + 0.1 * float(prompt_tokens)


@dataclass
class TaskBudgetRecord:
    """Per-task token consumption and status tracking."""

    task_id: str
    allocated_p90_budget: float = 2500.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    effective_tokens: float = 0.0
    calls: int = 0
    solved: bool = False
    unfeasible: bool = False
    model_history: list[str] = field(default_factory=list)


class BudgetLedger:
    """Adaptive-greedy token budget tracker with per-task p90 enforcement and atomic persistence."""

    def __init__(
        self,
        global_budget_limit: float | None = None,
        out_file: str | Path | None = None,
    ) -> None:
        self.global_budget_limit = global_budget_limit
        self.out_file = Path(out_file) if out_file else (
            Path(os.environ["OUT"]) if os.environ.get("OUT") else None
        )
        self.start_time = time.monotonic()
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.total_effective_tokens: float = 0.0
        self.proxy_exhausted: bool = False

        self._tasks: dict[str, TaskBudgetRecord] = {}
        self._answers: dict[str, str] = {}
        self._rewards_earned: float = 0.0
        self._lock = asyncio.Lock()

    def register_task(self, task_id: str, allocated_p90_budget: float) -> TaskBudgetRecord:
        record = self._tasks.get(task_id)
        if record is None:
            record = TaskBudgetRecord(
                task_id=task_id,
                allocated_p90_budget=max(50.0, float(allocated_p90_budget)),
            )
            self._tasks[task_id] = record
        else:
            record.allocated_p90_budget = max(50.0, float(allocated_p90_budget))
        return record

    def check_task_can_proceed(self, task_id: str) -> None:
        """Raise if global budget is exhausted or the task exceeded its p90 budget."""
        if self.proxy_exhausted:
            raise GlobalBudgetExhaustedException("Proxy or global token limit reached.")
        if (
            self.global_budget_limit is not None
            and self.total_effective_tokens >= self.global_budget_limit
        ):
            self.proxy_exhausted = True
            raise GlobalBudgetExhaustedException(
                f"Global effective token budget ({self.global_budget_limit:.1f}) exhausted."
            )
        rec = self._tasks.get(task_id)
        if rec is not None and rec.effective_tokens > rec.allocated_p90_budget:
            raise EarlyStopException(
                f"Task {task_id} exceeded allocated p90 budget: "
                f"{rec.effective_tokens:.1f} > {rec.allocated_p90_budget:.1f}"
            )

    def record_llm_call(
        self,
        task_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        enforce_limit: bool = True,
    ) -> float:
        """Record token usage for a single LLM call and return delta C_eff."""
        delta_eff = compute_effective_tokens(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_completion_tokens += int(completion_tokens)
        self.total_reasoning_tokens += int(reasoning_tokens)
        self.total_effective_tokens = round(self.total_effective_tokens + delta_eff, 4)

        rec = self._tasks.setdefault(task_id, TaskBudgetRecord(task_id=task_id))
        rec.prompt_tokens += int(prompt_tokens)
        rec.completion_tokens += int(completion_tokens)
        rec.reasoning_tokens += int(reasoning_tokens)
        rec.effective_tokens = round(rec.effective_tokens + delta_eff, 4)
        rec.calls += 1
        rec.model_history.append(model)

        if (
            self.global_budget_limit is not None
            and self.total_effective_tokens > self.global_budget_limit
        ):
            self.proxy_exhausted = True
            if enforce_limit:
                raise GlobalBudgetExhaustedException(
                    f"Global token budget ({self.global_budget_limit:.1f}) exceeded after call."
                )

        if enforce_limit and rec.effective_tokens > rec.allocated_p90_budget:
            raise EarlyStopException(
                f"Task {task_id} exceeded p90 budget ({rec.effective_tokens:.1f} > {rec.allocated_p90_budget:.1f})."
            )
        return delta_eff

    def get_task_record(self, task_id: str) -> TaskBudgetRecord:
        return self._tasks.setdefault(task_id, TaskBudgetRecord(task_id=task_id))

    @staticmethod
    def _atomic_write_text(target_path: Path, content: str) -> None:
        """Write text atomically using a temporary file in `/tmp` or same directory plus fsync."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target_path.name}.",
                dir=str(target_path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write(content)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_name, target_path)
        except OSError:
            # Fallback to direct write + flush + fsync if directory permissions disallow temp file creation
            with target_path.open("w", encoding="utf-8") as fp:
                fp.write(content)
                fp.flush()
                os.fsync(fp.fileno())

    async def persist_verified_answer(
        self,
        task_dir: Path,
        task_id: str,
        answer: str,
        reward: float = 0.0,
    ) -> Path:
        """Atomically persist `<task_dir>/answer.txt` and synchronously update `$OUT` under lock."""
        clean_answer = str(answer).strip()
        answer_path = task_dir / "answer.txt"

        async with self._lock:
            self._atomic_write_text(answer_path, clean_answer + "\n")
            rec = self.get_task_record(task_id)
            if not rec.solved:
                rec.solved = True
                self._rewards_earned += float(reward)
            self._answers[task_id] = clean_answer

            if self.out_file is not None:
                out_payload = {
                    "updated_at": time.time(),
                    "solved_count": len(self._answers),
                    "total_reward_earned": round(self._rewards_earned, 2),
                    "total_effective_tokens": round(self.total_effective_tokens, 2),
                    "answers": dict(sorted(self._answers.items())),
                }
                self._atomic_write_text(
                    self.out_file,
                    json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n",
                )
        return answer_path

    def summary(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_effective_tokens": round(self.total_effective_tokens, 2),
            "solved_tasks": sum(1 for r in self._tasks.values() if r.solved),
            "unfeasible_tasks": sum(1 for r in self._tasks.values() if r.unfeasible),
            "total_reward_earned": round(self._rewards_earned, 2),
            "elapsed_sec": round(time.monotonic() - self.start_time, 3),
        }
