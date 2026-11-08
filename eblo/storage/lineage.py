"""Persistent lineage storage (JSONL + SQLite) and AST fitness cache for EBLO."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from eblo.core.domain import FitnessReport, Individual, TaskSpec


class ASTFitnessCache:
    """Two-strategy AST hash cache: SQLite persistent store and fast in-memory dictionary store."""

    def __init__(self, backend: str = "sqlite", db_path: str | Path = "runs/ast_cache.sqlite") -> None:
        self.backend = backend
        self.db_path = Path(db_path)
        self._memory_store: dict[str, FitnessReport] = {}
        self.hits: int = 0
        self.misses: int = 0

        if self.backend == "sqlite":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ast_fitness_cache (
                    task_id TEXT NOT NULL,
                    ast_hash TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (task_id, ast_hash)
                )
                """
            )
            conn.commit()

    def get(self, task_id: str, ast_hash: str) -> FitnessReport | None:
        key = f"{task_id}:{ast_hash}"
        if key in self._memory_store:
            self.hits += 1
            cached = self._memory_store[key]
            return FitnessReport(
                is_syntax_valid=cached.is_syntax_valid,
                smoke_passed=cached.smoke_passed,
                pass_rate=cached.pass_rate,
                execution_time_ms=cached.execution_time_ms,
                memory_peak_kb=cached.memory_peak_kb,
                total_score=cached.total_score,
                error_trace=cached.error_trace,
                cache_hit=True,
            )

        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT report_json FROM ast_fitness_cache WHERE task_id = ? AND ast_hash = ?",
                    (task_id, ast_hash),
                ).fetchone()
            if row is not None:
                data = json.loads(row[0])
                data["cache_hit"] = True
                report = FitnessReport(**data)
                self._memory_store[key] = report
                self.hits += 1
                return report

        self.misses += 1
        return None

    def put(self, task_id: str, ast_hash: str, report: FitnessReport) -> None:
        key = f"{task_id}:{ast_hash}"
        stored_report = FitnessReport(
            is_syntax_valid=report.is_syntax_valid,
            smoke_passed=report.smoke_passed,
            pass_rate=report.pass_rate,
            execution_time_ms=report.execution_time_ms,
            memory_peak_kb=report.memory_peak_kb,
            total_score=report.total_score,
            error_trace=report.error_trace,
            cache_hit=False,
        )
        self._memory_store[key] = stored_report
        if self.backend == "sqlite":
            payload = json.dumps(stored_report.to_dict(), ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ast_fitness_cache (task_id, ast_hash, report_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, ast_hash, payload, time.time()),
                )
                conn.commit()


class LineageRecorder:
    """Records full evolutionary genealogy and telemetry into JSONL and SQLite tables."""

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "lineage.jsonl"
        self.sqlite_path = self.run_dir / "lineage.sqlite"
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS individuals (
                    chromosome_id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    island_id INTEGER NOT NULL,
                    parent_ids_json TEXT NOT NULL,
                    mutation_history_json TEXT NOT NULL,
                    ast_hash TEXT NOT NULL,
                    pass_rate REAL NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    memory_peak_kb REAL NOT NULL,
                    total_score REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def record_generation(
        self,
        generation: int,
        individuals: list[Individual],
        active_mutation: str,
        cache_hits: int,
    ) -> None:
        """Append all individuals from a generation to `lineage.jsonl` and `lineage.sqlite`."""
        now = time.time()
        with self.jsonl_path.open("a", encoding="utf-8") as fp, sqlite3.connect(self.sqlite_path) as conn:
            for ind in individuals:
                record: dict[str, Any] = {
                    "timestamp": now,
                    "generation": generation,
                    "active_mutation": active_mutation,
                    "cumulative_cache_hits": cache_hits,
                    **ind.to_dict(),
                }
                line = json.dumps(record, ensure_ascii=False)
                fp.write(line + "\n")

                conn.execute(
                    """
                    INSERT OR REPLACE INTO individuals (
                        chromosome_id, generation, island_id, parent_ids_json,
                        mutation_history_json, ast_hash, pass_rate, execution_time_ms,
                        memory_peak_kb, total_score, cache_hit, payload_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ind.chromosome.id,
                        generation,
                        ind.island_id,
                        json.dumps(list(ind.chromosome.parent_ids)),
                        json.dumps(list(ind.chromosome.mutation_history)),
                        ind.phenotype.ast_hash,
                        ind.fitness.pass_rate,
                        ind.fitness.execution_time_ms,
                        ind.fitness.memory_peak_kb,
                        ind.fitness.total_score,
                        1 if ind.fitness.cache_hit else 0,
                        line,
                        now,
                    ),
                )
            conn.commit()

    def save_best_solution(self, best: Individual, task: TaskSpec) -> Path:
        """Save the highest-scoring phenotype code and metadata to the run directory."""
        best_code_path = self.run_dir / f"best_{task.task_id}.py"
        header = (
            f"# Task: {task.task_id}\n"
            f"# Chromosome ID: {best.chromosome.id}\n"
            f"# Generation: {best.chromosome.generation}\n"
            f"# Total Score: {best.fitness.total_score:.6f}\n"
            f"# Pass Rate: {best.fitness.pass_rate:.4f}\n"
            f"# Execution Time (ms): {best.fitness.execution_time_ms:.4f}\n"
            f"# Strategy: {best.chromosome.algorithmic_strategy}\n\n"
        )
        best_code_path.write_text(header + best.phenotype.code + "\n", encoding="utf-8")
        summary_path = self.run_dir / "best_summary.json"
        summary_path.write_text(
            json.dumps(best.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return best_code_path
