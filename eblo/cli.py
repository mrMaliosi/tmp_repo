"""CLI entrypoint for EBLO with Rich progress dashboard and optional Typer/argparse support."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel

from eblo.config.schema import EBLOConfig
from eblo.core.domain import TaskSpec
from eblo.core.engine import EvolutionEngine


def load_task_from_module(task_path: str | Path) -> TaskSpec:
    """Dynamically load a `TaskSpec` from a Python file or directory containing `task.py`."""
    path = Path(task_path)
    if path.is_dir():
        candidate = path / "task.py"
        if not candidate.exists():
            raise FileNotFoundError(f"No task.py found inside directory: {path}")
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    spec = importlib.util.spec_from_file_location("eblo_user_task", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_task_spec") and callable(mod.get_task_spec):
        task_spec = mod.get_task_spec()
        if isinstance(task_spec, TaskSpec):
            return task_spec
    if hasattr(mod, "TASK_SPEC") and isinstance(mod.TASK_SPEC, TaskSpec):
        return mod.TASK_SPEC

    raise ValueError(f"Module {path} must define `get_task_spec() -> TaskSpec` or `TASK_SPEC: TaskSpec`.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eblo",
        description="Evolutionary Blazing-fast LLM Orchestrator (EBLO) CLI",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config" / "default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=str(Path(__file__).resolve().parent / "examples" / "sample_task" / "task.py"),
        help="Path to task Python file or directory containing task.py.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override number of evolutionary generations.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Override population size.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generational", "steady_state", "island"],
        default=None,
        help="Override evolution mode.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Run using built-in deterministic mock LLM responder instead of live server.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID for output lineage directory.",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    console = Console()

    config = EBLOConfig.from_yaml(args.config)
    if args.generations is not None:
        config.experiment.generations = args.generations
    if args.pop_size is not None:
        config.experiment.population_size = args.pop_size
    if args.mode is not None:
        config.experiment.evolution_mode = args.mode  # type: ignore[assignment]
    if args.mock_llm:
        config.llm.provider = "mock"

    task_spec = load_task_from_module(args.task)
    engine = EvolutionEngine(config=config, console=console)

    result = await engine.run(
        task=task_spec,
        run_id=args.run_id,
        show_dashboard=True,
    )

    best = result.best_individual
    console.print(
        Panel(
            f"[bold green]Best Score:[/bold green] {best.fitness.total_score:.4f} | "
            f"[bold yellow]Pass Rate:[/bold yellow] {best.fitness.pass_rate * 100:.1f}% | "
            f"[bold magenta]Runtime:[/bold magenta] {best.fitness.execution_time_ms:.3f} ms\n"
            f"[bold cyan]Lineage JSONL:[/bold cyan] {result.lineage_jsonl_path}\n"
            f"[bold cyan]Best Code Saved:[/bold cyan] {result.best_solution_path}",
            title=f"EBLO Run Complete — Task `{task_spec.task_id}`",
            expand=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
