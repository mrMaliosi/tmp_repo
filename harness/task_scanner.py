"""Fast zero-token scanner for 100 competition tasks (< 1.5 sec total execution)."""

from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_IGNORED_NAMES = {".DS_Store", "__MACOSX", "Thumbs.db"}
_NUMBER_RE = re.compile(r"(?<![a-zA-Z_])-?\d+(?:\.\d+)?(?![a-zA-Z_])")
_FORMULA_RE = re.compile(r"[=<>+\-*/^]{1,2}|\b(?:sum|avg|mean|min|max|mod|sqrt|gcd|lcm)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b(?:person_\d+|company_\d+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_COLUMN_REF_RE = re.compile(r"`([a-zA-Z0-9_]+)`|'([a-zA-Z0-9_]+)'|\"([a-zA-Z0-9_]+)\"")


def is_ignored_fs_entry(name: str) -> bool:
    """Return True for macOS metadata junk (`.DS_Store`, `__MACOSX`, `._*`) and hidden files."""
    if not name or name in _IGNORED_NAMES or name.startswith("._") or name.startswith("."):
        return True
    return False


@dataclass
class TaskFeatures:
    """Zero-token static feature vector and workspace metadata for a single competition task."""

    task_id: str
    task_dir: Path
    domain: str  # coding | math | data_analysis | search
    reward: float
    prompt: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    # Text features
    prompt_char_len: int = 0
    prompt_word_count: int = 0
    number_count: int = 0
    formula_count: int = 0
    question_count: int = 0
    constraint_markers: int = 0

    # Workspace general features
    workspace_files_count: int = 0
    workspace_total_bytes: int = 0

    # Coding domain features
    ast_cyclomatic_complexity: int = 0
    test_assert_count: int = 0
    has_syntax_error_in_tests: bool = False
    contradictory_asserts: bool = False

    # Data analysis domain features
    csv_row_count: int = 0
    csv_col_count: int = 0
    csv_nan_or_corrupt_rows: int = 0
    csv_columns: list[str] = field(default_factory=list)
    missing_requested_columns: list[str] = field(default_factory=list)

    # Search / KB domain features
    kb_doc_count: int = 0
    kb_entity_count: int = 0
    kb_graph_edges: int = 0
    missing_query_entities: list[str] = field(default_factory=list)
    kb_numeric_contradictions: int = 0

    # Math domain features
    math_paradox_markers: int = 0

    def to_feature_vector(self) -> list[float]:
        """Return numerical feature vector x_i for classical ML token & feasibility estimation."""
        domain_one_hot = [
            1.0 if self.domain == "math" else 0.0,
            1.0 if self.domain == "coding" else 0.0,
            1.0 if self.domain == "data_analysis" else 0.0,
            1.0 if self.domain == "search" else 0.0,
        ]
        return domain_one_hot + [
            float(self.reward),
            float(self.prompt_char_len),
            float(self.prompt_word_count),
            float(self.number_count),
            float(self.formula_count),
            float(self.question_count),
            float(self.constraint_markers),
            float(self.workspace_files_count),
            float(self.workspace_total_bytes) / 1024.0,
            float(self.ast_cyclomatic_complexity),
            float(self.test_assert_count),
            float(self.csv_row_count),
            float(self.csv_col_count),
            float(self.csv_nan_or_corrupt_rows),
            float(self.kb_doc_count),
            float(self.kb_graph_edges),
        ]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_dir"] = str(self.task_dir)
        return d


class ZeroTokenTaskScanner:
    """Scans all task directories in `--tasks-dir` without invoking any LLM calls."""

    def scan_tasks_dir(self, tasks_dir: str | Path) -> list[TaskFeatures]:
        root = Path(tasks_dir)
        if not root.exists():
            return []

        features_list: list[TaskFeatures] = []
        entries = sorted(root.iterdir(), key=lambda p: p.name)
        for entry in entries:
            if is_ignored_fs_entry(entry.name) or not entry.is_dir():
                continue
            task_json_path = entry / "task.json"
            if not task_json_path.exists():
                continue
            tf = self.scan_single_task(entry, task_json_path)
            if tf is not None:
                features_list.append(tf)
        return features_list

    def scan_single_task(self, task_dir: Path, task_json_path: Path) -> TaskFeatures | None:
        try:
            raw_data = json.loads(task_json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

        task_id = str(raw_data.get("task_id") or raw_data.get("id") or task_dir.name)
        prompt = str(
            raw_data.get("question")
            or raw_data.get("description")
            or raw_data.get("prompt")
            or raw_data.get("task")
            or ""
        )
        reward = float(raw_data.get("reward", 10.0))
        domain = self._infer_domain(task_id, raw_data, task_dir)

        words = prompt.split()
        numbers = _NUMBER_RE.findall(prompt)
        formulas = _FORMULA_RE.findall(prompt)
        questions = prompt.count("?")
        constraints = len(
            re.findall(r"\b(?:must|strictly|exact|only|cannot|impossible|never|constraint)\b", prompt, re.I)
        )

        tf = TaskFeatures(
            task_id=task_id,
            task_dir=task_dir,
            domain=domain,
            reward=reward,
            prompt=prompt,
            raw_metadata=raw_data,
            prompt_char_len=len(prompt),
            prompt_word_count=len(words),
            number_count=len(numbers),
            formula_count=len(formulas),
            question_count=questions,
            constraint_markers=constraints,
        )

        workspace_dir = task_dir / "workspace"
        if workspace_dir.exists() and workspace_dir.is_dir():
            self._scan_workspace(tf, workspace_dir)

        if domain == "math":
            self._scan_math_features(tf)

        return tf

    @staticmethod
    def _infer_domain(task_id: str, raw_data: dict[str, Any], task_dir: Path) -> str:
        explicit = str(raw_data.get("domain") or raw_data.get("category") or "").lower().strip()
        if explicit in {"coding", "math", "data_analysis", "search"}:
            return explicit
        lower_id = task_id.lower()
        for prefix in ("coding", "math", "data_analysis", "search"):
            if lower_id.startswith(prefix):
                return prefix
        ws = task_dir / "workspace"
        if (ws / "kb").exists():
            return "search"
        if list(ws.glob("*.csv")) if ws.exists() else []:
            return "data_analysis"
        if list(ws.glob("*.py")) if ws.exists() else []:
            return "coding"
        return "math"

    def _scan_workspace(self, tf: TaskFeatures, workspace_dir: Path) -> None:
        file_count = 0
        total_bytes = 0
        for root_str, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if not is_ignored_fs_entry(d)]
            for fname in files:
                if is_ignored_fs_entry(fname):
                    continue
                fpath = Path(root_str) / fname
                file_count += 1
                try:
                    total_bytes += fpath.stat().st_size
                except OSError:
                    pass

        tf.workspace_files_count = file_count
        tf.workspace_total_bytes = total_bytes

        if tf.domain == "coding":
            self._scan_coding_workspace(tf, workspace_dir)
        elif tf.domain == "data_analysis":
            self._scan_data_workspace(tf, workspace_dir)
        elif tf.domain == "search":
            self._scan_search_workspace(tf, workspace_dir)

    def _scan_coding_workspace(self, tf: TaskFeatures, workspace_dir: Path) -> None:
        complexity = 0
        assert_count = 0
        syntax_err = False
        contradictory = False

        for py_file in sorted(workspace_dir.glob("*.py")):
            if is_ignored_fs_entry(py_file.name):
                continue
            source = py_file.read_text(encoding="utf-8", errors="replace")
            is_test = py_file.name.startswith("test_") or py_file.name.endswith("_test.py")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                if is_test:
                    syntax_err = True
                continue

            if not is_test:
                for node in ast.walk(tree):
                    if isinstance(
                        node,
                        (ast.If, ast.For, ast.While, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.BoolOp),
                    ):
                        complexity += 1
            else:
                # Count asserts and check for direct contradictory assertions f(x) == a and f(x) == b
                call_expectations: dict[str, set[str]] = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        assert_count += 1
                        if (
                            isinstance(node.test, ast.Compare)
                            and len(node.test.ops) == 1
                            and isinstance(node.test.ops[0], ast.Eq)
                            and isinstance(node.test.left, ast.Call)
                        ):
                            call_repr = ast.unparse(node.test.left)
                            rhs_repr = ast.unparse(node.test.comparators[0])
                            seen = call_expectations.setdefault(call_repr, set())
                            seen.add(rhs_repr)
                            if len(seen) > 1:
                                contradictory = True
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr.startswith("assert")
                    ):
                        assert_count += 1
                        if node.func.attr == "assertEqual" and len(node.args) >= 2:
                            call_repr = ast.unparse(node.args[0])
                            rhs_repr = ast.unparse(node.args[1])
                            seen = call_expectations.setdefault(call_repr, set())
                            seen.add(rhs_repr)
                            if len(seen) > 1:
                                contradictory = True

        tf.ast_cyclomatic_complexity = complexity
        tf.test_assert_count = assert_count
        tf.has_syntax_error_in_tests = syntax_err
        tf.contradictory_asserts = contradictory

    def _scan_data_workspace(self, tf: TaskFeatures, workspace_dir: Path) -> None:
        csv_files = [p for p in sorted(workspace_dir.glob("*.csv")) if not is_ignored_fs_entry(p.name)]
        if not csv_files:
            tf.missing_requested_columns = ["__NO_CSV_FILE__"]
            return

        primary_csv = csv_files[0]
        rows = 0
        nan_rows = 0
        headers: list[str] = []
        try:
            with primary_csv.open("r", encoding="utf-8", errors="replace", newline="") as fp:
                reader = csv.reader(fp)
                try:
                    raw_headers = next(reader)
                    headers = [h.strip() for h in raw_headers if h.strip()]
                except StopIteration:
                    headers = []
                expected_cols = len(headers)
                for row in reader:
                    rows += 1
                    if len(row) != expected_cols or any(
                        cell.strip() in {"", "NaN", "nan", "null", "NULL", "None"} for cell in row
                    ):
                        nan_rows += 1
        except Exception:
            nan_rows += 1

        tf.csv_row_count = rows
        tf.csv_col_count = len(headers)
        tf.csv_nan_or_corrupt_rows = nan_rows
        tf.csv_columns = headers

        # Check if prompt explicitly references backticked/quoted column names that do not exist
        header_lower_set = {h.lower() for h in headers}
        missing_cols: list[str] = []
        for match in _COLUMN_REF_RE.findall(tf.prompt):
            token = next((m for m in match if m), "")
            if (
                token
                and "_" in token
                and not token.endswith(".csv")
                and token.lower() not in header_lower_set
            ):
                missing_cols.append(token)
        tf.missing_requested_columns = missing_cols

    def _scan_search_workspace(self, tf: TaskFeatures, workspace_dir: Path) -> None:
        kb_dir = workspace_dir / "kb"
        if not kb_dir.exists():
            kb_dir = workspace_dir
        md_files = [p for p in sorted(kb_dir.glob("*.md")) if not is_ignored_fs_entry(p.name)]
        tf.kb_doc_count = len(md_files)

        entities_in_kb: set[str] = set()
        edge_count = 0
        fact_values: dict[tuple[str, str], set[str]] = {}
        contradictions = 0

        kv_pattern = re.compile(
            r"\b(person_\d+|company_\d+)\b[^\n:]{0,30}\b(revenue|salary|age|founded|employees|budget|score)\b[^\d\n-]{0,15}(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            stem = md_file.stem.lower()
            entities_in_kb.add(stem)
            found_ents = {e.lower() for e in _ENTITY_RE.findall(text)}
            entities_in_kb.update(found_ents)
            edge_count += max(0, len(found_ents) - 1)

            for ent, attr, val in kv_pattern.findall(text):
                key = (ent.lower(), attr.lower())
                bucket = fact_values.setdefault(key, set())
                bucket.add(val)
                if len(bucket) > 1:
                    contradictions += 1

        tf.kb_entity_count = len(entities_in_kb)
        tf.kb_graph_edges = edge_count
        tf.kb_numeric_contradictions = contradictions

        query_ents = {
            e.lower()
            for e in re.findall(r"\b(?:person_\d+|company_\d+|project_\d+)\b", tf.prompt, re.IGNORECASE)
        }
        missing = sorted(e for e in query_ents if e not in entities_in_kb)
        tf.missing_query_entities = missing

    @staticmethod
    def _scan_math_features(tf: TaskFeatures) -> None:
        lower = tf.prompt.lower()
        paradox_count = 0
        # Detect classic unsatisfiable constraints in math prompts
        if "unsatisfiable" in lower or "no solution exists" in lower or "impossible invariant" in lower:
            paradox_count += 1
        # Check for simple contradictory equality pairs like `x + y = 10` and `x + y = 25`
        eq_matches = re.findall(r"([a-z]\s*\+\s*[a-z])\s*=\s*(-?\d+)", lower)
        lhs_map: dict[str, set[str]] = {}
        for lhs, rhs in eq_matches:
            norm_lhs = re.sub(r"\s+", "", lhs)
            vals = lhs_map.setdefault(norm_lhs, set())
            vals.add(rhs)
            if len(vals) > 1:
                paradox_count += 1
        # Check parity traps (e.g., sum of two even integers equals an odd integer)
        if re.search(r"even\s+integers.*sum\s+is\s+\d*[13579]\b", lower):
            paradox_count += 1
        tf.math_paradox_markers = paradox_count
