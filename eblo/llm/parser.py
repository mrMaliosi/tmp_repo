"""LLM output parser for extracting <think> reasoning blocks, executable code, and AST hashes."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

from eblo.core.domain import Phenotype

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py|[a-zA-Z0-9_+-]*)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class _DocstringStripper(ast.NodeTransformer):
    """AST transformer that removes module, class, and function docstrings for canonical hashing."""

    def _strip_docstring(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            remaining = body[1:]
            return remaining if remaining else [ast.Pass()]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node


def compute_normalized_ast_hash(code: str) -> str:
    """Compute SHA-256 hash of AST-normalized Python code (comments and docstrings stripped).

    If code fails to parse as valid Python syntax, falls back to hashing stripped source text
    prefixed with `raw:` so invalid syntax still gets a deterministic hash.
    """
    cleaned = code.strip()
    if not cleaned:
        return hashlib.sha256(b"empty").hexdigest()
    try:
        tree = ast.parse(cleaned)
        tree = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(tree)
        canonical_source = ast.unparse(tree).strip()
        return hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return hashlib.sha256(f"raw:{cleaned}".encode("utf-8")).hexdigest()


def extract_thinking_and_code(raw_response: str) -> tuple[str | None, str]:
    """Separate `<think>...</think>` blocks from executable source code."""
    if not raw_response:
        return None, ""

    think_matches = [m.strip() for m in _THINK_PATTERN.findall(raw_response) if m.strip()]
    remainder = _THINK_PATTERN.sub("", raw_response).strip()

    if not think_matches and "<think>" in remainder.lower():
        unclosed_match = _UNCLOSED_THINK_PATTERN.search(remainder)
        if unclosed_match:
            think_matches.append(unclosed_match.group(1).strip())
            remainder = _UNCLOSED_THINK_PATTERN.sub("", remainder).strip()

    reasoning_trace = "\n\n".join(think_matches) if think_matches else None

    code_matches = _CODE_FENCE_PATTERN.findall(remainder)
    if code_matches:
        selected_code = max((c.strip() for c in code_matches if c.strip()), key=len, default="")
    else:
        selected_code = remainder.strip()

    return reasoning_trace, selected_code


def _extract_without_think(raw_response: str) -> tuple[str | None, str]:
    think_matches = [m.strip() for m in _THINK_PATTERN.findall(raw_response) if m.strip()]
    remainder = _THINK_PATTERN.sub("", raw_response).strip()
    return ("\n\n".join(think_matches) if think_matches else None), remainder


def extract_json_payload(raw_response: str) -> Any:
    """Extract and parse JSON object or array after stripping `<think>` blocks and markdown fences."""
    _, cleaned = extract_thinking_and_code(raw_response)
    text = cleaned.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    _, without_think = _extract_without_think(raw_response)
    for candidate in (text, without_think):
        for open_char, close_char in (("[", "]"), ("{", "}")):
            start = candidate.find(open_char)
            end = candidate.rfind(close_char)
            if start != -1 and end != -1 and end > start:
                snippet = candidate[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Failed to parse JSON payload from LLM response: {raw_response[:200]!r}")


def parse_phenotype(
    chromosome_id: str,
    raw_response: str,
    generation_latency_ms: float,
) -> Phenotype:
    """Build an immutable Phenotype from an LLM completion response."""
    reasoning_trace, code = extract_thinking_and_code(raw_response)
    ast_hash = compute_normalized_ast_hash(code)
    return Phenotype(
        chromosome_id=chromosome_id,
        code=code,
        ast_hash=ast_hash,
        reasoning_trace=reasoning_trace,
        generation_latency_ms=round(generation_latency_ms, 3),
    )
