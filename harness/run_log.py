"""Append-only JSONL run logger for post-run diagnostics.

Every LLM call, retry, task outcome, and the final stop reason are recorded so a
failed run can always be explained after the fact (the competition harness must
never die silently).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Best-effort JSONL logger: never raises, degrades to no-op on write errors."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.enabled = False
        self._t0 = time.monotonic()
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fp:
                    fp.write("")
                self.enabled = True
            except OSError:
                self.enabled = False

    def event(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            self.enabled = False
