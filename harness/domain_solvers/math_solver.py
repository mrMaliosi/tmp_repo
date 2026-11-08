"""Domain solver for `math_XXXX` tasks using local verification in `/tmp`."""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from harness.jiuwen_agent import JiuwenAgentGateway, extract_code_or_text
from harness.model_router import ModelRouter
from harness.task_scanner import TaskFeatures


class MathDomainSolver:
    """Solves mathematical tasks by executing deterministic or LLM-generated verification scripts in `/tmp`."""

    def __init__(
        self,
        gateway: JiuwenAgentGateway,
        router: ModelRouter,
        system_prompt: str,
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.system_prompt = system_prompt

    async def solve(self, tf: TaskFeatures, qualifying_mode: bool = False) -> str | None:
        # Step 1: Try zero-token local symbolic/arithmetic evaluation if explicit expression exists
        direct_expr = tf.raw_metadata.get("expression")
        if isinstance(direct_expr, str) and re.fullmatch(r"[0-9+\-*/()%.\s]+", direct_expr):
            try:
                val = eval(direct_expr, {"__builtins__": {}}, {})
                return str(val)
            except Exception:
                pass

        # Step 2: Model-assisted Python solver generation + verification in /tmp
        max_attempts = 2
        last_error = ""
        for attempt in range(max_attempts):
            model = self.router.select_model(tf, attempt=attempt, qualifying_mode=qualifying_mode)
            is_reason = not model.endswith("-noreason")
            max_toks = self.router.max_tokens_for_domain("math", is_reasoning_model=is_reason)

            user_msg = f"Problem ({tf.task_id}):\n{tf.prompt}\n"
            if last_error:
                user_msg += f"\nPrevious script failed with error:\n{last_error}\nFix the script and print ONLY the final answer."

            raw_resp = await self.gateway.chat_completion(
                task_id=tf.task_id,
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_toks,
                temperature=0.0 if attempt == 0 else 0.2,
            )

            extracted = extract_code_or_text(raw_resp)
            if "print(" in extracted or "import " in extracted or "def " in extracted:
                ok, out_or_err = await self._run_script_in_tmp(extracted, tf.task_id)
                if ok and out_or_err.strip():
                    return out_or_err.strip().splitlines()[-1].strip()
                last_error = out_or_err
            elif extracted.strip():
                # Direct scalar answer returned
                lines = [ln.strip() for ln in extracted.strip().splitlines() if ln.strip()]
                if lines:
                    return lines[-1]

        return None

    @staticmethod
    async def _run_script_in_tmp(code: str, task_id: str) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix=f"math_{task_id}_") as tmp_dir:
            script_path = Path(tmp_dir) / "solve_math.py"
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
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "TimeoutError in math verification script."

            if proc.returncode == 0:
                return True, stdout_b.decode("utf-8", errors="replace").strip()
            return False, stderr_b.decode("utf-8", errors="replace").strip()
