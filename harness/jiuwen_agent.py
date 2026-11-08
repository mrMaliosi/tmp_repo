"""Wrapper over `openJiuwen` (`openjiuwen==0.1.19`) with In-Flight token budget enforcement and context compression."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable

import httpx

from harness.budget_ledger import BudgetLedger, GlobalBudgetExhaustedException

try:
    import openjiuwen  # type: ignore[import-not-found]

    HAS_OPENJIUWEN = True
except ImportError:
    openjiuwen = None
    HAS_OPENJIUWEN = False

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Fallback preference order for transparent model substitution when the proxy
# returns a plan-related model-access error (HTTP 403/404, e.g. NeuralDeep
# free-tier `X-Window: free-model`): cheapest and free-tier-friendly first.
_FALLBACK_MODEL_ORDER: tuple[str, ...] = (
    "qwen3.6-35b-a3b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b-noreason",
    "qwen3.8-27b",
)

_MODEL_ACCESS_BODY_MARKERS: tuple[str, ...] = (
    "free-model",
    "free model",
    "тариф",
    "доступ",
    "access to model",
    "model access",
    "not allowed",
    "permission",
    "model not found",
    "model_not_found",
)

# Rate-limit resilience: transient 429 answers are retried with capped backoff
# (base 1.5s, x2, capped at 8s). After MAX_RETRIES_429 consecutive failures the
# old budget-exhaustion semantics kick in.
MAX_RETRIES_429 = 3


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff for rate-limit retries: 1.5s, 3s, 6s (capped at 8s)."""
    return min(8.0, 1.5 * (2 ** (attempt - 1)))


def compress_prompt_context(text: str, max_chars: int = 2400) -> str:
    """Compress whitespace and truncate redundant middle sections to save prompt tokens."""
    collapsed = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(collapsed) <= max_chars:
        return collapsed
    half = max_chars // 2 - 20
    return collapsed[:half] + "\n...[truncated]...\n" + collapsed[-half:]


def extract_code_or_text(raw: str) -> str:
    """Strip `<think>...</think>` blocks and extract code block if present."""
    cleaned = _THINK_RE.sub("", raw or "").strip()
    matches = _CODE_BLOCK_RE.findall(cleaned)
    if matches:
        return max((m.strip() for m in matches if m.strip()), key=len, default=cleaned)
    return cleaned


class JiuwenAgentGateway:
    """Budget-aware wrapper over `openJiuwen` agent runtime and `$LLM_BASE_URL` proxy."""

    def __init__(
        self,
        ledger: BudgetLedger,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: float = 45.0,
        mock_handler: Callable[[str, str, list[dict[str, str]], int], dict[str, Any]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:8080/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or "competition-proxy-key"
        # Optional Host header override (e.g. to pass DNS-rebinding protection of local LLM proxies
        # when the container reaches them via `host.docker.internal` on Docker Desktop).
        # On the competition host (`--network host`) this env is unset and the default behavior is kept.
        self.host_header = os.environ.get("LLM_HOST_HEADER") or None
        self.timeout_sec = timeout_sec
        self.mock_handler = mock_handler
        self.blocked_models: set[str] = set()
        self.logger: Any = None  # optional RunLogger injected by the runner
        self.consecutive_net_errors = 0
        self._http_client: httpx.AsyncClient | None = None
        self.openjiuwen_version: str = getattr(openjiuwen, "__version__", "0.1.19-compat")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            headers: dict[str, str] = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.host_header:
                headers["Host"] = self.host_header
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_sec),
            )
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()

    def resolve_model(self, requested_model: str) -> str | None:
        """Return `requested_model` unless it is blocked by a plan-related 403/404;
        otherwise return the best still-allowed fallback model (or None if all 4 are blocked)."""
        if requested_model not in self.blocked_models:
            return requested_model
        for candidate in _FALLBACK_MODEL_ORDER:
            if candidate not in self.blocked_models:
                return candidate
        return None

    @staticmethod
    def _is_model_access_error(response: httpx.Response) -> bool:
        """True when HTTP 403/404 means "this model is unavailable on your plan"
        (e.g. NeuralDeep free tier answers 403 with header `X-Window: free-model`)
        rather than a fatal proxy/auth failure. Such errors must trigger a per-call
        model fallback, never a global run abort."""
        if response.status_code not in {403, 404}:
            return False
        if response.headers.get("x-window"):
            return True
        body_lower = response.text.lower()
        return any(marker in body_lower for marker in _MODEL_ACCESS_BODY_MARKERS)

    async def chat_completion(
        self,
        task_id: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Execute an LLM call via openJiuwen / proxy while strictly tracking effective token cost."""
        self.ledger.check_task_can_proceed(task_id)

        compressed_messages = [
            {
                "role": m.get("role", "user"),
                "content": compress_prompt_context(m.get("content", ""), max_chars=3200),
            }
            for m in messages
        ]

        if self.mock_handler is not None:
            await asyncio.sleep(0.001)
            resp_data = self.mock_handler(task_id, model, compressed_messages, max_tokens)
            content = str(resp_data.get("content", ""))
            usage = resp_data.get("usage") or self._estimate_usage(compressed_messages, content, model)
            prompt_toks = int(usage.get("prompt_tokens", 0))
            comp_toks = int(usage.get("completion_tokens", 0))
            reason_toks = int(usage.get("reasoning_tokens", 0))
            delta_eff = self.ledger.record_llm_call(
                task_id=task_id,
                model=model,
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                reasoning_tokens=reason_toks,
                enforce_limit=True,
            )
            self._log_llm_call(task_id, model, prompt_toks, comp_toks, reason_toks, delta_eff)
            return content

        client = await self._get_client()

        # Transparent model substitution: a plan-related model-access error must
        # degrade the model tier for this run, not kill the whole tournament.
        resolved_model = self.resolve_model(model)
        if resolved_model is None:
            self.ledger.proxy_exhausted = True
            raise GlobalBudgetExhaustedException(
                "All 4 competition models are blocked by the proxy (model-access 403/404)."
            )
        model = resolved_model

        data: dict[str, Any] | None = None
        retried_429 = 0
        try:
            for _ in range(len(_FALLBACK_MODEL_ORDER) + MAX_RETRIES_429):
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": compressed_messages,
                    "max_tokens": int(max_tokens),
                    "temperature": round(float(temperature), 3),
                }
                response = await client.post("/chat/completions", json=payload)

                if self._is_model_access_error(response):
                    # Model unavailable on the current plan/key: block it and retry
                    # the very same call on the next allowed model.
                    self.blocked_models.add(model)
                    next_model = self.resolve_model(model)
                    if self.logger:
                        self.logger.event("model_blocked", task_id=task_id, blocked=model, fallback=next_model)
                    if next_model is None:
                        self.ledger.proxy_exhausted = True
                        raise GlobalBudgetExhaustedException(
                            "All 4 competition models are blocked by the proxy "
                            f"(last: HTTP {response.status_code}: {response.text[:120]})."
                        )
                    model = next_model
                    continue

                if response.status_code in {401, 402, 403, 429}:
                    body_lower = response.text.lower()
                    # 429 = transient rate limit: throttle and retry instead of losing the task.
                    if response.status_code == 429 and retried_429 < MAX_RETRIES_429:
                        retried_429 += 1
                        delay = _backoff_delay(retried_429)
                        if self.logger:
                            self.logger.event("rate_limited", task_id=task_id, retry=retried_429, delay=delay)
                        # Respect server-provided Retry-After if present
                        try:
                            ra = float(response.headers.get("retry-after", ""))
                            if 0 < ra <= 60.0:
                                delay = ra
                        except ValueError:
                            pass
                        await asyncio.sleep(delay)
                        continue
                    if response.status_code in {401, 402, 403} or "budget" in body_lower or "limit" in body_lower:
                        self.ledger.proxy_exhausted = True
                        if self.logger:
                            self.logger.event(
                                "proxy_exhausted",
                                task_id=task_id,
                                status=response.status_code,
                                body=response.text[:200],
                            )
                        raise GlobalBudgetExhaustedException(
                            f"Proxy returned HTTP {response.status_code}: {response.text[:200]}"
                        )
                response.raise_for_status()
                data = response.json()
                break
        except httpx.HTTPError as exc:
            self.consecutive_net_errors += 1
            if self.consecutive_net_errors >= 3:
                # Proxy stopped answering entirely (e.g. hard cutoff after token limit):
                # stop the run instead of burning the 1h time limit on timeouts.
                self.ledger.proxy_exhausted = True
                if self.logger:
                    self.logger.event(
                        "proxy_unreachable", task_id=task_id, streak=self.consecutive_net_errors, detail=str(exc)[:200]
                    )
                raise GlobalBudgetExhaustedException(
                    f"Proxy unreachable after {self.consecutive_net_errors} consecutive errors: {exc}"
                ) from exc
            if self.logger:
                self.logger.event(
                    "net_error", task_id=task_id, streak=self.consecutive_net_errors, detail=str(exc)[:200]
                )
            raise RuntimeError(f"Proxy communication error on task {task_id}: {exc}") from exc

        if data is None:
            raise RuntimeError(f"Proxy on task {task_id}: all fallback models exhausted.")

        self.consecutive_net_errors = 0

        choices = data.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        content = str(msg.get("content") or "")
        reasoning_text = str(msg.get("reasoning_content") or msg.get("reasoning") or "")

        usage = data.get("usage") or {}
        prompt_toks = int(usage.get("prompt_tokens") or self._approx_tokens(str(compressed_messages)))
        comp_toks = int(usage.get("completion_tokens") or self._approx_tokens(content))

        details = usage.get("completion_tokens_details") or {}
        reason_toks = int(
            usage.get("reasoning_tokens")
            or details.get("reasoning_tokens")
            or (self._approx_tokens(reasoning_text) if reasoning_text else 0)
        )
        if not model.endswith("-noreason") and reason_toks == 0 and "<think>" in content:
            think_parts = _THINK_RE.findall(content)
            reason_toks = sum(self._approx_tokens(tp) for tp in think_parts)

        delta_eff = self.ledger.record_llm_call(
            task_id=task_id,
            model=model,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            reasoning_tokens=reason_toks,
            enforce_limit=True,
        )
        self._log_llm_call(task_id, model, prompt_toks, comp_toks, reason_toks, delta_eff)
        return content

    def _log_llm_call(
        self,
        task_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        delta_eff: float,
    ) -> None:
        """Record per-call token usage to the JSONL run log (best-effort).

        This is the calibration data source: tools/calibrate_sim.py aggregates
        `llm_call` events by domain/model to fit the tournament simulator.
        """
        if self.logger:
            self.logger.event(
                "llm_call",
                task_id=task_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                delta_eff=round(delta_eff, 2),
            )

    @classmethod
    def _estimate_usage(
        cls,
        messages: list[dict[str, str]],
        content: str,
        model: str,
    ) -> dict[str, int]:
        prompt_str = "\n".join(m.get("content", "") for m in messages)
        p_toks = cls._approx_tokens(prompt_str)
        c_toks = cls._approx_tokens(content)
        r_toks = 0 if model.endswith("-noreason") else max(60, c_toks * 2)
        return {
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "reasoning_tokens": r_toks,
        }
