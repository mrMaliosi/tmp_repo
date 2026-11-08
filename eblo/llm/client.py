"""Asynchronous OpenAI-compatible LLM client with concurrency semaphore, retries, and streaming."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import httpx

from eblo.config.schema import LLMConfig
from eblo.examples.sample_task.solutions import (
    CANDIDATE_BITMASK_DP_FAST,
    CANDIDATE_DFS_SLOW,
    CANDIDATE_GREEDY_BUGGY,
)


class AsyncLLMClient:
    """Non-blocking LLM client targeting OpenAI-compatible servers (such as Strata / vLLM / llama.cpp)."""

    def __init__(
        self,
        config: LLMConfig,
        mock_responder: Callable[[list[dict[str, str]], float, bool], str] | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._mock_responder = mock_responder
        self._client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout_sec),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> tuple[str, float]:
        """Send chat completion request bounded by semaphore with exponential backoff retries."""
        async with self._semaphore:
            start_ns = time.perf_counter_ns()
            if self.config.provider == "mock" or self._mock_responder is not None:
                if self._mock_responder is not None:
                    await asyncio.sleep(0.002)
                    content = self._mock_responder(messages, temperature, json_mode)
                else:
                    content = self._default_mock_response(messages, temperature, json_mode)
                latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                return content, latency_ms

            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": round(float(temperature), 4),
                "stream": self.config.stream,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            last_exc: Exception | None = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    client = await self._get_http_client()
                    if self.config.stream:
                        text = await self._complete_streaming(client, payload)
                    else:
                        response = await client.post("/chat/completions", json=payload)
                        response.raise_for_status()
                        data = response.json()
                        text = self._extract_message_text(data)
                    latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                    return text, latency_ms
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.config.max_retries:
                        backoff_sec = min(4.0, 0.25 * (2**attempt))
                        await asyncio.sleep(backoff_sec)
            raise RuntimeError(
                f"LLM completion failed after {self.config.max_retries + 1} attempts: {last_exc}"
            ) from last_exc

    async def _complete_streaming(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> str:
        chunks: list[str] = []
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                stripped = line.strip()
                if not stripped or not stripped.startswith("data:"):
                    continue
                data_str = stripped[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning_piece:
                            if not chunks or "<think>" not in "".join(chunks):
                                chunks.append("<think>")
                            chunks.append(str(reasoning_piece))
                        content_piece = delta.get("content")
                        if content_piece:
                            if "<think>" in "".join(chunks) and "</think>" not in "".join(chunks):
                                chunks.append("</think>\n")
                            chunks.append(str(content_piece))
                except json.JSONDecodeError:
                    continue
        full_text = "".join(chunks)
        if "<think>" in full_text and "</think>" not in full_text:
            full_text += "</think>"
        return full_text

    @staticmethod
    def _extract_message_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Empty choices in LLM response: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning and "<think>" not in content:
            return f"<think>\n{reasoning.strip()}\n</think>\n{content}"
        return str(content)

    @staticmethod
    def _default_mock_response(
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        if json_mode or "Return JSON" in prompt_text or "valid JSON object" in prompt_text or "JSON format" in prompt_text:
            if "strategies" in prompt_text and "Generate" in prompt_text:
                return (
                    "<think>Synthesizing initial algorithmic strategies for P_0.</think>\n"
                    + json.dumps(
                        {
                            "strategies": [
                                {
                                    "system_persona": "Greedy Heuristic Prototyping Engineer",
                                    "algorithmic_strategy": "Construct a fast greedy nearest-neighbor tour.",
                                    "constraints": ["Use standard library only"],
                                    "temperature": 0.4,
                                },
                                {
                                    "system_persona": "Branch-and-Bound Optimization Architect",
                                    "algorithmic_strategy": "Implement recursive branch-and-bound DFS ordered by earliest deadline.",
                                    "constraints": ["Maintain best upper bound", "Prune infeasible arrivals"],
                                    "temperature": 0.65,
                                },
                                {
                                    "system_persona": "State Space Search Engineer",
                                    "algorithmic_strategy": "Implement backtracking search with feasibility pruning.",
                                    "constraints": ["Avoid recursion depth traps"],
                                    "temperature": 0.75,
                                },
                            ]
                        }
                    )
                )
            return (
                "<think>Refining strategy via evolutionary genetic operator to use Pareto Bitmask DP.</think>\n"
                + json.dumps(
                    {
                        "system_persona": "Principal Algorithm Engineer",
                        "algorithmic_strategy": (
                            "Use bitmask dynamic programming with Pareto state dominance pruning "
                            "and local variable binding for blazing-fast execution."
                        ),
                        "constraints": [
                            "Strictly prune states exceeding time window deadlines",
                            "Eliminate dominated (cost, finish_time) states per (mask, u)",
                        ],
                    }
                )
            )

        lower = prompt_text.lower()
        if "pareto" in lower or "bitmask dynamic programming" in lower or "reflexion" in lower:
            code = CANDIDATE_BITMASK_DP_FAST
            thought = "Applying bitmask dynamic programming with Pareto dominance pruning over (cost, finish_time)."
        elif "branch-and-bound" in lower or "dfs" in lower or "backtracking" in lower:
            code = CANDIDATE_DFS_SLOW
            thought = "Implementing exact DFS branch-and-bound ordered by earliest deadline."
        else:
            code = CANDIDATE_GREEDY_BUGGY
            thought = "Trying a simple greedy nearest-neighbor construction first."

        return f"<think>\n{thought}\n</think>\n```python\n{code.strip()}\n```"
