"""Tests for transparent model fallback on plan-related model-access errors (403/404).

Simulates the NeuralDeep free-tier behavior: paid models (`qwen3.8-27b*`) answer
HTTP 403 with header `X-Window: free-model`, while `qwen3.6-35b-a3b(-noreason)` work.
The gateway must substitute the model and continue the run instead of aborting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import unittest

import httpx

from harness.budget_ledger import BudgetLedger, GlobalBudgetExhaustedException
from harness.jiuwen_agent import JiuwenAgentGateway

PAID_MODELS = {"qwen3.8-27b", "qwen3.8-27b-noreason"}
FREE_MODELS = {"qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"}


def _fake_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode("utf-8"))
    model = body["model"]
    if model in PAID_MODELS:
        return httpx.Response(
            403,
            headers={"X-Window": "free-model"},
            json={"error": {"message": "Model requires a paid plan", "type": "permission_error"}},
        )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 2, "reasoning_tokens": 0},
        },
    )


def _make_gateway() -> JiuwenAgentGateway:
    ledger = BudgetLedger(global_budget_limit=10_000.0)
    gateway = JiuwenAgentGateway(ledger=ledger)

    async def _mocked_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="https://api.test.local/v1",
            headers={"Authorization": "Bearer sk-test"},
            transport=httpx.MockTransport(_fake_handler),
        )

    gateway._get_client = _mocked_client  # type: ignore[method-assign]
    return gateway


class TestModelAccessFallback(unittest.IsolatedAsyncioTestCase):
    async def test_403_free_model_falls_back_and_solves(self) -> None:
        gw = _make_gateway()
        try:
            # Request a paid model that the proxy refuses (free-tier 403).
            answer = await gw.chat_completion(
                task_id="t1",
                model="qwen3.8-27b-noreason",
                messages=[{"role": "user", "content": "2+2?"}],
                max_tokens=10,
            )
            self.assertEqual(answer, "42")
            # The call must have been re-routed to a free-tier model and billed.
            self.assertNotIn("qwen3.8-27b-noreason", gw.ledger.get_task_record("t1").model_history)
            self.assertIn(gw.ledger.get_task_record("t1").model_history[-1], FREE_MODELS)
            # Blocked model must be remembered for subsequent calls.
            self.assertIn("qwen3.8-27b-noreason", gw.blocked_models)
            self.assertFalse(gw.ledger.proxy_exhausted)
        finally:
            await gw.aclose()

    async def test_blocked_model_is_resolved_before_http_call(self) -> None:
        gw = _make_gateway()
        try:
            await gw.chat_completion(
                task_id="t1",
                model="qwen3.8-27b-noreason",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            # Second call to the same blocked model must be resolved client-side
            # to a free model without hitting the proxy again.
            answered = await gw.chat_completion(
                task_id="t2",
                model="qwen3.8-27b-noreason",
                messages=[{"role": "user", "content": "hi again"}],
                max_tokens=5,
            )
            self.assertEqual(answered, "42")
            self.assertNotIn(
                "qwen3.8-27b-noreason", gw.ledger.get_task_record("t2").model_history
            )
        finally:
            await gw.aclose()

    async def test_all_models_blocked_raises_exhaustion(self) -> None:
        ledger = BudgetLedger(global_budget_limit=10_000.0)
        gw = JiuwenAgentGateway(ledger=ledger)
        gw.blocked_models = {
            "qwen3.8-27b", "qwen3.8-27b-noreason",
            "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason",
        }
        with self.assertRaises(GlobalBudgetExhaustedException):
            await gw.chat_completion(
                task_id="t3",
                model="qwen3.6-35b-a3b-noreason",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
        self.assertTrue(ledger.proxy_exhausted)

    async def test_403_without_model_markers_still_aborts(self) -> None:
        """A generic 403 (quota/subscription) must still abort the run, as before."""
        ledger = BudgetLedger(global_budget_limit=10_000.0)
        gw = JiuwenAgentGateway(ledger=ledger)

        async def _mocked_client() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                base_url="https://api.test.local/v1",
                headers={"Authorization": "Bearer sk-test"},
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(403, json={"error": "quota exceeded"})
                ),
            )

        gw._get_client = _mocked_client  # type: ignore[method-assign]
        with self.assertRaises(GlobalBudgetExhaustedException):
            await gw.chat_completion(
                task_id="t4",
                model="qwen3.6-35b-a3b-noreason",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
        self.assertTrue(ledger.proxy_exhausted)

    async def test_429_rate_limit_is_retried_not_fatal(self) -> None:
        """Transient 429 must be retried with backoff; the task must not be lost."""
        ledger = BudgetLedger(global_budget_limit=10_000.0)
        gw = JiuwenAgentGateway(ledger=ledger)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "rate limit exceeded"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "42"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 2, "reasoning_tokens": 0},
                },
            )

        async def _mocked_client() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                base_url="https://api.test.local/v1",
                headers={"Authorization": "Bearer sk-test"},
                transport=httpx.MockTransport(handler),
            )

        gw._get_client = _mocked_client  # type: ignore[method-assign]
        answer = await gw.chat_completion(
            task_id="t5",
            model="qwen3.6-35b-a3b-noreason",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        self.assertEqual(answer, "42")
        self.assertEqual(calls["n"], 2)  # one 429 + one success
        self.assertFalse(ledger.proxy_exhausted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
