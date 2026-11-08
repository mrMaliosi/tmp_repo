"""Smoke test for the real LLM proxy (NeuralDeep or any OpenAI-compatible API).

Verifies before a paid run:
1. Key validity (`GET /models`) and which of the 4 competition models are visible.
2. One tiny `chat/completions` call per competition model (~16 tokens), reporting
   plan-related 403s and the effective token cost of each call.

Usage (PowerShell):
    $env:LLM_BASE_URL = "https://api.neuraldeep.ru/v1"
    $env:LLM_API_KEY  = "sk-..."
    python tools/smoke_api.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

COMPETITION_MODELS = (
    "qwen3.6-35b-a3b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b-noreason",
    "qwen3.8-27b",
)


async def main() -> int:
    base = os.environ.get("LLM_BASE_URL", "https://api.neuraldeep.ru/v1").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        print("ERROR: LLM_API_KEY is not set.")
        return 1
    if not key.startswith("sk-"):
        print(f"WARNING: key does not start with 'sk-' ('{key[:6]}...').")
        print("         It looks like an invite code. Get a real key at https://hub.neuraldeep.ru/app/keys")

    print(f"Base URL: {base}")
    async with httpx.AsyncClient(
        base_url=base,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(45.0),
    ) as client:
        # --- 1) Key validity and model visibility -----------------------------
        try:
            r = await client.get("/models")
        except httpx.HTTPError as exc:
            print(f"ERROR: cannot reach the proxy: {exc}")
            return 1

        if r.status_code != 200:
            print(f"ERROR: GET /models -> HTTP {r.status_code}: {r.text[:200]}")
            return 1

        try:
            model_ids = {m.get("id", "") for m in r.json().get("data", [])}
        except Exception:
            model_ids = set()
        print(f"Key is valid. {len(model_ids)} models visible.")
        for m in COMPETITION_MODELS:
            print(f"  [{'visible' if m in model_ids else 'NOT in list'}] {m}")

        # --- 2) One tiny call per competition model ----------------------------
        print("\nTiny chat calls (max_tokens=16):")
        usable = 0
        for m in COMPETITION_MODELS:
            try:
                r = await client.post(
                    "/chat/completions",
                    json={
                        "model": m,
                        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                        "max_tokens": 16,
                        "temperature": 0.0,
                    },
                )
            except httpx.HTTPError as exc:
                print(f"  [{m}] network error: {exc}")
                continue
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})
                c_eff = (
                    int(usage.get("completion_tokens", 0))
                    + int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))
                    + 0.1 * int(usage.get("prompt_tokens", 0))
                )
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                print(f"  [OK ] {m}: reply={content.strip()[:20]!r} C_eff={c_eff:.1f}")
                usable += 1
            elif r.status_code in {403, 404}:
                print(f"  [403] {m}: unavailable on this plan -> harness will auto-fallback")
            else:
                print(f"  [{r.status_code}] {m}: {r.text[:120]}")

        print(f"\nResult: {usable}/{len(COMPETITION_MODELS)} competition models usable.")
        if usable == 0:
            print("No usable models - check the key/plan before running the harness.")
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
