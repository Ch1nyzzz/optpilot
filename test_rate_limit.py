"""Test Together AI rate limits for multiple models by increasing concurrency."""

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.environ.get("TOGETHER_AI_API_KEY") or os.environ.get("together_ai_api")
BASE_URL = "https://api.together.xyz/v1"

MODELS = [
    "Qwen/Qwen3.5-397B-A17B",
    "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2.5",
]

# Simple prompt that returns quickly
MESSAGES = [{"role": "user", "content": "Say 'hello' and nothing else."}]


async def single_call(client: AsyncOpenAI, model: str, idx: int) -> tuple[int, bool, str]:
    """Make a single API call. Returns (index, success, error_or_empty)."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=MESSAGES,
            max_tokens=16,
            temperature=0,
        )
        return (idx, True, "")
    except Exception as e:
        return (idx, False, str(e)[:120])


async def test_model_concurrency(model: str, max_concurrency: int = 50, step: int = 5) -> int:
    """Binary-search style: ramp up concurrency until we hit rate limits."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=0)

    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")

    last_success_n = 0

    # Test increasing concurrency levels
    levels = list(range(step, max_concurrency + 1, step))
    if levels and levels[-1] != max_concurrency:
        levels.append(max_concurrency)

    for n in levels:
        tasks = [single_call(client, model, i) for i in range(n)]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

        successes = sum(1 for _, ok, _ in results if ok)
        failures = sum(1 for _, ok, _ in results if not ok)
        errors = set(err for _, ok, err in results if not ok)

        status = "✓ ALL OK" if failures == 0 else f"✗ {failures} FAILED"
        print(f"  n={n:3d}  {status}  successes={successes}  time={elapsed:.1f}s")
        if errors:
            for e in errors:
                print(f"         error: {e[:100]}")

        if failures == 0:
            last_success_n = n
        else:
            # If more than half failed, stop ramping
            if failures > n * 0.5:
                print(f"  → Stopping: too many failures at n={n}")
                break
            last_success_n = successes  # partial success count

        # Small pause between levels to avoid cascading rate limits
        await asyncio.sleep(2)

    await client.close()
    print(f"  ▶ Max observed concurrency for {model}: ~{last_success_n}")
    return last_success_n


async def main():
    if not API_KEY:
        print("ERROR: No Together AI API key found in .env")
        return

    print(f"Together AI Rate Limit Concurrency Test")
    print(f"API Key: {API_KEY[:12]}...")
    print(f"Models: {len(MODELS)}")

    results = {}
    for model in MODELS:
        try:
            max_c = await test_model_concurrency(model, max_concurrency=50, step=5)
            results[model] = max_c
        except Exception as e:
            print(f"  ERROR testing {model}: {e}")
            results[model] = -1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, max_c in results.items():
        print(f"  {model:40s}  max_concurrency ≈ {max_c}")


if __name__ == "__main__":
    asyncio.run(main())
