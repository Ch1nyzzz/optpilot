from __future__ import annotations

import asyncio
import threading
import time

import pytest

from optpilot.llm import (
    AsyncModelRateLimiter,
    ModelRateLimiter,
    _normalize_model_family,
)


def test_model_rate_limiter_waits_for_rpm_window() -> None:
    current = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return current[0]

    def sleep_fn(delay_s: float) -> None:
        sleeps.append(delay_s)
        current[0] += delay_s

    limiter = ModelRateLimiter(
        max_concurrency=10,
        rpm=2,
        clock=clock,
        sleep=sleep_fn,
    )

    limiter.acquire()
    limiter.release()
    limiter.acquire()
    limiter.release()
    limiter.acquire()
    limiter.release()

    assert sleeps == [60.0]


def test_model_rate_limiter_caps_concurrency() -> None:
    limiter = ModelRateLimiter(max_concurrency=1, rpm=1000)
    first_entered = threading.Event()
    allow_first_to_exit = threading.Event()
    second_entered = threading.Event()

    def first_worker() -> None:
        with limiter.limit():
            first_entered.set()
            allow_first_to_exit.wait(timeout=1)

    def second_worker() -> None:
        with limiter.limit():
            second_entered.set()

    t1 = threading.Thread(target=first_worker)
    t2 = threading.Thread(target=second_worker)

    t1.start()
    assert first_entered.wait(timeout=1)

    t2.start()
    time.sleep(0.05)
    assert second_entered.is_set() is False

    allow_first_to_exit.set()
    t1.join(timeout=1)
    t2.join(timeout=1)

    assert second_entered.is_set() is True


def test_model_family_normalization() -> None:
    assert _normalize_model_family("zai-org/GLM-5") == "glm5"
    assert _normalize_model_family("MiniMaxAI/MiniMax-M2.5") == "minimax_m2_5"
    assert _normalize_model_family("some/other-model") == "default"


@pytest.mark.anyio
async def test_async_model_rate_limiter_waits_for_rpm_window() -> None:
    current = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return current[0]

    async def sleep_fn(delay_s: float) -> None:
        sleeps.append(delay_s)
        current[0] += delay_s

    limiter = AsyncModelRateLimiter(
        max_concurrency=10,
        rpm=2,
        clock=clock,
        sleep=sleep_fn,
    )

    await limiter.acquire()
    limiter.release()
    await limiter.acquire()
    limiter.release()
    await limiter.acquire()
    limiter.release()

    assert sleeps == [60.0]


@pytest.mark.anyio
async def test_async_model_rate_limiter_caps_concurrency() -> None:
    limiter = AsyncModelRateLimiter(max_concurrency=1, rpm=1000)
    first_entered = asyncio.Event()
    allow_first_to_exit = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_worker() -> None:
        async with limiter:
            first_entered.set()
            await allow_first_to_exit.wait()

    async def second_worker() -> None:
        async with limiter:
            second_entered.set()

    task1 = asyncio.create_task(first_worker())
    await asyncio.wait_for(first_entered.wait(), timeout=1)

    task2 = asyncio.create_task(second_worker())
    await asyncio.sleep(0.05)
    assert second_entered.is_set() is False

    allow_first_to_exit.set()
    await asyncio.wait_for(task1, timeout=1)
    await asyncio.wait_for(task2, timeout=1)

    assert second_entered.is_set() is True
