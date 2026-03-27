"""Together AI LLM client wrapper."""

import asyncio
from collections import deque
from contextlib import contextmanager
import json
import re
import threading
import time
from typing import Any

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

from optpilot.config import (
    LLM_DEFAULT_MAX_CONCURRENCY,
    LLM_DEFAULT_RPM,
    LLM_GLM5_MAX_CONCURRENCY,
    LLM_GLM5_RPM,
    LLM_MINIMAX_M2_5_MAX_CONCURRENCY,
    LLM_MINIMAX_M2_5_RPM,
    SYSTEM_MODEL,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
)

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_limiters: dict[str, "ModelRateLimiter"] = {}
_async_limiters: dict[str, "AsyncModelRateLimiter"] = {}
_limiters_lock = threading.Lock()
_async_limiters_lock = threading.Lock()
_httpx_limits = httpx.Limits(
    max_connections=256,
    max_keepalive_connections=128,
    keepalive_expiry=60.0,
)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            timeout=300,
            max_retries=2,
            http_client=DefaultHttpxClient(
                limits=_httpx_limits,
            ),
        )
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            timeout=300,
            max_retries=2,
            http_client=DefaultAsyncHttpxClient(
                limits=_httpx_limits,
            ),
        )
    return _async_client


class ModelRateLimiter:
    """Thread-safe concurrency and RPM limiter for one model family."""

    def __init__(
        self,
        max_concurrency: int,
        rpm: int,
        *,
        window_s: float = 60.0,
        clock: callable = time.monotonic,
        sleep: callable = time.sleep,
    ):
        self.max_concurrency = max(1, max_concurrency)
        self.rpm = max(1, rpm)
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()

    @contextmanager
    def limit(self):
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def acquire(self) -> None:
        self._semaphore.acquire()
        try:
            self._acquire_rpm_slot()
        except Exception:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()

    def _acquire_rpm_slot(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                cutoff = now - self.window_s
                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                if len(self._request_times) < self.rpm:
                    self._request_times.append(now)
                    return

                wait_s = max(self.window_s - (now - self._request_times[0]), 0.01)

            self._sleep(wait_s)


class AsyncModelRateLimiter:
    """Async thread-safe concurrency and RPM limiter for one model family."""

    def __init__(
        self,
        max_concurrency: int,
        rpm: int,
        *,
        window_s: float = 60.0,
        clock: callable = time.monotonic,
        sleep: callable = asyncio.sleep,
    ):
        self.max_concurrency = max(1, max_concurrency)
        self.rpm = max(1, rpm)
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._request_times: deque[float] = deque()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        try:
            await self._acquire_rpm_slot()
        except Exception:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()

    async def _acquire_rpm_slot(self) -> None:
        while True:
            async with self._lock:
                now = self._clock()
                cutoff = now - self.window_s
                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                if len(self._request_times) < self.rpm:
                    self._request_times.append(now)
                    return

                wait_s = max(self.window_s - (now - self._request_times[0]), 0.01)

            await self._sleep(wait_s)


def _normalize_model_family(model: str) -> str:
    lowered = model.lower()
    if "glm-5" in lowered or "glm5" in lowered:
        return "glm5"
    if "minimax" in lowered and ("m2.5" in lowered or "m2_5" in lowered):
        return "minimax_m2_5"
    return "default"


def _build_rate_limiter(model: str) -> ModelRateLimiter:
    family = _normalize_model_family(model)
    if family == "glm5":
        return ModelRateLimiter(
            max_concurrency=LLM_GLM5_MAX_CONCURRENCY,
            rpm=LLM_GLM5_RPM,
        )
    if family == "minimax_m2_5":
        return ModelRateLimiter(
            max_concurrency=LLM_MINIMAX_M2_5_MAX_CONCURRENCY,
            rpm=LLM_MINIMAX_M2_5_RPM,
        )
    return ModelRateLimiter(
        max_concurrency=LLM_DEFAULT_MAX_CONCURRENCY,
        rpm=LLM_DEFAULT_RPM,
    )


def _build_async_rate_limiter(model: str) -> AsyncModelRateLimiter:
    family = _normalize_model_family(model)
    if family == "glm5":
        return AsyncModelRateLimiter(
            max_concurrency=LLM_GLM5_MAX_CONCURRENCY,
            rpm=LLM_GLM5_RPM,
        )
    if family == "minimax_m2_5":
        return AsyncModelRateLimiter(
            max_concurrency=LLM_MINIMAX_M2_5_MAX_CONCURRENCY,
            rpm=LLM_MINIMAX_M2_5_RPM,
        )
    return AsyncModelRateLimiter(
        max_concurrency=LLM_DEFAULT_MAX_CONCURRENCY,
        rpm=LLM_DEFAULT_RPM,
    )


def get_rate_limiter(model: str) -> ModelRateLimiter:
    family = _normalize_model_family(model)
    with _limiters_lock:
        limiter = _limiters.get(family)
        if limiter is None:
            limiter = _build_rate_limiter(model)
            _limiters[family] = limiter
        return limiter


def get_async_rate_limiter(model: str) -> AsyncModelRateLimiter:
    family = _normalize_model_family(model)
    with _async_limiters_lock:
        limiter = _async_limiters.get(family)
        if limiter is None:
            limiter = _build_async_rate_limiter(model)
            _async_limiters[family] = limiter
        return limiter


def _extract_json_dict(content: str) -> dict:
    """Parse a dict-like JSON response, allowing fenced code blocks."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    code_match = re.search(r'```(?:json)?\s*\n(.*?)```', content, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    depth = 0
    start = -1
    for i, c in enumerate(content):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(content[start:i+1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    start = -1
                    continue

    raise ValueError(f"Failed to parse JSON from LLM response: {content[:300]}")


JSON_REPAIR_PROMPT = """\
Convert the following content into a single valid JSON object.

Requirements:
- Preserve the original fields and values as much as possible.
- Output ONLY the JSON object.
- Do not add markdown fences or explanations.

Content:
{content}
"""


def _repair_json_dict(content: str, model: str) -> dict:
    repaired = call_llm(
        [{"role": "user", "content": JSON_REPAIR_PROMPT.format(content=content)}],
        model=model,
        temperature=0.0,
        max_tokens=4096,
        max_retries=2,
    )
    return _extract_json_dict(repaired)


async def _arepair_json_dict(content: str, model: str) -> dict:
    repaired = await acall_llm(
        [{"role": "user", "content": JSON_REPAIR_PROMPT.format(content=content)}],
        model=model,
        temperature=0.0,
        max_tokens=4096,
        max_retries=2,
    )
    return _extract_json_dict(repaired)


def call_llm(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    max_retries: int = 3,
) -> str:
    """Call Together AI LLM with exponential backoff.

    Some reasoning models may consume part of max_tokens for internal
    reasoning before producing visible content. Keep max_tokens high enough
    for both reasoning and final output, especially on Judge calls.
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            with get_rate_limiter(model).limit():
                resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if not content and resp.choices[0].finish_reason == "length":
                # Reasoning model ran out of tokens before producing content
                if attempt < max_retries - 1:
                    kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 65536)
                    print(f"  Empty content (length limit), retrying with max_tokens={kwargs['max_tokens']}...")
                    continue
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            err_str = str(e)
            if "rate_limit" in err_str or "429" in err_str:
                wait = 30 * (attempt + 1)  # rate limit: 30s, 60s, 90s
            else:
                wait = 2 ** (attempt + 1)
            print(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""


async def acall_llm(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    max_retries: int = 3,
) -> str:
    """Async Together AI call with model-family concurrency and RPM limits."""
    client = get_async_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            async with get_async_rate_limiter(model):
                resp = await client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if not content and resp.choices[0].finish_reason == "length":
                if attempt < max_retries - 1:
                    kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 65536)
                    print(f"  Empty content (length limit), retrying with max_tokens={kwargs['max_tokens']}...")
                    continue
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            err_str = str(e)
            if "rate_limit" in err_str or "429" in err_str:
                wait = 30 * (attempt + 1)
            else:
                wait = 2 ** (attempt + 1)
            print(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
    return ""


async def acall_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    tool_executor: Any,
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    max_turns: int = 20,
    max_retries: int = 3,
) -> list[dict]:
    """Async LLM agent loop with tool calling.

    Calls the LLM, executes any tool calls via ``tool_executor(name, args)``,
    feeds results back, and repeats until the LLM stops calling tools or
    ``max_turns`` is reached.

    Args:
        tool_executor: callable(name: str, arguments: dict) -> str

    Returns the full message history including tool call/result messages.
    """
    client = get_async_client()
    msgs = list(messages)

    for _turn in range(max_turns):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = None
        for attempt in range(max_retries):
            try:
                async with get_async_rate_limiter(model):
                    resp = await client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                err_str = str(e)
                if "rate_limit" in err_str or "429" in err_str:
                    wait = 30 * (attempt + 1)
                else:
                    wait = 2 ** (attempt + 1)
                print(f"LLM tool call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

        if resp is None:
            break

        choice = resp.choices[0]
        assistant_msg = choice.message

        # Append the assistant message
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ]
        msgs.append(msg_dict)

        # If no tool calls, we're done
        if not assistant_msg.tool_calls:
            break

        # Execute each tool call and append results
        for tc in assistant_msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result_str = tool_executor(tc.function.name, args)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return msgs


def call_llm_json(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> dict:
    """Call LLM and parse JSON response.

    Does NOT use response_format=json_object (unreliable with reasoning models).
    Instead relies on prompt instructions + JSON extraction from output.
    """
    content = call_llm(
        messages, model=model, temperature=temperature, max_tokens=max_tokens,
    )

    if not content:
        raise ValueError("LLM returned empty content")
    try:
        return _extract_json_dict(content)
    except ValueError:
        return _repair_json_dict(content, model=model)


async def acall_llm_json(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> dict:
    """Async variant of ``call_llm_json``."""
    content = await acall_llm(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not content:
        raise ValueError("LLM returned empty content")
    try:
        return _extract_json_dict(content)
    except ValueError:
        return await _arepair_json_dict(content, model=model)
