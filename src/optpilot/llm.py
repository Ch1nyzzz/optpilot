"""Together AI LLM client wrapper."""

import json
import re
import time
from typing import Any

from openai import OpenAI

from optpilot.config import TOGETHER_API_KEY, TOGETHER_BASE_URL, SYSTEM_MODEL

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            timeout=300,
            max_retries=2,
        )
    return _client


def call_llm(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> str:
    """Call Together AI LLM with exponential backoff.

    GLM-5 is a reasoning model - it uses part of max_tokens for internal
    reasoning (invisible) before producing visible content. Set max_tokens
    high enough to accommodate both reasoning and output.
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


def call_llm_json(
    messages: list[dict[str, str]],
    model: str = SYSTEM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
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

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract JSON from markdown code blocks
    code_match = re.search(r'```(?:json)?\s*\n(.*?)```', content, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Extract first { ... } block
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
                    return json.loads(content[start:i+1])
                except json.JSONDecodeError:
                    start = -1
                    continue

    raise ValueError(f"Failed to parse JSON from LLM response: {content[:300]}")
