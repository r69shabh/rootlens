"""Thin model-agnostic LLM adapter (BYOK harness, design principle 7).

Text-in / text-out only. The tool protocol lives in the prompt as JSON, so the
agent loop is fully decoupled from any provider's function-calling API and the
transcript stays inspectable.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod

# Retry knobs. 3 attempts with 1s/2s backoff covers a transient 5xx; rate
# limits (429) get their own slower lane: up to 6 attempts with a 35s sleep,
# which also lets a per-minute quota refill (e.g. Gemini free tier: 5/min).
# The per-call timeout is generous because tool-call prompts can be slow.
MAX_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 6
RETRY_BACKOFF_S = 1.0
RATE_LIMIT_SLEEP_S = 35.0
LLM_TIMEOUT_S = 60.0


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "ratelimit" in type(exc).__name__.lower()


def _chat_with_retry(call):
    """Run a provider chat() with bounded retries on transient errors.

    `call` is a zero-arg callable that performs one request and returns the
    response object. We retry on any exception except ValueError (which is the
    contract-violation signal from the agent loop and should propagate).
    """
    rate_limit_hits = 0
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + MAX_RATE_LIMIT_RETRIES):
        try:
            return call()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on any transient provider error
            last_exc = exc
            if _is_rate_limit(exc) and rate_limit_hits < MAX_RATE_LIMIT_RETRIES:
                rate_limit_hits += 1
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            if attempt >= MAX_RETRIES - 1 + rate_limit_hits:
                break
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_exc}") from last_exc


class LLMClient(ABC):
    model_name: str = "abstract"

    def __init__(self) -> None:
        self.usage: list[dict] = []  # {input_tokens, output_tokens, estimated}

    @property
    def total_tokens(self) -> int:
        return sum(u["input_tokens"] + u["output_tokens"] for u in self.usage)

    @abstractmethod
    def chat(self, system: str, messages: list[dict]) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text."""

    def complete(self, system: str, user: str) -> str:
        return self.chat(system, [{"role": "user", "content": user}])


class ScriptedLLMClient(LLMClient):
    """Returns queued responses. Used by tests, replay mode and offline demos."""

    model_name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, system: str, messages: list[dict]) -> str:
        self.calls.append({"system": system, "messages": messages})
        if not self.responses:
            raise RuntimeError("ScriptedLLMClient exhausted; add more scripted responses")
        response = self.responses.pop(0)
        prompt_chars = len(system) + sum(len(str(m.get("content", ""))) for m in messages)
        self.usage.append(
            {
                "input_tokens": max(1, prompt_chars // 4),
                "output_tokens": max(1, len(response) // 4),
                "estimated": True,
            }
        )
        return response


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        from openai import OpenAI  # lazy import: llm extras are optional

        self.model_name = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = _chat_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                timeout=LLM_TIMEOUT_S,
                messages=[{"role": "system", "content": system}, *messages],
            )
        )
        if resp.usage:
            self.usage.append(
                {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "estimated": False,
                }
            )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        super().__init__()
        import anthropic  # lazy import: llm extras are optional

        self.model_name = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = _chat_with_retry(
            lambda: self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0,
                timeout=LLM_TIMEOUT_S,
                system=system,
                messages=messages,
            )
        )
        if resp.usage:
            self.usage.append(
                {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "estimated": False,
                }
            )
        # Some content blocks (tool_use, etc.) don't have .text; filter
        # defensively rather than crashing on non-text-only responses.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class GeminiClient(OpenAIClient):
    """Gemini via its OpenAI-compatible endpoint. Model override with
    GEMINI_MODEL (default gemini-3.6-flash)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(
            model=model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )


def get_client(provider: str, model: str | None = None) -> LLMClient:
    if provider == "scripted":
        return ScriptedLLMClient([])
    if provider == "openai":
        return OpenAIClient(model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(model or "claude-sonnet-4-5")
    if provider == "gemini":
        return GeminiClient(model)
    raise ValueError(f"unknown provider {provider!r}")


def parse_json_response(text: str) -> dict:
    """Robustly extract the JSON object from a model response.

    Always raises ValueError on failure (the agent loop checks for ValueError
    to send a "invalid JSON, try again" nudge); the raw json.JSONDecodeError
    must never escape, since the loop's `except ValueError` path is the only
    recovery mechanism.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fallback: extract the outermost {...} block from surrounding prose
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"model response is not JSON: {text[:200]!r} ({exc.msg})") from None
    raise ValueError(f"model response is not JSON: {text[:200]!r}")
