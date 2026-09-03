"""Thin model-agnostic LLM adapter (BYOK harness, design principle 7).

Text-in / text-out only. The tool protocol lives in the prompt as JSON, so the
agent loop is fully decoupled from any provider's function-calling API and the
transcript stays inspectable.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


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
        self.usage.append({
            "input_tokens": max(1, prompt_chars // 4),
            "output_tokens": max(1, len(response) // 4),
            "estimated": True,
        })
        return response


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        super().__init__()
        from openai import OpenAI  # lazy import: llm extras are optional
        self.model_name = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[{"role": "system", "content": system}, *messages],
        )
        if resp.usage:
            self.usage.append({"input_tokens": resp.usage.prompt_tokens,
                               "output_tokens": resp.usage.completion_tokens,
                               "estimated": False})
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        super().__init__()
        import anthropic  # lazy import: llm extras are optional
        self.model_name = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=0,
            system=system,
            messages=messages,
        )
        if resp.usage:
            self.usage.append({"input_tokens": resp.usage.input_tokens,
                               "output_tokens": resp.usage.output_tokens,
                               "estimated": False})
        return "".join(block.text for block in resp.content if block.type == "text")


def get_client(provider: str, model: str | None = None) -> LLMClient:
    if provider == "scripted":
        return ScriptedLLMClient([])
    if provider == "openai":
        return OpenAIClient(model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(model or "claude-sonnet-4-5")
    raise ValueError(f"unknown provider {provider!r}")


def parse_json_response(text: str) -> dict:
    """Robustly extract the JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"model response is not JSON: {text[:200]!r}") from None
