"""Replay mode: cache full LLM transcripts per scenario so demos and evals do
not depend on a live API call succeeding (architecture 4.5 / risk table)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from diagnosis.llm_client import LLMClient


def _key(system: str, messages: list[dict]) -> str:
    payload = json.dumps({"system": system, "messages": messages},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ReplayCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            self.records = json.loads(self.path.read_text())

    def get(self, system: str, messages: list[dict]) -> str | None:
        rec = self.records.get(_key(system, messages))
        return rec["response"] if rec else None

    def put(self, system: str, messages: list[dict], response: str,
            model_name: str = "") -> None:
        self.records[_key(system, messages)] = {
            "response": response, "model_name": model_name,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2, default=str))

    def __len__(self) -> int:
        return len(self.records)


class RecordingLLMClient(LLMClient):
    """Wraps a live client; every chat call is recorded into the cache."""

    def __init__(self, inner: LLMClient, cache: ReplayCache) -> None:
        self.inner = inner
        self.cache = cache
        self.model_name = inner.model_name

    def chat(self, system: str, messages: list[dict]) -> str:
        response = self.inner.chat(system, messages)
        self.cache.put(system, messages, response, self.model_name)
        return response


class ReplayLLMClient(LLMClient):
    """Answers only from the cache. A miss is a loud error, not a silent live call."""

    model_name = "replay"

    def __init__(self, cache: ReplayCache) -> None:
        self.cache = cache

    def chat(self, system: str, messages: list[dict]) -> str:
        response = self.cache.get(system, messages)
        if response is None:
            raise LookupError(
                "replay cache miss: this transcript step was never recorded. "
                "Re-run with --record to refresh the cache for this scenario."
            )
        return response
