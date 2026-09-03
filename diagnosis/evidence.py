"""Evidence store / audit trail: every tool call is logged and citable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvidenceEntry:
    call_id: str
    tool: str
    args: dict
    result: object
    row_count: int
    duration_ms: float
    ts: str

    def summary_for_llm(self) -> str:
        return json.dumps(
            {"call_id": self.call_id, "tool": self.tool, "args": self.args,
             "result": self.result},
            default=str,
        )


@dataclass
class EvidenceStore:
    scenario_id: str = "default"
    entries: list[EvidenceEntry] = field(default_factory=list)
    _counter: int = 0

    def log(self, tool: str, args: dict, result, duration_ms: float) -> EvidenceEntry:
        self._counter += 1
        entry = EvidenceEntry(
            call_id=f"call_{self._counter:03d}",
            tool=tool,
            args=args,
            result=result,
            row_count=len(result) if isinstance(result, list) else (1 if result is not None else 0),
            duration_ms=round(duration_ms, 2),
            ts=datetime.now(UTC).isoformat(),
        )
        self.entries.append(entry)
        return entry

    def get(self, call_id: str) -> EvidenceEntry | None:
        return next((e for e in self.entries if e.call_id == call_id), None)

    def transcript_for_llm(self, last_n: int | None = None) -> str:
        items = self.entries[-last_n:] if last_n else self.entries
        return "\n".join(e.summary_for_llm() for e in items)

    def to_json(self) -> list[dict]:
        return [
            {"call_id": e.call_id, "tool": e.tool, "args": e.args, "result": e.result,
             "row_count": e.row_count, "duration_ms": e.duration_ms, "ts": e.ts}
            for e in self.entries
        ]
