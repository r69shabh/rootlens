"""Report layer: verdict card + evidence trail as exportable markdown."""

from __future__ import annotations

import json

from diagnosis.agent import DiagnosisResult
from diagnosis.evidence import EvidenceStore


def to_markdown(
    result: DiagnosisResult, store: EvidenceStore | None = None, ground_truth: dict | None = None
) -> str:
    lines = ["# RootLens diagnosis report", ""]
    lines.append(f"**Status:** {result.status}")
    if result.status == "verdict":
        lines += [
            f"**Root cause:** `{result.root_cause}`",
            f"**Confidence:** {result.confidence}",
            "",
            "## Disconfirmation checks",
        ]
        lines += [f"- {d}" for d in result.disconfirmation] or ["- (none recorded)"]
    elif result.status == "inconclusive":
        lines += ["", f"**Inconclusive — missing:** {result.missing}"]
    if result.impact:
        lines += ["", "## Business impact", ""]
        est = result.impact.get("estimated", result.impact)
        for k, v in est.items():
            lines.append(f"- {k}: {v}")
    if result.evidence_call_ids:
        lines += ["", "## Evidence chain"]
        for cid in result.evidence_call_ids:
            entry = store.get(cid) if store else None
            if entry:
                args = json.dumps(entry.args, default=str)
                lines.append(
                    f"- `{cid}` **{entry.tool}** — args: `{args}` ({entry.row_count} rows)"
                )
            else:
                lines.append(f"- `{cid}` (not found in audit trail)")
    if ground_truth:
        lines += [
            "",
            "## Ground truth (eval only)",
            f"- expected: {ground_truth.get('expected_labels')}",
        ]
    return "\n".join(lines) + "\n"
