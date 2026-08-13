"""Dependency-free helpers for selecting frozen evaluation-plan rows."""

from __future__ import annotations


def select_plan_rows(
    plan: dict,
    candidate_id: str,
    sample_id: str | None = None,
) -> list[dict]:
    """Select one candidate, optionally requiring one exact sample row."""
    candidate_rows = [
        row for row in plan.get("samples", []) if row.get("candidate_id") == candidate_id
    ]
    if not candidate_rows:
        raise ValueError(f"generation plan has no rows for candidate {candidate_id!r}")
    if sample_id is None:
        return candidate_rows
    rows = [row for row in candidate_rows if row.get("sample_id") == sample_id]
    if len(rows) != 1:
        raise ValueError(
            "generation plan must contain exactly one row for candidate "
            f"{candidate_id!r} and sample {sample_id!r}; found {len(rows)}"
        )
    return rows
