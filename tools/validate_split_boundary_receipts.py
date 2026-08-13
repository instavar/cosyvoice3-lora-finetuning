#!/usr/bin/env python3
"""Validate frozen frontend and vLLM receipts for a split-boundary probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_rows(rows: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError(f"every {label} row must contain sample_id")
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"duplicate {label} sample id: {sample_id}")
        result[sample_id] = row
    return result


def _request_identity(request: dict[str, Any]) -> tuple[int, str]:
    count = request.get("output_token_count")
    digest = request.get("output_token_sha256")
    if not isinstance(count, int) or count < 0:
        raise ValueError("request receipt has invalid output_token_count")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("request receipt has invalid output_token_sha256")
    return count, digest


def build_split_boundary_report(
    plan: dict[str, Any],
    protocol: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    generation_plan_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Fail closed on coverage drift, then compare frozen request identities."""
    expected = _unique_rows(plan.get("samples"), label="generation-plan")
    observed = _unique_rows(observations, label="observation")
    if set(expected) != set(observed):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(
            f"observation coverage mismatch; missing={missing}, unexpected={unexpected}"
        )

    calibration = protocol.get("calibration_expectations")
    if not isinstance(calibration, dict):
        raise ValueError("protocol must contain calibration_expectations")

    validation_rows = []
    for sample_id, planned in expected.items():
        row = observed[sample_id]
        prompt_id = row.get("prompt_id")
        expected_calibration = calibration.get(prompt_id)
        if not isinstance(expected_calibration, dict):
            raise ValueError(f"missing calibration for prompt {prompt_id!r}")
        frontend = row.get("frontend_segmentation")
        sampling = row.get("vllm_sampling")
        if not isinstance(frontend, dict) or not isinstance(sampling, dict):
            raise ValueError(f"sample {sample_id} is missing frontend or vLLM evidence")
        segments = frontend.get("segments")
        requests = sampling.get("requests")
        if not isinstance(segments, list) or not isinstance(requests, list):
            raise ValueError(f"sample {sample_id} has malformed segment or request evidence")
        checks = {
            "candidate_matches": row.get("candidate_id") == planned.get("candidate_id"),
            "prompt_matches": prompt_id == planned.get("prompt_id"),
            "seed_matches": row.get("seed") == planned.get("seed"),
            "requested_text_matches": row.get("requested_text") == planned.get("text"),
            "segment_count_matches_calibration": frontend.get("segment_count")
            == expected_calibration.get("segment_count"),
            "segment_hashes_match_calibration": [
                item.get("normalized_text_sha256") for item in segments
            ]
            == expected_calibration.get("segment_hashes"),
            "segment_token_counts_match_calibration": [
                item.get("text_token_count") for item in segments
            ]
            == expected_calibration.get("segment_token_counts"),
            "vllm_request_count_matches_frontend": frontend.get(
                "request_count_matches"
            )
            is True,
            "receipt_count_matches_frontend": len(requests)
            == frontend.get("segment_count"),
            "receipts_complete": all(
                request.get("status") == "complete" for request in requests
            ),
            "request_ordinals_complete": [
                request.get("request_ordinal") for request in requests
            ]
            == list(range(1, len(requests) + 1)),
        }
        validation_rows.append(
            {
                "sample_id": sample_id,
                "valid_audio": row.get("valid") is True,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    candidates = plan.get("candidate_ids")
    seeds = plan.get("seeds")
    if not isinstance(candidates, list) or not isinstance(seeds, list):
        raise ValueError("generation plan must contain candidate_ids and seeds")
    comparisons = []
    for candidate_id in candidates:
        for seed in seeds:
            group = {
                row.get("prompt_id"): row
                for row in observations
                if row.get("candidate_id") == candidate_id and row.get("seed") == seed
            }
            if set(group) != {"prefix", "tail", "combined"}:
                raise ValueError(
                    f"candidate {candidate_id!r} seed {seed} lacks prefix, tail, or combined"
                )
            prefix_requests = group["prefix"]["vllm_sampling"]["requests"]
            tail_requests = group["tail"]["vllm_sampling"]["requests"]
            combined_requests = group["combined"]["vllm_sampling"]["requests"]
            if not (
                len(prefix_requests) == 1
                and len(tail_requests) == 1
                and len(combined_requests) == 2
            ):
                raise ValueError(
                    f"candidate {candidate_id!r} seed {seed} has unexpected request counts"
                )
            standalone_prefix = _request_identity(prefix_requests[0])
            standalone_tail = _request_identity(tail_requests[0])
            combined_prefix = _request_identity(combined_requests[0])
            combined_tail = _request_identity(combined_requests[1])
            comparisons.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "prefix": {
                        "token_count_equal": standalone_prefix[0]
                        == combined_prefix[0],
                        "token_sha256_equal": standalone_prefix[1]
                        == combined_prefix[1],
                        "standalone_token_count": standalone_prefix[0],
                        "combined_token_count": combined_prefix[0],
                        "standalone_token_sha256": standalone_prefix[1],
                        "combined_token_sha256": combined_prefix[1],
                    },
                    "tail": {
                        "token_count_equal": standalone_tail[0] == combined_tail[0],
                        "token_sha256_equal": standalone_tail[1] == combined_tail[1],
                        "standalone_token_count": standalone_tail[0],
                        "combined_token_count": combined_tail[0],
                        "standalone_token_sha256": standalone_tail[1],
                        "combined_token_sha256": combined_tail[1],
                    },
                }
            )

    return {
        "schema_version": "1.0.0",
        "generation_plan_sha256": generation_plan_sha256,
        "protocol_sha256": protocol_sha256,
        "expected_row_count": len(expected),
        "observed_row_count": len(observed),
        "valid_audio_count": sum(row.get("valid") is True for row in observations),
        "instrumentation_pass_count": sum(row["passed"] for row in validation_rows),
        "all_instrumentation_checks_passed": all(
            row["passed"] for row in validation_rows
        ),
        "rows": validation_rows,
        "request_hash_comparisons": comparisons,
        "evidence_boundary": (
            "Receipt validation checks frozen plan identity, calibrated frontend "
            "previews, serial live request coverage, and token hashes. It does not "
            "prove kernel behavior, deterministic execution, content faithfulness, "
            "waveform identity, or perceptual quality."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    report = build_split_boundary_report(
        plan,
        protocol,
        observations,
        generation_plan_sha256=_sha256(args.plan),
        protocol_sha256=_sha256(args.protocol),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["all_instrumentation_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
