from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_split_boundary_receipts import build_split_boundary_report  # noqa: E402


class SplitBoundaryReceiptTests(unittest.TestCase):
    def fixture(self):
        candidates = ["upstream", "seeded"]
        seeds = [42]
        prompts = {
            "prefix": (["a" * 64], [64]),
            "tail": (["b" * 64], [20]),
            "combined": (["a" * 64, "b" * 64], [64, 20]),
        }
        samples = []
        observations = []
        for candidate in candidates:
            for prompt_id, (hashes, counts) in prompts.items():
                sample_id = f"{candidate}--{prompt_id}--seed-42"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "prompt_id": prompt_id,
                        "seed": 42,
                        "text": prompt_id,
                    }
                )
                request_hashes = (
                    ["c" * 64]
                    if prompt_id == "prefix"
                    else ["d" * 64]
                    if prompt_id == "tail"
                    else ["c" * 64, "d" * 64 if candidate == "seeded" else "e" * 64]
                )
                observations.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "prompt_id": prompt_id,
                        "seed": 42,
                        "requested_text": prompt_id,
                        "valid": True,
                        "frontend_segmentation": {
                            "segment_count": len(hashes),
                            "segments": [
                                {
                                    "normalized_text_sha256": digest,
                                    "text_token_count": count,
                                }
                                for digest, count in zip(hashes, counts)
                            ],
                            "request_count_matches": True,
                        },
                        "vllm_sampling": {
                            "requests": [
                                {
                                    "status": "complete",
                                    "request_ordinal": index,
                                    "output_token_count": 10,
                                    "output_token_sha256": digest,
                                }
                                for index, digest in enumerate(request_hashes, start=1)
                            ]
                        },
                    }
                )
        plan = {"candidate_ids": candidates, "seeds": seeds, "samples": samples}
        protocol = {
            "calibration_expectations": {
                prompt_id: {
                    "segment_count": len(hashes),
                    "segment_hashes": hashes,
                    "segment_token_counts": counts,
                }
                for prompt_id, (hashes, counts) in prompts.items()
            }
        }
        return plan, protocol, observations

    def build(self, plan, protocol, observations):
        return build_split_boundary_report(
            plan,
            protocol,
            observations,
            generation_plan_sha256="f" * 64,
            protocol_sha256="e" * 64,
        )

    def test_validates_receipts_and_expected_hash_relations(self) -> None:
        plan, protocol, observations = self.fixture()
        report = self.build(plan, protocol, observations)
        self.assertTrue(report["all_instrumentation_checks_passed"])
        comparisons = {
            row["candidate_id"]: row for row in report["request_hash_comparisons"]
        }
        self.assertTrue(comparisons["upstream"]["prefix"]["token_sha256_equal"])
        self.assertFalse(comparisons["upstream"]["tail"]["token_sha256_equal"])
        self.assertTrue(comparisons["seeded"]["tail"]["token_sha256_equal"])

    def test_rejects_missing_and_duplicate_observations(self) -> None:
        plan, protocol, observations = self.fixture()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self.build(plan, protocol, observations[:-1])
        duplicated = observations + [copy.deepcopy(observations[0])]
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            self.build(plan, protocol, duplicated)

    def test_fails_check_on_calibration_drift(self) -> None:
        plan, protocol, observations = self.fixture()
        observations[0]["frontend_segmentation"]["segments"][0][
            "normalized_text_sha256"
        ] = "0" * 64
        report = self.build(plan, protocol, observations)
        self.assertFalse(report["all_instrumentation_checks_passed"])
        self.assertFalse(
            report["rows"][0]["checks"]["segment_hashes_match_calibration"]
        )

    def test_rejects_unexpected_request_shape(self) -> None:
        plan, protocol, observations = self.fixture()
        combined = next(row for row in observations if row["prompt_id"] == "combined")
        combined["vllm_sampling"]["requests"] = combined["vllm_sampling"][
            "requests"
        ][:1]
        with self.assertRaisesRegex(ValueError, "unexpected request counts"):
            self.build(plan, protocol, observations)


if __name__ == "__main__":
    unittest.main()
