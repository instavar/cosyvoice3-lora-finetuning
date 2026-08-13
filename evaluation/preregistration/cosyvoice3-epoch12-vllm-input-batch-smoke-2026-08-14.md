# CosyVoice3 vLLM input-batch receipt smoke preregistration

Date: 2026-08-14 Asia/Singapore, cross-checked against GitHub Actions server
timestamp `2026-08-13T18:55:32Z` from run `31733246713`.

## Purpose

Verify corrected request-receipt capture on the live vLLM 0.15.1
`GPUInputBatch` path before using those receipts in a split-boundary study.
The first smoke remains preserved as negative evidence and is not replaced.

## Frozen slice

The two-row plan is
`cosyvoice3-epoch12-vllm-input-batch-smoke-plan-2026-08-14.json`, SHA-256
`2143c7db8fcd0107aceec92d4fa037b076d8b5a159097fd181fe69298171da5b`.
It uses the `neutral-brief` prompt and seed 42 for an unchanged upstream
profile and a request-seeded profile, each in a fresh process. Failed or
invalid outputs remain in this new evidence root and are not replaced.

The producing revision is
`d86e5c5149ec95c688d4e17dcd77ccb560d50a54`. The evaluator revision is
`2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0.

## Artifact boundary

Both rows use the retained derived vLLM export with tree SHA-256
`624f504ea3558f391279fcbcdc7109a38849ea277bb5ab1ef00323c82a30cc22`
and artifact-set SHA-256
`db7b3b425b4f8b4353c8111d49f57ebbf3f351a04352eadb3adb2ae80a26c866`.
Verify the artifact before and after generation.

## Acceptance and falsifiers

Each row must retain exactly one schema 1.0.0 request receipt captured at
`vllm.v1.worker.gpu_input_batch.InputBatch.add_request`. It must contain the
materialized temperature, top-p, top-k, request limits, a positive output-token
count, and a SHA-256 token hash without token content.

The upstream row must show no request-local generator, identify the
process-global generator path, and record engine configuration seed 0. The
seeded row must record a request-local generator initialized with seed 42.
Engine configuration seed 0 is not the evolving global generator state and
must not be interpreted as a per-request seed.

The smoke fails if the input-batch hook is not observed exactly once, vLLM
version drifts, either row is missing, an invalid row is replaced, artifact
verification fails, or token content is retained.

## Scope

A pass validates only the process-local request-receipt instrumentation on the
frozen runtime and artifact. It does not establish content quality, perceptual
quality, runtime equivalence, deterministic execution, or a causal explanation
for the earlier long-form sampling effect.
