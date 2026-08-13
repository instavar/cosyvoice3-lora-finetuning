# CosyVoice3 Base and epoch-12 long-form preregistration

Date: 2026-08-13

Status: frozen before candidate generation. This document binds one focused
same-conditioning comparison. It is not a result report.

## Planned comparison

- runner revision: `e9775c563bcdafc2c229265f6493580191c9977b`
- evaluator revision: `982367abc7837cb6da5ebb94192c9642dea62fce`
- candidates: `cosyvoice3-base` and `cosyvoice3-epoch12`
- prompt: `cadence-two-minute`
- seed: `20260812`
- generation route: `inference_zero_shot` for both candidates
- speed: `1.0`
- text frontend: enabled
- precision: FP16
- device: one NVIDIA RTX 3090 Ti

The unchanged Base and epoch-12 LoRA candidates must receive the same prompt
text, retained reference WAV, exact reference transcript, seed, speed,
frontend, and upstream generation route. Base mode must load no adapter.
Adapter mode must load the exact epoch-12 PEFT artifact.

## Frozen plan identities

- prompt pack `instavar-singapore-english` version 1.2.0 SHA-256:
  `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- generation-plan file SHA-256:
  `266e384bc4f874536e85be4fd5dc143fa1a65590d5f7d126c93bef2c84a6cfe6`
- canonical generation-plan SHA-256:
  `ffbd2edb68cf45e63a5e295ffd1a117cffdbd80d11857f5e9fc1a1b8dba6ba3c`
- speaker-reference catalog file SHA-256:
  `c915afb7bf9616840bf5a07c0742e8176ee3faae32435c9397b8887bea005a19`
- canonical speaker-reference catalog SHA-256:
  `a25fc55fd8e4565192c0cc94dfdeda338f8bc2614c072623d82b31456d2d8f4e`
- speaker-reference assignment file SHA-256:
  `2d56cbced0196b4035bdc8ada887b1623865103371a6fae5cd84bf0e73d2fb72`
- canonical speaker-reference assignment SHA-256:
  `0fd99ca68a7cb803bcd7fe8ed553a265a31bb8d847ecbb66c39e0a2d5fe266a1`

The assignment freezes `female01-reference` for both candidates before
generation. It uses the evaluator's fixed mean-of-cosines aggregation policy.

## Artifact identities

- retained reference audio SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- retained reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- upstream pretrained-model tree: 7,451,850,238 bytes, 55 files, SHA-256
  `bca8f91aa0b8eab361be85256de4b8d7f56e34ceeb8fd62feaef0298f40ededc`
- epoch-12 adapter tree: 8,682,447 bytes, 3 files, SHA-256
  `8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`
- Base candidate artifact-set SHA-256, including generation reference:
  `7c9fb1b5d255cd899bc2b562b1f0b58540f73e24f3902ec5276a535183ed7e54`
- adapter candidate artifact-set SHA-256, including generation reference:
  `7b4b30762864daaa60a355b9445e9722b393ef0b3c472157f4905e74c9dfb2da`

The artifact-set digests use evaluator revision `982367a` tree and canonical
record fingerprinting. They bind declared bytes, not loader honesty or host
trust.

## Decision rule and claim boundary

The objective pipeline must attempt all nine metrics required by the generation
plan, preserve failures, and report `proves_adaptation_benefit: false` unless a
separate valid decision rule establishes otherwise. No objective composite
score or automatic winner is defined.

The focused blind pack may route only speaker identity, cadence variation,
long-form monotony, naturalness, artifact severity, and listening fatigue.
Accent fidelity, lexical pronunciation, and emotion obedience are outside this
prompt's criterion scope. No rating may be inferred or fabricated.

Possible outcomes are deliberately non-directional. Runtime validity, WER,
speaker similarity, signal proxies, duration, memory, and speed may differ in
either direction. Any notable delta must be treated as one-pair evidence that
prioritizes replication and blinded review, not as a general quality or causal
claim.
