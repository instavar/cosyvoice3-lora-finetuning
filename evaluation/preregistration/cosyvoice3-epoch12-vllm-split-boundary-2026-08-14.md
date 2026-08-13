# CosyVoice3 vLLM split-boundary preregistration

Date: 2026-08-14 Asia/Singapore, cross-checked against GitHub Actions server
timestamp `2026-08-13T19:10:16Z` from run `31734483431`.

## Question

Does the earlier long-form request-seed association follow the number of
CosyVoice frontend synthesis requests rather than text length alone, and does a
request-local seed reset make an identical second segment reproduce its
standalone token sequence?

## Frozen files

- Prompt specification:
  `cosyvoice3-epoch12-vllm-split-boundary-prompt-spec-2026-08-14.json`, file
  SHA-256
  `98a24807af374649eb93a58aa11a7650f9e8e47a532e51118b4d9b1a6da27fb8`,
  canonical SHA-256
  `22ca95b19601dcb72af636cacb8c25ea54d4fff3e940a664726748a68aa1f576`.
- Generation plan:
  `cosyvoice3-epoch12-vllm-split-boundary-plan-2026-08-14.json`, SHA-256
  `f3cf6fcac69a278708509b84f48c2459cafaa0794cb95414476c9dab890905ee`.
- Protocol:
  `cosyvoice3-epoch12-vllm-split-boundary-protocol-2026-08-14.json`, SHA-256
  `15862e5dd260dc4de3586101c23d4a4ba7a5e08852801e48bae15c7de63ef6c6`.

The producing revision is
`a4df080810c225f54a4e5142eacc403cc27f310e`. The evaluator revision is
`2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0. The runtime is
vLLM 0.15.1.

## Non-generative calibration

Before any synthesis, the actual Fun-CosyVoice3-0.5B frontend preview produced:

| Prompt | Segment count | Token counts | Segment hashes |
| --- | ---: | --- | --- |
| Prefix | 1 | 64 | `b009798c...6006` |
| Tail | 1 | 20 | `066f0d02...112c` |
| Combined | 2 | 64, 20 | prefix hash, tail hash |

The combined input is the exact prefix followed immediately by the exact tail.
It deliberately has no whitespace after the prefix period. That mechanical
form preserves identical normalized segments and is not intended as a
naturalness or copywriting prompt. Calibration inspected no generated audio or
output tokens.

## Matrix and process isolation

The frozen matrix has two candidates, three prompts, and three seeds for 18
rows. The profiles are unchanged upstream vLLM and request-seeded vLLM. Every
row must run through `--sample-id` in a fresh process. This prevents a prior
row from advancing the process-global generator and contaminating a matched
comparison.

Every observation must retain the frontend preview receipt, the live vLLM
request receipts, output-token counts and hashes, exact WAV hash, timing,
memory, and validity result. All failures remain in place and are not rerun as
replacements.

## Hypotheses and falsifiers

1. Live vLLM request count equals the calibrated segment count for every row.
2. Within each profile and seed, combined request one equals standalone prefix
   in output-token count and hash.
3. Within request-seeded and each seed, combined request two equals standalone
   tail in output-token count and hash because both requests receive the same
   request-local seed.
4. Within upstream and each seed, combined request two may differ from
   standalone tail because request one has advanced the process-global
   generator.

Any missing row, calibration drift, incomplete request receipt, request-count
mismatch, artifact drift, or silent replacement falsifies the instrumentation
claim. Token equality is diagnostic evidence only. It does not prove waveform
identity, deterministic execution, runtime equivalence, or quality.

## Primary gate and secondary evidence

Content faithfulness remains primary. Any unevaluable row, requested-text WER
above 0.10, repeated four-gram excess above 0.05, or at least two
reference-exclusive four-gram hits blocks promotion. Speed, memory, speaker
similarity, signal validity, prosody proxies, and any hash relationship cannot
override that gate.

After generation, verify artifact fingerprints again and produce complete
coverage, audio probes, faster-whisper ASR, SpeechBrain ECAPA similarity,
prosody proxies, content-faithfulness evidence, and explicit request-hash
comparisons. Human judgments of accent, cadence, fatigue, naturalness, or
preference are outside this mechanical probe.

## Scope

Direct results apply only to the frozen epoch-12 export, reference, three probe
texts, three seeds, vLLM 0.15.1, and recorded host. The method generalizes to
other segmented generation systems: bind frontend segmentation, isolate
process state, capture request-level evidence, and retain falsifying rows. The
specific hashes and any observed generator behavior do not generalize without
replication.
