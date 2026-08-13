# CosyVoice3 epoch-12 three-way runtime preregistration

Date: 2026-08-13

## Question

Does the observed epoch-12 vLLM content drift first appear when the PEFT adapter
is merged, or only after the merged model is exported and decoded through
vLLM?

## Frozen conditions

The machine-readable plan is
`cosyvoice3-epoch12-three-way-runtime-plan-2026-08-13.json`. It contains one
neutral Singapore English prompt, seeds 42, 314159, and 20260812, and these
conditions:

1. `cosyvoice3-epoch12-adapter-pytorch`: PEFT adapter on the PyTorch route.
2. `cosyvoice3-epoch12-merged-pytorch`: the same adapter merged in memory with
   `merge_and_unload`, followed by PyTorch decoding.
3. `cosyvoice3-epoch12-merged-vllm`: the retained derived export decoded by
   vLLM.

Every condition uses the same pretrained checkpoint, adapter, reference WAV,
reference transcript, prompt text, seed list, frontend setting, speed, and
minimum-audio checks. Each seed is attempted exactly once. Failed rows are
retained and are not replaced.

## Primary gate

The content-faithfulness report is primary. A condition fails promotion if any
planned row has requested-text WER above 0.10, repeated four-gram excess above
0.05, at least two reference-exclusive four-gram hits, or an unevaluable ASR
row. Runtime speed, speaker similarity, prosody proxies, and listening ratings
cannot override a failed content gate.

## Diagnostic interpretation

- If merged PyTorch tracks adapter PyTorch while vLLM diverges, the evidence
  points downstream of the in-memory merge. Export serialization and vLLM
  decoding remain jointly implicated.
- If merged PyTorch diverges from adapter PyTorch and tracks vLLM, the evidence
  points to merge behavior before runtime selection.
- If all three conditions differ materially, the result remains confounded by
  decoding sensitivity or nondeterminism and requires a stronger reproduction.
- If all three fail the content gate, none is promoted even if the relative
  pattern helps localize the defect.

These patterns are diagnostic, not causal proof. Three seeds from one short
prompt cannot establish general runtime behavior.

## Artifact and equivalence boundary

The adapter directory has an exact source artifact binding. The retained vLLM
export has a derived artifact binding with recorded conversion provenance. The
in-memory merged PyTorch model has no persistent post-merge artifact tree, so
this experiment cannot use exact-artifact runtime-equivalence semantics. It
must use matched comparisons and explicitly record
`proves_runtime_equivalence: false`.

Exact WAV equality is recorded by seed as a diagnostic only. Equality would
not establish semantic equivalence, and inequality would not identify the
cause.

## Secondary evidence

For all valid rows, record audio probes, faster-whisper ASR, SpeechBrain ECAPA
speaker similarity against the frozen reference assignment, prosody proxies,
generation time, real-time factor, peak GPU allocation, and exact WAV hashes.
Stage a blind listening pack only after the objective artifacts are complete.
The applicable listening criteria are speaker identity, Singapore English
accent fidelity, naturalness, and artifact severity.
