# CosyVoice3 epoch-12 four-way runtime preregistration

Date: 2026-08-13

## Question

Does the observed epoch-12 divergence appear when the merged model is
serialized and reloaded, or only when the retained derived export is decoded
through vLLM?

## Frozen conditions

The machine-readable plan is
`cosyvoice3-epoch12-four-way-runtime-plan-2026-08-13.json`, SHA-256
`7cbd70c1b3353e514fa1fdc16bdc1ee0c4e31c16475329dbc13dcd1ae00884ef`.
It contains one neutral Singapore English prompt, seeds 42, 314159, and
20260812, and these conditions:

1. `cosyvoice3-epoch12-adapter-pytorch`: PEFT adapter through PyTorch.
2. `cosyvoice3-epoch12-merged-pytorch`: `merge_and_unload` in memory, then
   PyTorch.
3. `cosyvoice3-epoch12-reloaded-merged-pytorch`: a once-created safetensors
   artifact loaded into a fresh pretrained PyTorch process.
4. `cosyvoice3-epoch12-merged-vllm`: the retained derived export through vLLM.

The source adapter tree SHA-256 is
`8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`.
The persisted merged PyTorch artifact will be created exactly once after this
preregistration, then fingerprinted and reused. Every condition uses the same
pretrained checkpoint, reference WAV, reference transcript, prompt text,
frontend setting, speed, and seed list. Every seed is attempted exactly once.
Failures are retained and are not replaced.

## Primary gate

Content faithfulness remains primary. A condition fails promotion if any
planned row has requested-text WER above 0.10, repeated four-gram excess above
0.05, at least two reference-exclusive four-gram hits, or an unevaluable ASR
row. Speed, speaker similarity, audio validity, prosody proxies, and listening
ratings cannot override a failed content gate.

## Diagnostic interpretation

- If adapter, in-memory merged, and reloaded merged PyTorch emit identical
  audio while vLLM diverges, the bounded evidence points to vLLM decoding after
  serialization.
- If adapter and in-memory merged PyTorch match but reloaded merged PyTorch and
  vLLM diverge in the same direction, serialization or reload behavior remains
  implicated.
- If reloaded merged PyTorch differs from both neighboring conditions, inspect
  the persisted artifact format and loader before interpreting runtime metrics.
- If all four fail the content gate, none is promoted even when the relative
  pattern localizes the defect.

These outcomes are diagnostics, not causal proof. One prompt and three seeds do
not establish general runtime behavior.

## Artifact boundary

The adapter input, persisted merged PyTorch tree, and retained vLLM export must
have separate content-addressed bindings. Both merged representations are
derived artifacts with pinned converter provenance. They must not enter an
exact-artifact `compare-runtimes` claim. Use matched comparisons and record
`proves_runtime_equivalence: false`.

The persisted artifact manifest and loader checks establish declared byte
identity and reject ordinary path substitution. They do not attest which bytes
the framework used internally, host trust, numerical equivalence, or TTS
quality.

## Secondary evidence

For all valid rows, record exact WAV hashes, audio probes, faster-whisper ASR,
SpeechBrain ECAPA speaker similarity against the frozen assignment, prosody
proxies, generation time, real-time factor, and peak GPU allocation. Stage a
counterbalanced blind pack only after objective evidence is complete. Applicable
criteria are speaker identity, Singapore English accent fidelity, naturalness,
and artifact severity.
