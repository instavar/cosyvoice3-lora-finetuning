# Evaluator 0.45 resume instrumentation

Date: 2026-08-14, Asia/Singapore

## Change

Future guarded CosyVoice3 LoRA checkpoints write sidecar-bound
`optimizer-state.pt` and `scheduler-state.pt` files in addition to the combined
`runtime-state.pt` loader source. The five evaluator 0.45 roles now map as
follows:

| Evaluator role | CosyVoice checkpoint member |
| --- | --- |
| `model_state` | `adapter_model.safetensors` |
| `optimizer_state` | `optimizer-state.pt` |
| `scheduler_state` | `scheduler-state.pt` |
| `trainer_state` | `training-state.json` |
| `rng_state` | `runtime-state.pt` |

The runtime file also contains optimizer, scheduler, AMP scaler, and RNG state
because the existing loader consumes that combined structure. The decomposed
optimizer and scheduler copies make the required file roles independently
addressable without breaking the old loader. Older guarded checkpoints that
contain only the combined file remain resumable under their original sidecars.

`evaluator_lora_artifact_paths(...)` requires the new files, rechecks the live
sidecar manifest, and rejects ambiguous adapter files or cross-role hardlinks.

## OOD and compatibility controls

Dependency-free tests cover:

- the five-role mapping for a new checkpoint;
- a legacy combined-only checkpoint that still resumes but cannot claim 0.45
  readiness;
- ambiguous adapter model files;
- cross-role optimizer and scheduler hardlinks;
- source-level confirmation that the trainer writes both decomposed files;
- sidecar and continuation-state byte drift;
- interruption during partial publication; and
- unsupported DeepSpeed, multi-rank, and completed-target continuation.

The public contract workflow pins evaluator revision
`29c38cfd86b889abc8b79df063c817dd8f684903` and verifies its schema 1.1 receipt
builder and comparison APIs.

## Evidence boundary

No model training or GPU test was run for this instrumentation change. It
establishes repository behavior and dependency-free contract coverage only.
Historical checkpoints and the selected epoch-12 adapter predate the new files
and schema 1.1 live-conditioning receipts. They are not upgraded.

A stronger comparison must preregister and fingerprint the Base artifact,
dataset-lineage receipt, training controls, and initial state, then build
independent uninterrupted and interrupted-resumed receipts before inspecting
the outcome. Both runs must reach the same target update and the resumed run
must bind an observed interruption before that target.

Even a passing comparison establishes byte equality only for the declared
files. It does not prove trainer semantics, hidden floating-point equivalence,
adaptation benefit, perceptual quality, vLLM equivalence, or distributed resume.
