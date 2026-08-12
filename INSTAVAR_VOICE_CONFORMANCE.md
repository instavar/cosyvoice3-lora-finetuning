# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) PyTorch LoRA recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to commit `31bc7b7b97bb7a291a44fc1591620960c2cc2d2d`.

The executable recipe proves that the external CosyVoice checkout equals its recorded upstream revision plus exactly the companion patch set. It applies explicit `MAX_EPOCH` and `LEARNING_RATE` values after loading the full model graph, audits grouped splits, selects one safe adapter, reloads it in a fresh process, evaluates the frozen plan, and packages evidence. A DeepSpeed run must use the same learning rate in its JSON optimizer configuration. The merged vLLM path remains outside this PyTorch lifecycle and needs a separate matched equivalence run.

Capability schema 1.2 records each LoRA lifecycle stage separately and preserves the six invalid emotion-control rows as a negative lifecycle result. It also names the exact blockers for a matched base-model comparison.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

The cross-validation monitor warns when its configured metric stops improving. Pass `--early-stop-on-cv-overfit` to stop all distributed workers together at the epoch boundary after the configured patience is exhausted. The option is off by default, so existing training jobs retain their prior behavior.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
