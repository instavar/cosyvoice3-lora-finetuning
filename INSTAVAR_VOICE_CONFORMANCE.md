# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) PyTorch LoRA recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to commit `8feadf7bbda75abe1c305c63e362c41b86451cda`.

The executable recipe proves that the external CosyVoice checkout equals its recorded upstream revision plus exactly the companion patch set. It applies explicit `MAX_EPOCH` and `LEARNING_RATE` values after loading the full model graph, audits grouped splits, selects one safe adapter, reloads it in a fresh process, evaluates the frozen plan, packages evidence, and publishes it through a content-addressed external retention contract. A DeepSpeed run must use the same learning rate in its JSON optimizer configuration. The merged vLLM path remains outside this PyTorch lifecycle and needs a separate matched equivalence run.

For one-process `torch_ddp`, the lifecycle also enables the [guarded epoch-boundary continuation contract](README.md#guarded-epoch-boundary-continuation). It binds the effective config, model and prepared-data bytes, patched upstream sources, runtime, output identity, adapter, optimizer, scheduler, scaler, RNG, progress, and CV monitor history. An exact newest owned epoch package and explicit trust are required before continuation state loads. Inference adapters and guarded continuation packages are separate artifact types. DeepSpeed, multi-rank, and multi-worker guarded resume fail closed rather than inheriting an unsupported claim.

Preflight excludes the work, source, upstream, model-dependency, base-checkpoint, and prepared-data trees from the persistence destination. It probes fsynced no-overwrite hard-link publication and binds the resolved path, filesystem device, and directory inode through packaging. This establishes dependency-free retention mechanics, not a real retained package, backup, restore, rights approval, runtime equivalence, or complete hostile-filesystem defense.

Capability schema 1.2 records each LoRA lifecycle stage separately and preserves
the six invalid rows assigned to emotion-control categories as a negative
lifecycle result. A later audit established that the historical runner silently
ignored their instructions, so those rows do not support an emotion-control
claim. The corrected route uses `inference_instruct2` and remains
repository-declared until a real-model rerun. The manifest also names the exact
blockers for a matched base-model comparison.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

The cross-validation monitor warns when its configured metric stops improving. Pass `--early-stop-on-cv-overfit` to stop all distributed workers together at the epoch boundary after the configured patience is exhausted. The option is off by default, so existing training jobs retain their prior behavior.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
