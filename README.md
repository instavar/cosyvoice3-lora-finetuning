# CosyVoice3 LoRA Fine-Tuning

LoRA fine-tuning tools for [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) v3 (Fun-CosyVoice3-0.5B). Companion repo for single-speaker voice cloning on a 24GB consumer GPU.

> **Status:** LoRA run completed. Best checkpoint at epoch 12 (CV loss 3.044). Standard PyTorch and merged-weight vLLM 0.15.1 inference were validated end to end on an RTX 3090 Ti on 2026-07-28. Perceptual quality ranking remains pending.

## Why this repo exists

CosyVoice's upstream training code supports full SFT only. LoRA fine-tuning requires:

1. PEFT integration for the Qwen2-based LLM backbone
2. Selective layer freezing with configurable unfreezing
3. LoRA-aware checkpoint save/load (adapters, not full weights)
4. Overfitting detection in the training loop (upstream only saves, never gates)

This repo provides all four, plus evaluation and checkpoint management scripts.

## What's included

```
tools/train_cosyvoice3_lora.py        - LoRA training with PEFT + DeepSpeed Stage 2
tools/infer_cosyvoice3_lora.py        - LoRA inference (loads adapter on top of pretrained)
tools/generate_cosyvoice3_samples.py  - Batch sample generation with seed retry and metadata
tools/infer_cosyvoice3_hybrid.py      - Hybrid text normalization inference (wetext + ttsfrd)
tools/prune_deepspeed_checkpoints.py  - Checkpoint retention/pruning by metric or age
patches/                              - Upstream patches for CV monitoring + overfitting detection
configs/                              - DeepSpeed and training YAML configs
```

## Quick start

```bash
# 1. Clone CosyVoice and apply patches
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git apply ../cosyvoice3-lora-finetuning/patches/*.patch

# 2. Copy tools into the CosyVoice repo
cp ../cosyvoice3-lora-finetuning/tools/*.py tools/

# 3. Install dependencies
pip install peft  # Required for LoRA

# 4. Prepare your data (see Data Preparation below)

# 5. Audit JSONL manifests for the same corpus and split assignment, then train
export INSTAVAR_VOICE_EVAL_DIR=/path/to/instavar-voice-evaluation
../cosyvoice3-lora-finetuning/scripts/run_with_corpus_audit.sh \
  --split train=your_data/audit/train.jsonl \
  --split validation=your_data/audit/validation.jsonl \
  --split test=your_data/audit/test.jsonl \
  --group-field recording_id \
  -- torchrun --nnodes=1 --nproc_per_node=1 \
    tools/train_cosyvoice3_lora.py \
    --train_engine deepspeed \
    --model llm \
    --config examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
    --train_data your_data/train/parquet/data.list \
    --cv_data your_data/dev/parquet/data.list \
    --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
    --model_dir exp/your_run/lora \
    --deepspeed_config configs/ds_stage2_lora.json \
    --lora-r 16 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj
```

## LoRA run results

Trained on IMDA NSC FEMALE_01 (16,535 train / 870 dev utterances), RTX 3090 Ti (24 GB).

### LoRA vs full SFT comparison

| Metric | LoRA run | Previous full SFT | Improvement |
|--------|----------|-------------------|-------------|
| Trainable params | 2.16M (0.44%) | 506M (100%) | 234x fewer |
| Best CV loss | 3.044 (epoch 12) | 2.900 (epoch 1) | - |
| Epochs before overfit | 12 | 1 | 12x more useful training |
| Checkpoint size | 8.3 MB | 4 GB | 480x smaller |
| Total storage (all ckpts) | 1.7 GB | 174 GB | 102x smaller |
| Grad norm (best region) | 1.4-4.0 | 4.1-19+ | No explosion |
| Training speed | 6.9 samples/sec | 3.8 samples/sec | 1.8x faster |
| Peak VRAM | 6.96 GB | 13.08 GB | 47% less |

CV loss numbers are not directly comparable (LoRA trains adapter weights only, changing what the loss measures), but the stability improvement is clear.

### CV loss curve (LoRA run)

```
Epoch  0: 3.211  (baseline)
Epoch  5: 3.072  (improving)
Epoch 10: 3.046  (near best)
Epoch 12: 3.044  <-- best
Epoch 15: 3.053  (diverging)
Epoch 20: 3.062
Epoch 30: 3.122
Epoch 50: 3.197
Epoch 100: 3.336
Epoch 199: 3.472  (severe overfit)
```

Best checkpoint: **epoch 12**. Early stopping at epoch 15 would have been ideal.

## Recommended hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training mode | **LoRA** (not full SFT) | Prevents catastrophic forgetting, 480x smaller checkpoints |
| LoRA rank (r) | 16 | Good balance of capacity and efficiency for 506M model |
| LoRA alpha | 64 | Standard 4x rank scaling |
| LoRA targets | q_proj,k_proj,v_proj,o_proj | Attention projections only |
| Learning rate | **5e-5** | LoRA adapts faster than full SFT (which used 1e-5) |
| Max epochs | **20** | Best region was epoch 10-12 in the recorded run; stop early |
| Grad accumulation | 2 | Effective batch size of 2 |
| Early stopping patience | 3 | Hard stop when CV loss diverges |
| DeepSpeed | Stage 2 (no CPU offload) | Fits in 24 GB VRAM with 7 GB peak |

## Data preparation

CosyVoice3 requires data in parquet format with speech tokens and speaker embeddings. See the upstream `examples/libritts/cosyvoice3/run.sh` stages 0-3.

The pipeline:
1. Create `wav.scp`, `text`, `utt2spk`, `spk2utt`, `instruct` files
2. Extract campplus speaker embeddings (`tools/extract_embedding.py`)
3. Extract speech tokens (`tools/extract_speech_token.py`)
4. Create parquet files (`tools/make_parquet_list.py --instruct`)

**Important:** After step 2, verify embeddings are stored as `numpy.float32` arrays (not Python lists). See pitfall #6 below.

## Known pitfalls

### 1. Avoid full SFT by default for the recorded single-speaker setup

The upstream `cosyvoice/bin/train.py` trains all 506M parameters. Our recorded single-speaker run used this path and produced 174 GB of checkpoints with severe overfitting after epoch 1. Use `tools/train_cosyvoice3_lora.py` for the same adaptation regime. This result does not establish that full SFT fails for every dataset, budget, or regularization strategy.

### 2. Do not trust training loss alone

Training loss can keep dropping even as held-out behavior degrades. In our recorded full-SFT run, training loss reached 1.2 at epoch 4 while CV loss was already worse than epoch 0. Use CV loss for checkpoint control, then use held-out synthesis and blinded listening for quality selection.

### 3. Enable the fork's opt-in early stopping

Our `executor.py` patch adds CV monitoring and overfitting warnings. Pass `--early-stop-on-cv-overfit` to the LoRA trainer to stop at the epoch boundary after the configured patience is exhausted. The option is off by default for backward compatibility. The recorded LoRA run did not have this control and continued to 200 epochs even though the best observed region was much earlier.

### 4. Checkpoint size is a smell

Full SFT checkpoints are ~4 GB each. LoRA adapters should be ~8 MB. If your checkpoints are GB-sized, you are accidentally doing full SFT, not LoRA.

### 5. PEFT wrapping breaks `embed_tokens` access

After applying LoRA via PEFT, the CosyVoice3LM forward path `self.llm.model.model.embed_tokens(...)` breaks because `PeftModel` inserts an extra layer. The training script patches this by proxying `embed_tokens` onto `Qwen2ForCausalLM`:

```python
qwen2_causal = peft_model.model  # Qwen2ForCausalLM
if not hasattr(qwen2_causal, "embed_tokens"):
    qwen2_causal.embed_tokens = qwen2_causal.model.embed_tokens
```

Without this fix, training crashes with `AttributeError: 'Qwen2ForCausalLM' object has no attribute 'embed_tokens'`.

### 6. Embedding dtype in parquet data

The campplus embedding extractor may store embeddings as Python lists in the `.pt` files. When serialized to parquet, these become `numpy.object_` arrays, causing a `TypeError: can't convert np.ndarray of type numpy.object_` crash during training.

Fix: convert embeddings to `numpy.float32` before creating parquet files:

```python
import torch, numpy as np
data = torch.load("utt2embedding.pt", map_location="cpu")
fixed = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k, v in data.items()}
torch.save(fixed, "utt2embedding.pt")
```

### 7. Prompt formatting sensitivity

CosyVoice3 is sensitive to `<|endofprompt|>` placement and text segmentation. Fix a single prompt template for all comparisons. The `instruct` file in the data pipeline should use `You are a helpful assistant.<|endofprompt|>` consistently.

### 8. The Qwen pretrain path is CosyVoice-BlankEN

The `--qwen_pretrain_path` must point to `pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN` (the tokenizer/config directory inside the pretrained model), not a generic Qwen path. This directory is part of the CosyVoice3 download, not a separate Qwen model.

### 9. Upstream `max_epoch` overrides your config

The upstream `cosyvoice3.yaml` sets `max_epoch: 200`. If you pass a custom config YAML, the training script loads the upstream YAML first. Your override must modify the upstream file directly or ensure it is loaded last. Our LoRA run intended 20 epochs but ran all 200.

## Evaluation

### Executable Instavar Voice lifecycle

[`instavar-voice-backend.json`](instavar-voice-backend.json) binds the PyTorch
LoRA path to a five-stage executable recipe. Preflight verifies that the
external CosyVoice checkout equals its recorded upstream commit plus exactly
the four companion patches. The trainer now accepts explicit `--max_epoch` and
`--learning_rate` overrides after loading the full HyperPyYAML model graph. The
lifecycle requires both values, preventing the earlier 20-versus-200 epoch
mismatch and an implicit optimizer-rate mismatch from recurring silently. For
DeepSpeed, the JSON optimizer rate must equal `LEARNING_RATE`.
Set `DEEPSPEED_CONFIG` only when `TRAIN_ENGINE=deepspeed`; the PyTorch DDP path
does not require it.

The lifecycle audits grouped raw splits, writes model output under its unique
work directory, promotes only one exact adapter directory, strips optimizer
state from the inference package, reloads in a fresh process, runs the frozen
evaluation plan, and packages provenance. Validate it with evaluator revision
`6fa431f6ab6bb9867a5fc210a187523012323ecb`. Use the companion tools directly;
do not copy them into the external checkout, because unexpected checkout files
fail provenance verification. A pass covers the PyTorch adapter path only. The
merged vLLM path still requires a separate matched equivalence lifecycle.

### Frozen multi-prompt runtime evaluation

Use `tools/run_evaluation_suite.py` to execute a complete Instavar Voice plan
through one loaded PyTorch adapter. Add `--vllm-dir` to create or reuse a merged
vLLM export and run the same plan through that runtime. The runner uses each
frozen seed exactly once and records a failed row instead of searching for a
replacement seed. This differs intentionally from the exploratory sample
generator, where retries help operators find an audible example.

The runner also rejects implausibly short or silent output. CosyVoice can raise
inside its background LLM thread while the parent call still returns a roughly
0.04-second WAV, so process exit and non-empty audio are not sufficient runtime
evidence.

```bash
python tools/run_evaluation_suite.py \
  --cosyvoice-dir /path/to/CosyVoice \
  --pretrained-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --lora-dir exp/female01/cosyvoice3/llm/lora/epoch_12 \
  --prompt-wav /path/to/reference.wav \
  --prompt-text "The exact reference transcript." \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id cosyvoice3-epoch12-pytorch \
  --runtime-id pytorch \
  --output-dir evaluation/cosyvoice3-epoch12-pytorch
```

For a cross-runtime experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner rejects partial or malformed
bindings. A PyTorch adapter and a merged vLLM export must be recorded as
`exact` and `derived` respectively, so the shared evaluator will not treat
conversion provenance as exact artifact identity.

The early-stop option now synchronizes its decision across all initialized
training ranks with an all-reduce before any rank leaves the epoch loop. Run
the bounded control-plane smoke with:

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/check_distributed_early_stop.py \
  --output-dir evaluation/distributed-early-stop-smoke
```

That smoke proves rank agreement in the control helper. It does not replace a
real multi-rank CosyVoice training reproduction.

```bash
# Generate samples from a LoRA checkpoint
python tools/infer_cosyvoice3_lora.py \
    --pretrained-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --lora-dir exp/your_run/lora/epoch_12_whole \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --text "Text to synthesize" \
    --out-wav output.wav

# Batch evaluation with seed retry
python tools/generate_cosyvoice3_samples.py \
    --tags epoch_10_whole,epoch_12_whole,epoch_14_whole \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --out-dir samples/eval \
    --min-seconds 10
```

The inference helper restores CosyVoice's expected `embed_tokens` attribute after
PEFT wraps the Qwen2 model. This prevents the
`Qwen2ForCausalLM has no attribute embed_tokens` failure.

### vLLM inference with a LoRA checkpoint

CosyVoice's vLLM integration does not currently pass a per-request LoRA adapter
to vLLM. Use the merged-weight path instead: load the PEFT adapter, merge it into
the base Qwen2 model, export that merged model for vLLM, and then run the normal
CosyVoice pipeline.

```bash
python tools/infer_cosyvoice3_lora.py \
    --pretrained-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --lora-dir exp/your_run/lora/epoch_12_whole \
    --vllm-dir exp/your_run/vllm/epoch_12_merged \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --text "Text to synthesize" \
    --out-wav output-vllm.wav
```

The `--vllm-dir` must be a new path. Refusing an existing directory prevents an
export from a different adapter from being reused silently. The original LoRA
checkpoint is not modified. This path uses more disk space than runtime adapter
loading because it exports merged LLM weights.

Use a vLLM and Transformers combination supported by your checked-out CosyVoice
revision. Current upstream documentation supports vLLM 0.11.x or newer with the
V1 engine, or the legacy vLLM 0.9.0 path. Untested intermediate versions may not
be compatible.

The verified local combination was Python 3.10.19, PyTorch 2.9.1+cu128,
Transformers 4.57.6, vLLM 0.15.1, PEFT 0.18.1, NumPy 1.26.4, and TorchCodec
0.9.0. A fresh merged export produced a valid 24 kHz mono WAV with 7.08 seconds
of audio and an RTF of 0.178 on an RTX 3090 Ti. This verifies execution and
artifact validity for that version set. It does not establish perceptual quality
or compatibility with every newer vLLM release.

After verifying that an export belongs to the intended adapter, it can be reused
without loading and merging the LoRA again:

```bash
python tools/infer_cosyvoice3_lora.py \
    --pretrained-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --lora-dir exp/your_run/lora/epoch_12_whole \
    --vllm-dir exp/your_run/vllm/epoch_12_merged \
    --reuse-vllm-dir \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --text "Text to synthesize" \
    --out-wav output-vllm-reused.wav
```

Do not use process exit status alone as the success criterion. CosyVoice can
raise an exception in its background LLM thread while the parent process still
writes a very short WAV and exits successfully. Check the log for thread
exceptions, then validate sample rate, duration, frame count, and non-trivial
audio level. The verified vLLM sample had 169,920 frames, peak amplitude 0.798,
and RMS 0.124.

## Diagnosis: why the first run failed

The first CosyVoice3 run on IMDA NSC FEMALE_01 (17K utterances, RTX 3090 Ti) used **full SFT instead of LoRA** and did not reach production quality.

### 1. Full SFT caused catastrophic forgetting

All 506M parameters were trained via upstream `cosyvoice/bin/train.py`. Evidence: 174 GB of full-weight checkpoints at ~4 GB each.

### 2. Massive overfitting after epoch 1

| Epoch | CV Loss | Diagnosis |
|-------|---------|-----------|
| 0 | 3.012 | Baseline |
| 1 | **2.900** | Best (only epoch that improved) |
| 2 | 2.918 | Diverging |
| 3 | 3.046 | Worse than start |
| 4-41 | (not gated) | Continued without stopping |

### 3. Long-form generation fragility

At epochs 8-10, generating >10 second audio required 11-18 seed attempts. The model's autoregressive decoder was hitting EOS prematurely on most seeds.

### 4. Exhaustive evaluation failed

12+ evaluation directories tried different epochs (1, 8, 10, 30, 40), prompts, seeds, sentence splitting, and shorter prompts. None produced production quality.

## Cross-reference

| Model | Repo | Training | Result |
|-------|------|----------|--------|
| CosyVoice3 | This repo | LoRA | Best at epoch 12; quality eval pending |
| Qwen3-TTS | [instavar/qwen3-tts-lora-finetuning](https://github.com/instavar/qwen3-tts-lora-finetuning) | LoRA | Production-ready (epoch 10, scale 0.3-0.35) |
| IndexTTS2 | [instavar/indextts2-finetuning](https://github.com/instavar/indextts2-finetuning) | Full SFT | Production-ready (step 14000) |

## Related blog posts

- [CosyVoice LoRA Fine-Tuning — What Worked, What Didn't](https://instavar.com/blog/ai-production-stack/CosyVoice_LoRA_Finetuning_Production_Guide)
- [CosyVoice 2 vs 3 — Voice Cloning Quality Compared](https://instavar.com/blog/ai-production-stack/CosyVoice2_vs_CosyVoice3_IMDA_NSC_FEMALE_01)
- [Best Open-Source TTS Models for Production in 2026](https://instavar.com/blog/ai-production-stack/Best_Open_Source_TTS_Models_Production_2026)
- [TTS Model Decision Tree (2026)](https://instavar.com/blog/ai-production-stack/TTS_Model_Decision_Tree_2026)

## License

Tools in this repo are Apache-2.0 licensed. CosyVoice itself is under the [CosyVoice Community License](https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE).

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) records the validated PyTorch adapter and merged-weight vLLM paths, while keeping direct vLLM LoRA loading explicitly unsupported. It also freezes the shared objective and blinded-listening criteria that remain necessary before a perceptual promotion decision. CI validates the manifest against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation).

The lifecycle preserves invalid generations as explicit rows, then uses
evaluator revision `6fa431f6ab6bb9867a5fc210a187523012323ecb` to bind timing,
duration, and peak-memory fields to the frozen plan and live output audio. Use
the packaged `objective-observations.json`, not the raw generation file, for a
version 1.1 runtime comparison.

The pinned evaluator also provides schema 1.3 frozen speaker-reference
assignments, fixed per-reference aggregation, and embedding-value binding. A
producer must commit or otherwise timestamp the assignment plan before
generation for stronger chronology evidence. This companion does not bundle a
speaker encoder or execute that external stage. Runtime-bound observations alone
are not speaker-quality evidence.
