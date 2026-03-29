# CosyVoice3 LoRA Fine-Tuning

LoRA fine-tuning tools for [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) v3 (Fun-CosyVoice3-0.5B). Companion repo for single-speaker voice cloning on a 24GB consumer GPU.

> **Status:** LoRA run completed. Best checkpoint at epoch 12 (CV loss 3.044). Quality evaluation pending.

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

# 5. Run LoRA training
torchrun --nnodes=1 --nproc_per_node=1 \
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
| Max epochs | **20** | Best region is epoch 10-12; stop early |
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

### 1. Do not use full SFT

The upstream `cosyvoice/bin/train.py` trains all 506M parameters. Our first run used this path and produced 174 GB of checkpoints with severe overfitting after epoch 1. Use `tools/train_cosyvoice3_lora.py` instead.

### 2. Do not trust training loss alone

Training loss will keep dropping even as quality degrades. CV loss is the only reliable signal. Our full SFT run had training loss of 1.2 at epoch 4 while CV loss was already worse than epoch 0.

### 3. The upstream has no early stopping

Our `executor.py` patch adds CV monitoring and overfitting warnings, but it only logs — it does not stop training. You must manually stop or implement hard stopping. Our LoRA run trained 200 epochs when 20 would have sufficed.

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
