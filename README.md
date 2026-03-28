# CosyVoice3 LoRA Fine-Tuning

LoRA fine-tuning tools for [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) v3 (Fun-CosyVoice3-0.5B). Companion repo for single-speaker voice cloning on a 24GB consumer GPU.

> **Status:** Active development. The first full-SFT run did not reach production quality — this repo contains the corrected LoRA approach with the diagnosis and rerun configuration.

## Why this repo exists

CosyVoice's upstream training code supports full SFT only. LoRA fine-tuning requires:

1. PEFT integration for the Qwen2-based LLM backbone
2. Selective layer freezing with configurable unfreezing
3. LoRA-aware checkpoint save/load (adapters, not full weights)
4. Overfitting detection in the training loop (upstream only saves, never gates)

This repo provides all four, plus evaluation and checkpoint management scripts.

## What's included

```
tools/train_cosyvoice3_lora.py        – LoRA training with PEFT + DeepSpeed Stage 2
tools/infer_cosyvoice3_lora.py        – LoRA inference (loads adapter on top of pretrained)
tools/generate_cosyvoice3_samples.py  – Batch sample generation with seed retry and metadata
tools/infer_cosyvoice3_hybrid.py      – Hybrid text normalization inference (wetext + ttsfrd)
tools/prune_deepspeed_checkpoints.py  – Checkpoint retention/pruning by metric or age
patches/                              – Upstream patches for CV monitoring + overfitting detection
configs/                              – DeepSpeed and training YAML configs
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

## Recommended hyperparameters

These are based on diagnosing a failed full-SFT run (see Diagnosis below).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training mode | **LoRA** (not full SFT) | Prevents catastrophic forgetting, 100x smaller checkpoints |
| LoRA rank (r) | 16 | Good balance of capacity and efficiency for 506M model |
| LoRA alpha | 64 | Standard 4x rank scaling |
| LoRA targets | q_proj,k_proj,v_proj,o_proj | Attention projections only |
| Learning rate | **5e-5** | LoRA adapts faster than full SFT (which used 1e-5) |
| Max epochs | **20** | Best region was epoch 1 for full SFT; LoRA converges faster |
| Grad accumulation | 2 | Effective batch size of 2 |
| Early stopping patience | 3 | Hard stop when CV loss diverges |
| DeepSpeed | Stage 2 (no CPU offload) | Fits in 24GB VRAM for 506M model |

## Data preparation

CosyVoice3 requires data in parquet format with speech tokens and speaker embeddings. See the upstream `examples/libritts/cosyvoice3/run.sh` stages 0-3, or use our `prepare_female01_data.sh` as a reference for converting GPT-SoVITS slicer output.

The pipeline:
1. Create `wav.scp`, `text`, `utt2spk`, `spk2utt`, `instruct` files
2. Extract campplus speaker embeddings (`tools/extract_embedding.py`)
3. Extract speech tokens (`tools/extract_speech_token.py`)
4. Create parquet files (`tools/make_parquet_list.py --instruct`)

## Diagnosis: why the first run failed

The first CosyVoice3 run on IMDA NSC FEMALE_01 (17K utterances, RTX 3090 Ti) did not reach production quality. Root causes:

### 1. Full SFT instead of LoRA

The run used upstream `cosyvoice/bin/train.py` (full 506M parameter training) instead of the LoRA script. This caused catastrophic forgetting of the pretrained model's generalization ability. Evidence: 174GB of full-weight checkpoints.

### 2. Massive overfitting after epoch 1

| Epoch | CV Loss | Diagnosis |
|-------|---------|-----------|
| 0 | 3.012 | Baseline |
| 1 | **2.900** | Best (only epoch that improved) |
| 2 | 2.918 | Diverging |
| 3 | 3.046 | Worse than start |
| 4-41 | (not gated) | Continued training without stopping |

Training loss kept dropping (3.5 to 1.2 by epoch 4) with grad_norm rising from 4 to 19 — textbook memorization.

### 3. Long-form generation fragility

At the best checkpoints (epochs 8-10), generating >10 second audio required 11-18 seed attempts. Most seeds produced truncated or garbled output. The model's autoregressive decoder was hitting EOS prematurely.

### 4. Exhaustive evaluation couldn't save the run

12+ evaluation directories tried: different epochs (1, 8, 10, 30, 40), different prompts, different seeds, sentence splitting, shorter prompts. None produced production quality.

## Pitfalls

1. **Do not use full SFT** — the upstream `cosyvoice/bin/train.py` trains all 506M parameters. Use `tools/train_cosyvoice3_lora.py` instead.

2. **Do not trust training loss alone** — CV loss is the only reliable signal. Training loss will keep dropping even as quality degrades.

3. **The upstream has no early stopping** — our `executor.py` patch adds CV monitoring but only warns. You must manually stop training or implement hard stopping.

4. **Checkpoint size is a smell** — full SFT checkpoints are ~4GB each. LoRA adapters should be ~50MB. If your checkpoints are GB-sized, you're doing full SFT.

5. **Prompt formatting matters** — CosyVoice3 is sensitive to `<|endofprompt|>` placement. Fix a single template for all comparisons.

6. **Seed retry rate indicates quality** — if generating 10s audio needs >5 seed attempts, the checkpoint is bad.

7. **The Qwen pretrain path must be CosyVoice-BlankEN** (not a generic Qwen path). This is the tokenizer/config directory inside the pretrained model.

## Evaluation

```bash
# Generate samples from a LoRA checkpoint
python tools/infer_cosyvoice3_lora.py \
    --pretrained-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --lora-dir exp/your_run/lora/epoch_2_whole \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --text "Text to synthesize" \
    --out-wav output.wav

# Batch evaluation with seed retry
python tools/generate_cosyvoice3_samples.py \
    --tags epoch_2_whole,epoch_4_whole \
    --prompt-wav /path/to/prompt.wav \
    --prompt-text "Text matching the prompt audio" \
    --out-dir samples/eval \
    --min-seconds 10
```

## Cross-reference

| Model | Repo | Training | Result |
|-------|------|----------|--------|
| CosyVoice3 | This repo | LoRA (corrected) | In progress |
| Qwen3-TTS | [instavar/qwen3-tts-lora-finetuning](https://github.com/instavar/qwen3-tts-lora-finetuning) | LoRA | Production-ready (epoch 10, scale 0.3-0.35) |
| IndexTTS2 | [instavar/indextts2-finetuning](https://github.com/instavar/indextts2-finetuning) | Full SFT | Production-ready (step 14000) |

## License

Tools in this repo are MIT licensed. CosyVoice itself is under the [CosyVoice Community License](https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE).
