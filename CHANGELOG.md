# Changelog

## Unreleased

### Bug fixes

- Restore CosyVoice's expected `embed_tokens` access path after loading a PEFT
  adapter during inference.
- Add a `--vllm-dir` inference path that safely merges the LoRA adapter into the
  base Qwen2 model, exports fresh merged weights, registers the CosyVoice vLLM
  model, and enables vLLM decoding.
- Refuse existing vLLM export directories to prevent silently loading weights
  produced from a different adapter.

## v0.1.0 — First public CosyVoice3 LoRA fine-tuning pipeline

### What's included

- **`tools/train_cosyvoice3_lora.py`** — PEFT LoRA training with DeepSpeed Stage 2 (305 lines). Applies LoRA to Qwen2 attention projections, freezes base model, saves adapter-only checkpoints (~8 MB vs ~4 GB for full SFT).
- **`tools/infer_cosyvoice3_lora.py`** — LoRA inference: loads adapter on top of pretrained CosyVoice3 model.
- **`tools/generate_cosyvoice3_samples.py`** — Batch evaluation with seed retry logic, min-duration gating, and metadata.json output.
- **`tools/infer_cosyvoice3_hybrid.py`** — Hybrid text normalization inference combining wetext and ttsfrd frontends with automatic routing.
- **`tools/prune_deepspeed_checkpoints.py`** — Checkpoint retention and pruning by metric or age.
- **`patches/`** — Upstream patches adding CV loss monitoring and overfitting detection to the CosyVoice training loop.
- **`configs/`** — DeepSpeed Stage 2 and training YAML configs with recommended LoRA hyperparameters.

### Diagnosis of failed full-SFT run

The README documents why the first CosyVoice3 run (full SFT, 506M params, 41 epochs) did not reach production quality:
- Massive overfitting after epoch 1 (CV loss rose from 2.900 to 3.063)
- 174 GB of checkpoints, grad_norm explosion (4 to 19+)
- Long-form generation required 11-18 seed retries

### LoRA rerun results

The corrected LoRA run (2.16M trainable params, LR 5e-5) showed:
- Best checkpoint at epoch 12 (CV loss 3.044)
- 12 epochs of improvement before divergence (vs 1 epoch for full SFT)
- 8.3 MB checkpoints (480x smaller than full SFT)
- Grad norm stable at 1.4-4.0 (no explosion)
- 1.8x faster throughput (6.9 vs 3.8 samples/sec)

### Known pitfalls (7 documented)

From production fine-tuning on IMDA NSC FEMALE_01 (17K utterances, RTX 3090 Ti):
1. Do not use full SFT — use LoRA
2. Do not trust training loss alone — CV loss is the only reliable signal
3. Upstream has no early stopping
4. Checkpoint size is a smell (GB = full SFT, MB = LoRA)
5. PEFT wrapping breaks `embed_tokens` access path
6. Parquet data prep can produce `numpy.object_` embeddings
7. Prompt formatting sensitivity with `<|endofprompt|>` placement

### Related blog posts

- [CosyVoice LoRA Fine-Tuning — What Worked, What Didn't](https://instavar.com/blog/ai-production-stack/CosyVoice_LoRA_Finetuning_Production_Guide)
- [CosyVoice 2 vs 3 — Voice Cloning Quality Compared](https://instavar.com/blog/ai-production-stack/CosyVoice2_vs_CosyVoice3_IMDA_NSC_FEMALE_01)
- [Best Open-Source TTS Models for Production in 2026](https://instavar.com/blog/ai-production-stack/Best_Open_Source_TTS_Models_Production_2026)
