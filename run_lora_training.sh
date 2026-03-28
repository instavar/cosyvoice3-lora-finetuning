#!/usr/bin/env bash
set -euo pipefail

# CosyVoice3 LoRA Fine-Tuning Run Script
# Usage: ./run_lora_training.sh
#
# Prerequisites:
#   - CosyVoice repo cloned and patches applied
#   - cosyvoice conda env activated
#   - FEMALE_01 dataset prepared as parquet
#
# This script uses the LoRA training path (tools/train_cosyvoice3_lora.py)
# NOT the upstream full SFT path (cosyvoice/bin/train.py)

COSYVOICE_ROOT="/mnt/work/chee-wei-jie/voice-models/CosyVoice"
QWEN_PRETRAIN="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN"
CHECKPOINT="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B/llm.pt"
TRAIN_DATA="/mnt/work/chee-wei-jie/voice-models/FEMALE_01_cosyvoice3_data/train/parquet/data.list"
CV_DATA="/mnt/work/chee-wei-jie/voice-models/FEMALE_01_cosyvoice3_data/dev/parquet/data.list"
CONFIG="${COSYVOICE_ROOT}/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml"
DS_CONFIG="$(dirname "$0")/configs/ds_stage2_lora.json"

RUN_TAG="female01_cv3_lora_lr5e5_run1"
MODEL_DIR="/mnt/work/chee-wei-jie/voice-models/CosyVoice_runs/${RUN_TAG}/exp/female01/cosyvoice3/llm/deepspeed"
TB_DIR="/mnt/work/chee-wei-jie/voice-models/CosyVoice_runs/${RUN_TAG}/tensorboard"

mkdir -p "${MODEL_DIR}" "${TB_DIR}"

echo "Run LoRA train (CosyVoice3). RUN_TAG=${RUN_TAG}"
echo "LR=5e-5, LoRA r=16 alpha=64, max_epoch=20, DeepSpeed Stage 2"

cd "${COSYVOICE_ROOT}"

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS"

torchrun --nnodes=1 --nproc_per_node=1 \
    tools/train_cosyvoice3_lora.py \
    --train_engine deepspeed \
    --model llm \
    --config "${CONFIG}" \
    --train_data "${TRAIN_DATA}" \
    --cv_data "${CV_DATA}" \
    --qwen_pretrain_path "${QWEN_PRETRAIN}" \
    --checkpoint "${CHECKPOINT}" \
    --model_dir "${MODEL_DIR}" \
    --tensorboard_dir "${TB_DIR}" \
    --deepspeed_config "${DS_CONFIG}" \
    --num_workers 4 \
    --prefetch 100 \
    --lora-r 16 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    2>&1 | tee "/mnt/work/chee-wei-jie/voice-models/CosyVoice_runs/${RUN_TAG}/train.log"
