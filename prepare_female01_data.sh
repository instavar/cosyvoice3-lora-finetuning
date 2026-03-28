#!/usr/bin/env bash
set -euo pipefail

# Prepare FEMALE_01 slicer_opt data for CosyVoice3 training
# Converts GPT-SoVITS ASR list format -> CosyVoice3 parquet pipeline

COSYVOICE_ROOT="/mnt/work/chee-wei-jie/voice-models/CosyVoice"
FEMALE01_DIR="/mnt/work/chee-wei-jie/voice-models/FEMALE_01"
ASR_LIST="${FEMALE01_DIR}/asr_opt/slicer_opt.list"
SLICER_DIR="${FEMALE01_DIR}/slicer_opt"
PRETRAINED_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"

DATA_ROOT="/mnt/work/chee-wei-jie/voice-models/FEMALE_01_cosyvoice3_data"
TRAIN_DIR="${DATA_ROOT}/train"
DEV_DIR="${DATA_ROOT}/dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda env
source /home/chee-wei-jie/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

cd "${COSYVOICE_ROOT}"
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS"

echo "=== Step 0: Parse ASR list into wav.scp / text / utt2spk / spk2utt =="
mkdir -p "${TRAIN_DIR}" "${DEV_DIR}"
python3 "${SCRIPT_DIR}/parse_female01_asr.py" \
    "${ASR_LIST}" "${SLICER_DIR}" "${TRAIN_DIR}" "${DEV_DIR}"

echo "=== Step 1: Extract campplus speaker embeddings =="
for x in "${TRAIN_DIR}" "${DEV_DIR}"; do
    echo "Extracting embeddings for $(basename "$x")..."
    python3 tools/extract_embedding.py --dir "$x" \
        --onnx_path "${PRETRAINED_DIR}/campplus.onnx"
done

echo "=== Step 2: Extract speech tokens =="
for x in "${TRAIN_DIR}" "${DEV_DIR}"; do
    echo "Extracting speech tokens for $(basename "$x")..."
    python3 tools/extract_speech_token.py --dir "$x" \
        --onnx_path "${PRETRAINED_DIR}/speech_tokenizer_v3.onnx"
done

echo "=== Step 3: Create parquet files =="
for x in "${TRAIN_DIR}" "${DEV_DIR}"; do
    echo "Creating parquet for $(basename "$x")..."
    mkdir -p "$x/parquet"
    python3 tools/make_parquet_list.py --num_utts_per_parquet 1000 \
        --num_processes 4 \
        --instruct \
        --src_dir "$x" \
        --des_dir "$x/parquet"
done

echo "=== Done! Data prepared at =="
echo "Train: ${TRAIN_DIR}/parquet/data.list"
echo "Dev:   ${DEV_DIR}/parquet/data.list"
