#!/usr/bin/env python3
"""Parse FEMALE_01 ASR slicer list into CosyVoice3 training format."""
import os
import random
import sys

random.seed(42)

asr_list = sys.argv[1]
slicer_dir = sys.argv[2]
train_dir = sys.argv[3]
dev_dir = sys.argv[4]

entries = []
with open(asr_list) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        old_path, spk, lang, text = parts[0], parts[1], parts[2], parts[3]
        basename = os.path.basename(old_path)
        actual_path = os.path.join(slicer_dir, basename)
        if not os.path.exists(actual_path):
            continue
        utt_id = basename.replace(".wav", "")
        entries.append((utt_id, actual_path, "FEMALE_01", text.strip()))

random.shuffle(entries)
dev_count = max(50, int(len(entries) * 0.05))
dev_entries = entries[:dev_count]
train_entries = entries[dev_count:]

for split_dir, split_entries in [(train_dir, train_entries), (dev_dir, dev_entries)]:
    os.makedirs(split_dir, exist_ok=True)
    spk2utt = {}
    with open(os.path.join(split_dir, "wav.scp"), "w") as wf, \
         open(os.path.join(split_dir, "text"), "w") as tf, \
         open(os.path.join(split_dir, "utt2spk"), "w") as uf, \
         open(os.path.join(split_dir, "instruct"), "w") as inf:
        for utt_id, wav_path, spk, text in split_entries:
            wf.write(f"{utt_id} {wav_path}\n")
            tf.write(f"{utt_id} {text}\n")
            uf.write(f"{utt_id} {spk}\n")
            inf.write(f"{utt_id} You are a helpful assistant.<|endofprompt|>\n")
            spk2utt.setdefault(spk, []).append(utt_id)
    with open(os.path.join(split_dir, "spk2utt"), "w") as sf:
        for spk, utts in spk2utt.items():
            joined = " ".join(utts)
            sf.write(f"{spk} {joined}\n")
    print(f"{os.path.basename(split_dir)}: {len(split_entries)} utterances")
