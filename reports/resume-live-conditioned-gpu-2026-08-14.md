# Live-conditioned CosyVoice3 resume evidence

Date: 2026-08-14, Asia/Singapore

## Outcome

A real single-GPU CosyVoice3 LoRA run passed the Instavar Voice evaluator 0.45
live-conditioned resume contract. An uninterrupted two-update run and a second
run interrupted by `SIGTERM` after update 1 produced byte-identical final files
for all five required roles after the second run resumed in a fresh process.

The evaluator reported:

- `status: passed`
- `claim_tier: byte_exact_live_conditioned_artifact_set`
- `conditioning_artifacts_verified: true`
- `exact_resume_artifact_equivalence: true`
- `independent_artifact_storage_verified: true`
- report SHA-256
  `360498b8c30348e6966fea6716f6f5c617165c1e7b2b7fd08791d1f8b49ad4b1`

## Bound execution

The producer was `instavar/cosyvoice3-lora-finetuning` revision
`a9cba1bb8481daa7871f397bf6d26ffb9e0f1c7b`. The patched upstream source was
`FunAudioLLM/CosyVoice` revision
`a8b2d9d05483a32f92adb1610f56464263b3598c`, with Matcha-TTS revision
`dd9105b34bf2be2230f4aa1e4769fb586a3c824e`. Evaluation used
`instavar/instavar-voice-evaluation` revision
`29c38cfd86b889abc8b79df063c817dd8f684903`.

The test ran on one NVIDIA RTX 3090 Ti with Python 3.10.19, Torch 2.3.1+cu121,
CUDA 12.1, AMP enabled, DDP world size 1, one training row, one validation row,
and one optimizer update per epoch. Both conditions used seed 1234,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic Torch algorithms, disabled
TF32, deterministic cuDNN, LoRA rank 16, alpha 64, dropout 0.05, and the same
sorted target module set.

The conditioning receipt bound four live identities:

| Role | SHA-256 |
| --- | --- |
| Base `llm.pt` | `69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970` |
| Dataset lineage | `606fd175fb97b6cfa60a90afe82fe7268713844bff8093ca3b78b7c62ea1886b` |
| Training controls | `bc59d06d864b31c87a51e6bc21dc2081e4d6207857407d21a7e7c135826e8d90` |
| Serialized initial adapter tree | `5c86900c35ce936751ea2b6dca3de8c068021aa8eb699e91d3016c296294f6d8` |

The initial adapter weight files were independently produced and matched at
`72f4fae8f316590bd346b80bc833ef46566b2adfc21799676d4c7edba7822ff5`.

## Interruption and role results

The interruption harness waited until `resume_epoch_000000` contained its
sidecar and all five non-empty role files, then sent `SIGTERM` to the complete
`torchrun` process group. Torch Elastic recorded signal 15 and returned launcher
exit code 1. No epoch-1 guarded checkpoint, partial guarded directory, or
inference export existed before resume. The new process loaded completed update
1 and reached completed update 2.

| Evaluator role | Final SHA-256 |
| --- | --- |
| `model_state` | `ac1e0c2c669c436b61c33ccb9c444094d65ac1b748f69c2a7d6c886ac45367f7` |
| `optimizer_state` | `8ead82b3827409d5fc1522ae801c2a318da563d806a38dafba8f05e928cc5be8` |
| `scheduler_state` | `05cb00999504ff9a3630594d18cc9054de84e2e6582eee47cf6ac8d4d29adbf7` |
| `trainer_state` | `7c6f5cbe071991a7a15d6e7fa1e8b62e40e599903606e11bcbba422eab39f98e` |
| `rng_state` | `36e389c1e9f2af3b8a553332c7c3dea1fda746d88bed4c9daa7b0e2436848691` |

The epoch-0 role files also matched exactly before resume. The two sidecars were
not expected to be byte-identical because each honestly binds its own output
path and inode. Their run-specific values were excluded from the five-role
artifact-equivalence claim.

## OOD finding and repair

The first two one-update probes exposed process-dependent PEFT serialization of
the set-like `target_modules` field. Adapter tensors and all five checkpoint
roles already matched, but `adapter_config.json` did not. Companion-side
canonicalization now sorts `target_modules`, `modules_to_save`, and
`exclude_modules` while retaining sequence order for fields whose order can be
semantic. Two subsequent fresh processes produced identical initial and
checkpoint adapter metadata.

The interruption harness also corrected an evidence assumption: Torch Elastic
maps an observed process-group `SIGTERM` to launcher exit code 1, not shell-style
143. The retained receipt records both the signal sent and the actual launcher
result instead of rewriting the outcome.

## Retained evidence

The controlled GPU host retains the complete runs at:

`/mnt/ext4_4tb/chee-wei-jie/voice-models/instavar-cosyvoice3-resume-live-045-20260814`

A compact checksum-verified bundle without the large model and optimizer files
is retained locally at:

`/Users/CheeWeiJie/Downloads/desktop-tailscale-tts/cosyvoice3-resume-live-045-20260814`

It contains evaluator receipts, comparison plan and report, conditioning
metadata, role hashes, interruption logs and receipt, the PEFT ordering negative
probe, and `SHA256SUMS`.

## Evidence boundary

This validates exact serialized continuation for the declared single-process,
single-GPU, one-row, two-update LoRA slice. It does not prove model quality,
adaptation benefit, hidden state equality beyond the five declared files,
general numerical determinism, arbitrary datasets, data-loader workers,
multi-rank DDP, or DeepSpeed continuation. Those broader modes remain rejected
by guarded resume until their rank-local, worker, sampler, and collective
publication state can be represented and tested.
