# CosyVoice3 Base and epoch-12 corrected emotion-route preregistration

Date: 2026-08-13

Status: frozen before generation

## Question

Rerun the two `emotion_control` prompts that the historical runner silently
routed through `inference_zero_shot`. Compare unchanged CosyVoice3 Base with
the selected epoch-12 LoRA adapter while requiring every row to use the
corrected upstream `inference_instruct2` route.

This run can establish whether the exact route executes, whether each output is
runtime-valid and requested-text faithful, and whether a blind reviewer hears
the requested calm-versus-concerned contrast. It cannot establish general
emotion controllability, adaptation benefit, or model-family quality.

## Frozen identities

- runner source revision before this preregistration:
  `d0db0cecfb397cbd446c7b643d80df8fc4d33dd3`
- evaluator revision:
  `286f7e7052740d6fec9a5a4b5c27a0419ed310fd`
- Base candidate: `cosyvoice3-base`
- adapter candidate: `cosyvoice3-epoch12`
- prompt pack: `instavar-singapore-english` 1.2.0
- prompt-pack canonical SHA-256:
  `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- generation-plan file SHA-256:
  `4d8ab1115d9fa592b68daa9adbdefbe919066cbff97acaa21969fbf810ea9b50`
- canonical generation-plan SHA-256:
  `87d8a099385e30b3f08e77b14b41b5039e13f8fe9ced961dfb545c8001a76b51`
- Base artifact-set SHA-256:
  `7c9fb1b5d255cd899bc2b562b1f0b58540f73e24f3902ec5276a535183ed7e54`
- adapter artifact-set SHA-256:
  `7b4b30762864daaa60a355b9445e9722b393ef0b3c472157f4905e74c9dfb2da`
- epoch-12 PEFT tree SHA-256:
  `8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`
- retained reference WAV SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- retained reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- device: NVIDIA RTX 3090 Ti

The focused plan contains 12 rows: two candidates, prompts
`emotional-neutral` and `emotional-concerned`, and seeds `42`, `314159`, and
`20260812`. Both prompts use the same requested text. Their exact instructions
are `Read with calm confidence.` and `Read with restrained concern.`

## Route and generation contract

Base and adapter run in separate fresh processes from the same clean detached
public checkout. Both receive the same pretrained tree, reference WAV, plan,
seed, speed, text frontend, and requested instruction. The adapter process adds
only the frozen PEFT tree.

Every observation must record:

- `instruction_requested: true`
- `instruction_route: inference_instruct2`
- `instruction_applied: true`
- `applied_instruction` byte-equal to the plan instruction
- explicit `artifact_mode` and a mode-specific runtime identity

The run must fail closed rather than fall back to zero-shot if the installed
runtime lacks `inference_instruct2`. The upstream route accepts target text,
instruction, and reference WAV but not the reference transcript. That API
difference is part of the frozen route rather than an accidental omission.

## Preregistered evaluation

The required objective slots remain separate: requested-text WER, ECAPA speaker
similarity, invalid-output rate, duration, sample rate, silence fraction,
clipping fraction, real-time factor, and peak allocated CUDA memory. No
composite score or quality winner is defined.

The content-faithfulness diagnostic is frozen at:

- n-gram size: 4
- minimum reference-exclusive hits: 2
- repetition excess fraction threshold: 0.05
- WER threshold: 0.1

High WER, repeated n-gram excess, and retained-reference transcript overlap are
separate failures. `not_flagged` is not proof of fidelity or obedience.

If the samples are runtime-valid, generate a counterbalanced blind pack using
the evaluator's prompt-aware routing. Emotion and instruction obedience applies
because both prompts contain visible instructions. Speaker identity,
naturalness, and artifact severity may provide separate context. No rating,
preference, or instruction-obedience result will be invented.

## Stop and interpretation rules

- Preserve exactly one observation for every planned row, including invalid or
  failed output.
- Do not retry a seed or replace an unfavorable sample.
- Treat a missing or false route assertion as a route failure even if the WAV
  is otherwise valid.
- Treat runtime-valid audio as insufficient for content or instruction claims.
- Do not use ASR, ECAPA, silence, or deterministic prosody proxies as proof of
  emotion obedience.
- A Base or adapter failure applies to this exact route, prompt, seed, and
  artifact. It is not a family-wide result.
- Any threshold or route changed after inspecting output becomes post-hoc
  characterization and must be labeled as such.
