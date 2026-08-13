# Matched long-form Base and epoch-12 adapter evidence, 2026-08-13

## Result

The unchanged CosyVoice3 Base and the epoch-12 LoRA adapter both completed the
same long-form zero-shot generation contract, but neither produced acceptable
content. The adapter was materially worse on requested-text intelligibility.
This is a negative quality result, not a model promotion.

The evaluator accepted the complete nine-metric objective contract and reported
`proves_adaptation_benefit: false`. Basic WAV validity therefore did its job as
a runtime check, but it did not establish semantic faithfulness. Faster-whisper
found repetitions, omissions, garbling, and content from the retained reference
transcript in the hypotheses. The adapted output also received higher ECAPA
speaker similarity, which must not override its much worse requested-text WER.

## Frozen comparison

The plan was committed before generation in
[`evaluation/preregistration/cosyvoice3-base-epoch12-long-form-2026-08-13.md`](../evaluation/preregistration/cosyvoice3-base-epoch12-long-form-2026-08-13.md).
Hosted preregistration workflow `31669191246` passed at
`2026-08-13T05:06:51Z`.

- runner revision: `e9775c563bcdafc2c229265f6493580191c9977b`
- clean detached runner: same revision, clean before and after generation
- evaluator at generation: `982367abc7837cb6da5ebb94192c9642dea62fce`
- evaluator used for final unified validation:
  `d9c990522e166f22b1be4aa8bee770d7686af245` (version 0.36.0)
- prompt: `cadence-two-minute`
- seed: `20260812`
- route for both candidates: `inference_zero_shot`
- device: NVIDIA RTX 3090 Ti
- prompt-pack SHA-256:
  `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- canonical generation-plan SHA-256:
  `ffbd2edb68cf45e63a5e295ffd1a117cffdbd80d11857f5e9fc1a1b8dba6ba3c`
- canonical speaker assignment SHA-256:
  `0fd99ca68a7cb803bcd7fe8ed553a265a31bb8d847ecbb66c39e0a2d5fe266a1`
- retained reference WAV SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- retained reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- Base candidate artifact-set SHA-256:
  `7c9fb1b5d255cd899bc2b562b1f0b58540f73e24f3902ec5276a535183ed7e54`
- adapter candidate artifact-set SHA-256:
  `7b4b30762864daaa60a355b9445e9722b393ef0b3c472157f4905e74c9dfb2da`

Both candidates received the exact same requested text, reference WAV,
reference transcript, seed, speed, frontend, and upstream route. Base loaded no
adapter. The adapter condition loaded the exact three-file epoch-12 PEFT tree
with SHA-256
`8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`.

## Objective observations

| Metric | Base | Epoch 12 | Adapter minus Base |
|---|---:|---:|---:|
| WER | 0.281385 | 0.515152 | +0.233766 |
| ECAPA cosine | 0.760186 | 0.876703 | +0.116517 |
| duration, seconds | 112.24 | 60.96 | -51.28 |
| generation, seconds | 24.417717 | 22.616208 | -1.801509 |
| real-time factor | 0.217549 | 0.371001 | +0.153452 |
| peak CUDA bytes | 4,599,720,448 | 4,424,959,488 | -174,760,960 |
| silence fraction | 0.383464 | 0.373031 | -0.010433 |
| clipping fraction | 0 | 0 | 0 |
| sample rate | 24 kHz | 24 kHz | 0 |

- Base WAV SHA-256:
  `74a558f565e1957fb71d20048dd89a09e9fbc57b4837fe031d0cd6de447528b5`
- adapter WAV SHA-256:
  `5d2a20820b3f1a1c7c8243cb6f85e86c4207c3fcf575b71ccf34ee720438867c`
- final complete observations SHA-256:
  `bd3558e7e4c89a351ecf33f28acf90516c5a6f0e01e2627acafc89fb212cf918`
- final objective report SHA-256:
  `fa6f747fc3428916157e1e3eb8d73cc9887c194faa3dee03c9da36375e9d53ea`
- final matched objective SHA-256:
  `7551606dd535e1d14679abdc869e1314e3cc606d14be960d3e83ffd94bd6bc67`
- final matched prosody SHA-256:
  `5c85371b0b7c1bdbb4e4bbabbae6d4fec7c9ac9a65d536e958f98f23653af63e`

The objective matched output remained byte-identical when evaluator 0.36.0
revalidated the observations after prosody extraction. Hosted evaluator main
workflow `31670036671` passed at `2026-08-13T05:21:42Z`.

## Interpretation and limits

The result generalizes only to this Base checkpoint, epoch-12 adapter, retained
speaker reference, long-form prompt, seed, and software and hardware stack. A
single pair does not establish the family-wide quality of CosyVoice3 or LoRA.
It does establish four narrower points for this exact evidence:

1. Both artifact modes execute in fresh processes through the intended
   same-conditioning runtime path.
2. Signal-level validity does not rule out semantic corruption.
3. Higher speaker-embedding similarity does not compensate for substantially
   worse requested-text fidelity.
4. This adapter cannot be promoted from this run, and the Base result is not a
   quality success either.

The prosody comparison has no preregistered direction and no winner. Its
duration and pause deltas are confounded by the different corrupted content, so
they cannot support a cadence or naturalness claim.

A criterion-scoped blind pack was produced for two pseudonymous raters. It
covers speaker identity, cadence variation, long-form monotony, naturalness,
artifact severity, and listening fatigue. Accent fidelity, lexical
pronunciation, and emotion obedience were excluded because this prompt cannot
support them. The reveal remains unopened and no ratings were invented. Given
the objective semantic failure, blind ratings are not needed to reject a
quality-promotion claim, though completed ratings could still characterize the
failure.

## Follow-up

- Add requested-text repetition and retained-reference leakage diagnostics to
  the evaluator, with the two transcripts bound separately.
- Replicate across preregistered seeds before estimating failure frequency.
- Rerun the full prompt suite through the corrected instruction route.
- Keep speaker similarity, intelligibility, content faithfulness, prosody, and
  listening criteria separate. Do not collapse them into one score.

Private evidence is retained under
`/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260813_cosyvoice3_matched_long_form_v1`.
That path is an operational locator, not a public artifact guarantee.
