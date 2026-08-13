# Matched PyTorch and vLLM runtime comparison

Date: 2026-08-13

## Result

The frozen epoch-12 PyTorch adapter and its derived merged vLLM export both
completed all three planned seeds and produced structurally valid 24 kHz WAV
files. The vLLM runtime was faster, but both candidates failed the preregistered
content-faithfulness gate. The vLLM candidate had higher requested-text word
error rate, more repetition, longer output, and substantially higher peak GPU
allocation. This run does not establish runtime equivalence or a quality
promotion for the merged export.

The result is bounded to one neutral Singapore English prompt, three frozen
seeds, one retained speaker reference, the selected epoch-12 adapter, the one
derived export, and the recorded software revisions. It generalizes to the
promotion rule that runtime speed must not override a failed content gate. It
does not establish that every vLLM export or every CosyVoice3 prompt behaves the
same way.

## Frozen inputs

- Preregistration:
  `evaluation/preregistration/cosyvoice3-epoch12-pytorch-vllm-2026-08-13.md`
- Generation plan SHA-256:
  `6a0c4594ad7e5e37596dd4e5e60a410f90127c8a3e219a9d8d879e9e2febf19a`
- Companion revision:
  `458f808404f782077a9f056eca32eef1fa8431a3`
- Evaluator revision and version:
  `fa4fab5a2499f863d903baa45bfba0f090fe18af`, version 0.39
- Runtime artifact-set id:
  `cosyvoice3-epoch12-pytorch-vllm-20260813-v1`
- PyTorch exact artifact-set SHA-256:
  `de6cfceafb0155d8bf89336d4c39618b46bcf9ba230da5f5e0c806dfe0d0a87f`
- vLLM derived artifact-set SHA-256:
  `9e8bae4f44fab5474351bd2a06c49e915ecf379ec8ec1725e581fbae4ef164a4`
- Runtime artifact manifest SHA-256:
  `3e1587cae4e9cf4f70c4b3a891af5874077270de2252d097387bada7c00e9ae7`
- Reference WAV SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- Reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`

The vLLM binding is `derived`, not `exact`. The export was produced once with
`tools/infer_cosyvoice3_lora.py:enable_vllm_with_merged_lora` and then reused.
The evaluator therefore used an ordinary matched comparison rather than the
exact-artifact `compare-runtimes` claim.

## Objective results

Each candidate completed three of three rows with no invalid output. Means are
shown below. Confidence intervals and per-seed rows remain in the retained raw
artifacts.

| Metric | PyTorch adapter | Merged vLLM | vLLM minus PyTorch |
| --- | ---: | ---: | ---: |
| Requested-text WER | 0.505376 | 0.774194 | +0.268817 |
| Speaker similarity | 0.789487 | 0.814275 | +0.024788 |
| Output duration, seconds | 12.093333 | 17.586667 | +5.493333 |
| Generation time, seconds | 5.042185 | 2.754444 | -2.287741 |
| Real-time factor | 0.408885 | 0.159107 | -0.249779 |
| Peak GPU allocation, bytes | 4,269,554,517 | 7,872,665,771 | +3,603,111,253 |
| Silence fraction | 0.378121 | 0.397261 | +0.019139 |
| Clipping fraction | 0.000000 | 0.000000 | 0.000000 |

The vLLM generation-time and real-time-factor deltas favor vLLM. The vLLM WER
delta favors PyTorch, and its 95 percent bootstrap interval crosses a wide
range because there are only three pairs. The speaker-similarity delta also has
an interval crossing zero. No proxy establishes a perceptual winner.

## Content-faithfulness gate

Both candidates failed all three evaluable rows.

| Candidate | Mean WER | High-WER rows | Reference-overlap rows | Repetition rows |
| --- | ---: | ---: | ---: | ---: |
| PyTorch adapter | 0.505376 | 3 | 3 | 1 |
| Merged vLLM | 0.774194 | 3 | 1 | 2 |

The ASR hypotheses contain words from the retained reference transcript and
repeated requested-text fragments. This is deterministic text evidence of
content failure and possible conditioning leakage. It does not identify the
causal layer. Possible causes include the frontend conditioning path, the
adapter itself, the merge/export transform, or runtime decoding behavior.

## Prosody and audio identity

All three matched PyTorch and vLLM WAV hashes differ. Byte inequality proves
only that the files differ. Because the vLLM artifact is derived, neither equal
nor unequal bytes would by itself prove semantic runtime equivalence.

The matched prosody report contains three complete pairs for every recorded
proxy. Its signed deltas are descriptive only. They do not establish cadence,
monotony, accent fidelity, naturalness, or preference.

## Blind listening

Six identity-neutral audio files were staged with two counterbalanced rater
schedules. The applicable criteria are speaker identity, Singapore English
accent fidelity, naturalness, and artifact severity. The pack remains unrated.
Human ratings cannot rescue the failed content gate, but they can help localize
whether the non-faithful outputs also differ perceptually.

## Retained evidence

The raw artifacts remain on the controlled GPU host under
`voice-model-outputs/conformance/20260813_cosyvoice3_pytorch_vllm_v1`. They are
not committed because they include generated voice audio and large extractor
outputs.

- Generation attempt receipt SHA-256:
  `5e257b3b7bdf4ba5276bfe5135fac614b4f24f6f46c7b38331e2f5d6c5cdbbb4`
- ASR results SHA-256:
  `7547401f8f3eba24603eb560c937bb669ca284802e6da790e4b4bd409f368f2a`
- Speaker results SHA-256:
  `93914b537793a6f19753eb759378402f5b146c7b0f7515a82dac2b6ac4b8f1ac`
- Objective report SHA-256:
  `897c271b3fae80ce9e50ddc3f784f0db4c200f88140f93ac21639f66cffc376f`
- Content-faithfulness report SHA-256:
  `69a21bf03242de817e2fdda85f7a99391033fe803861a19b5b131d41b9eeb7e1`
- Matched objective comparison SHA-256:
  `6b16968c916246bda233118ab5a2f24984a45e6a2de7a021e4f6d64733f8dbd9`
- Matched prosody comparison SHA-256:
  `ca6b2856fd455bdcb02fb6b88c5d16b912af018ab398c8da8c0bc4d438d67928`
- Exact WAV comparison SHA-256:
  `7372f3ee68a52fbcfd524798c45057d5c07f8b4a850244ad89340cbeee8507d8`
- Blind audio manifest SHA-256:
  `addc5cb74f6e9f7219dd31ea608b37058a5dd34f00760cd9d3218cb889178e56`
- Blind review packet SHA-256:
  `db48d0f2afd3e9a45325c722b3164791ee3e318c66f1498543be86e66a48c87f`

## Decision and next experiment

Keep merged vLLM available as an experimental optimized runtime, but do not
promote this export as equivalent or content-conformant. The next controlled
run should isolate export from decoding:

1. Reproduce PyTorch inference from the merged weights without vLLM.
2. Compare adapter PyTorch, merged-weight PyTorch, and merged-weight vLLM under
   one new frozen plan.
3. Use at least one short neutral prompt, one pronunciation prompt, and one
   longer structural prompt across multiple seeds.
4. Require the content gate before interpreting speed, speaker, prosody, or
   human-preference results.

That three-way design can distinguish adapter-to-merge drift from
merged-model-to-runtime drift. This run alone cannot.
