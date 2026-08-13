# CosyVoice3 epoch-12 PyTorch and merged-vLLM preregistration

Date: 2026-08-13

Status: frozen before export preparation or evaluated generation

## Question

Compare the selected epoch-12 LoRA artifact through two runtime paths:

- PyTorch loads the source PEFT adapter at inference time; and
- vLLM loads an offline merged export derived from that adapter.

The experiment can measure bounded end-to-end differences in output validity,
requested-text fidelity, speaker proxy, duration, timing, memory, signal
diagnostics, deterministic prosody proxies, and blind listening. It cannot prove
bitwise, numerical, or perceptual runtime equivalence because the vLLM artifact
is a derived representation and the runtime implementations differ.

## Frozen identities

- companion source revision before preregistration:
  `f131f27598f54dc8e71b8ef98e322721b75c85e8`
- evaluator revision:
  `fa4fab5a2499f863d903baa45bfba0f090fe18af`
- PyTorch candidate: `cosyvoice3-epoch12-pytorch`
- merged-vLLM candidate: `cosyvoice3-epoch12-vllm`
- prompt pack: `instavar-singapore-english` 1.2.0
- prompt-pack canonical SHA-256:
  `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- generation-plan file SHA-256:
  `6a0c4594ad7e5e37596dd4e5e60a410f90127c8a3e219a9d8d879e9e2febf19a`
- canonical generation-plan SHA-256:
  `c4b598135aae0a4f6cf1a18fefe41a5245385afcc843a086d61026efeaa23e18`
- adapter artifact-set SHA-256:
  `7b4b30762864daaa60a355b9445e9722b393ef0b3c472157f4905e74c9dfb2da`
- epoch-12 PEFT tree SHA-256:
  `8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`
- retained reference WAV SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- retained reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- device: NVIDIA RTX 3090 Ti

The committed plan contains six rows: prompt `neutral-brief`, seeds `42`,
`314159`, and `20260812`, and both candidates. The prompt has no style
instruction, so both runtimes use `inference_zero_shot` with the same requested
text, reference WAV, reference transcript, seed, frontend, and speed.

## Derived-export preparation

The vLLM export does not exist when this protocol is frozen. After this
preregistration reaches public main:

1. use the frozen companion revision and exact PEFT tree to create one new
   merged export path;
2. do not treat any audio emitted during export preparation as an evaluated
   observation;
3. record Python, Torch, Transformers, PEFT, vLLM, and TorchCodec versions;
4. build a content-addressed runtime-artifact manifest that marks PyTorch as an
   `exact` binding to the source adapter and vLLM as `derived` with converter
   revision equal to the frozen companion commit;
5. verify the live source and export bytes immediately before and after
   evaluated generation; and
6. run evaluated vLLM generation only through explicit reuse of that frozen
   export.

The export digest is necessarily observed after conversion. It is evidence of
the preregistered conversion, not a preselected outcome. Any failed export is
preserved and ends the run. A second export cannot replace it under this plan.

## Runtime contract

Run each candidate in a separate fresh process from one clean detached public
checkout. The PyTorch rows must record `artifact_mode: adapter` and runtime ID
`pytorch-adapter`. The vLLM rows must record `artifact_mode: merged` and runtime
ID `vllm-merged`. Both must carry the same source artifact-set ID and digest;
the separate runtime manifest records exact versus derived relation.

Do not retry or replace a seed. Preserve invalid output. Reject tiny or silent
audio, background-thread exceptions, mismatched plan rows, artifact mutation,
an existing unverified export, partial artifact bindings, or runtime-ID drift.

## Evaluation

Required objective slots remain separate:

- requested-text WER;
- ECAPA speaker similarity;
- invalid-output rate;
- duration and sample rate;
- silence and clipping fractions;
- real-time factor; and
- peak allocated CUDA memory.

The preregistered content thresholds remain four-grams, two retained-reference
hits, repetition excess over `0.05`, and WER over `0.1`. No instruction-overlap
test applies because the selected prompt has no instruction.

Also compute a matched deterministic prosody-proxy comparison without assigning
a quality direction. Stage a counterbalanced blind pack for speaker identity,
naturalness, artifact severity, and whichever neutral-prompt criteria the
evaluator routes without override. Do not invent ratings.

Report exact WAV identity separately. Byte-identical WAVs are useful evidence
for this run but not a general determinism or runtime-equivalence proof.
Non-identical WAVs require objective and blind comparison; they are not an
automatic regression.

## Interpretation and stop rules

- The comparison is a runtime conversion assessment, not an adaptation-benefit
  comparison, because both candidates originate from the same adapter.
- A derived vLLM binding cannot be promoted to exact artifact identity.
- Matching objective means do not prove sample-level or perceptual equivalence.
- A faster runtime cannot win if it produces invalid or content-unfaithful
  output.
- One prompt and three seeds do not establish behavior for long form,
  instructions, pronunciation, accent, other speakers, or other versions.
- Any changed prompt, seed, export, converter source, package version, or
  threshold starts a new experiment.
