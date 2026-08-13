# Matched vLLM sampling-profile diagnosis

Date: 2026-08-14

## Result

The preregistered 45-row experiment found a bounded association between vLLM
request sampling and generated audio, but no condition passed content
faithfulness.

Two unchanged upstream vLLM runs were byte-identical for all nine prompt and
seed pairs. Adding only the plan-row request seed kept the six local-context and
names-and-numbers WAVs byte-identical to upstream, while changing all three
structured-long-form WAVs. Adding only top-p 0.8 after request seeding changed
all nine WAV hashes and produced one invalid near-silent output.

Every evaluable row exceeded the preregistered requested-text WER threshold of
0.10. No profile is promoted. The result does not establish PyTorch and vLLM
equivalence, better quality, correct pronunciation, accent fidelity, natural
cadence, or preference.

## Frozen design and chronology

- Sampling-control implementation commit:
  `19dded017efbd901abf0547330f7296cea757b17`
- Preregistration commit:
  `e9dd527c4c1fe0d6bf014a7cdb8f6d5c41e6ea71`
- Feature-branch workflow: GitHub Actions run `31727821584`, passed before
  generation.
- Main workflow: GitHub Actions run `31727880740`, passed before generation.
- Generation plan SHA-256:
  `dee7c76b2de146822ca9284027562af633732e8b0fb650eacb849ace9bdf3183`
- Protocol SHA-256:
  `d266a461770cf3a14a4ed53786158c97fc38ee204646edec80cb6849007ce9e9`
- Producer revision: `e9dd527c4c1fe0d6bf014a7cdb8f6d5c41e6ea71`
- Evaluator revision and version:
  `2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0.
- Prompts: `local-context`, `names-numbers`, and `structured-long-form`.
- Seeds: 42, 314159, and 20260812.
- Planned and observed rows: 45.
- Valid signal-level rows: 44.
- Invalid rows: one retained near-silent top-p output.
- Runtime artifact verification before generation, after generation, and after
  evaluation: passed.

The direct findings apply to the epoch-12 model, frozen reference, three
prompts, three seeds, retained artifacts, vLLM 0.15.1, and recorded GPU host.
The need for an unchanged rerun control generalizes to stochastic runtime
diagnosis. The observed profile effects do not generalize without replication.

## Conditions

| Candidate | Route | Request profile |
| --- | --- | --- |
| Reloaded merged PyTorch | Persisted safetensors in a fresh PyTorch process | Not applicable |
| vLLM upstream-a | Retained merged export in a fresh process | top-k 25, top-p 1.0, request seed null |
| vLLM request-seeded | Same export in a fresh process | upstream plus plan-row request seed |
| vLLM seeded top-p | Same export in a fresh process | request seed plus top-p 0.8 |
| vLLM upstream-b | Unchanged repeat after the profile runs | top-k 25, top-p 1.0, request seed null |

Every vLLM observation records the selected profile and effective request
parameters. These records describe supplied parameters, not internal sampler
execution. None of the profiles reproduces CosyVoice PyTorch repetition-aware
sampling.

The reloaded merged PyTorch artifact-set SHA-256 is
`35d67b96b76ebe9716036a94a371842876682d595670a4d26e03ded177f399ab`.
The retained vLLM export artifact-set SHA-256 is
`db7b3b425b4f8b4353c8111d49f57ebbf3f351a04352eadb3adb2ae80a26c866`.
Both are derived from the frozen adapter. Derived bindings record provenance
and do not support an exact-artifact runtime-equivalence claim.

## Exact audio identity

| Comparison | Byte-identical pairs | Interpretation |
| --- | ---: | --- |
| upstream-a versus upstream-b | 9 of 9 | No WAV drift in the unchanged fresh-process control |
| upstream-a versus request-seeded | 6 of 9 | Only the three structured-long-form rows changed |
| request-seeded versus seeded top-p | 0 of 9 | Every prompt and seed changed |

The unchanged control also had zero mean delta for speaker similarity, output
duration, peak allocation, clipping, silence, and all 11 recorded prosody
proxies. Mean generation time differed by 0.041962 seconds and mean real-time
factor by 0.002376, which is timing noise rather than an audio result.

CosyVoice split every structured-long-form row into two frontend synthesis
requests while the other selected prompts used one. Request seeding changed all
three two-request rows and none of the six one-request rows. This is an
association, not proof that the request boundary caused the effect. Text length,
token count, request count, and later stochastic decisions are confounded.

The row-level implementation does not retain request ordinals, the actual
runtime seed selected for an upstream request, generated token counts, or token
hashes. A boundary-focused follow-up needs that instrumentation before it can
distinguish length effects from request-reset effects.

## Objective means

| Metric | Reloaded PyTorch | Upstream-a | Request-seeded | Seeded top-p | Upstream-b |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid rows | 9 | 9 | 9 | 8 | 9 |
| Requested-text WER | 1.229207 | 0.576460 | 0.548091 | 0.525725 | 0.576460 |
| Speaker similarity | 0.492862 | 0.814921 | 0.820123 | 0.632377 | 0.814921 |
| Output duration, seconds | 46.422222 | 21.133333 | 21.515556 | 30.200000 | 21.133333 |
| Generation time, seconds | 16.069967 | 3.938017 | 4.017768 | 4.093616 | 3.979979 |
| Real-time factor | 0.347733 | 0.192785 | 0.194164 | 0.138183 | 0.195161 |
| Peak GPU allocation, bytes | 3,993,205,703 | 7,223,715,044 | 6,319,165,269 | 7,338,510,962 | 7,223,715,044 |

The seeded-versus-upstream mean WER delta was -0.028369. The seeded-top-p
versus seeded mean among eight valid matched pairs was -0.034059. Both
conditions still failed every row, so these small relative proxy differences
are not quality improvements. The top-p condition also increased invalid-output
rate by 0.111111, mean valid output duration by 7.745 seconds, and mean peak
allocation across attempts by about 1.02 GB.

The invalid top-p row was `local-context`, seed 20260812. Its retained WAV was
28.92 seconds but had peak amplitude `3.736294820555486e-05` and RMS
`6.095159506003256e-07`. Duration alone would have missed this failure.

## Content-faithfulness gate

| Candidate | Gate | High-WER rows | Reference-overlap rows | Repetition rows |
| --- | --- | ---: | ---: | ---: |
| Reloaded merged PyTorch | failed | 9 | 0 | 5 |
| vLLM upstream-a | failed | 9 | 4 | 0 |
| vLLM request-seeded | failed | 9 | 3 | 0 |
| vLLM seeded top-p | failed | 8 | 3 | 0 |
| vLLM upstream-b | failed | 9 | 4 | 0 |

The invalid top-p row also failed the candidate gate. Neither upstream run nor
any other condition produced a frozen accepted ASR form for `paiseh`. That miss
is recognition evidence only. It cannot determine how the word was pronounced.

Content failure outranks runtime speed, speaker similarity, output duration,
signal proxies, and listening impressions. The reloaded PyTorch condition was
especially corrupted, with mean WER 1.229207 and mean repeated four-gram excess
0.361505. The relative vLLM results do not make any condition production-ready.

## Blind listening

Forty-five identity-neutral WAVs were staged with two counterbalanced rater
schedules. The retained invalid WAV remains in the pack as a failed attempt.
Criterion routing covers speaker identity, Singapore English accent fidelity,
pronunciation where applicable, cadence, monotony, naturalness, artifact
severity, and listening fatigue. Emotion obedience is explicitly excluded
because the focused plan contains no instructed prompt.

The pack remains unrated. Human review can characterize pronunciation,
accent, cadence, naturalness, and fatigue, but cannot override the content gate.

## Retained evidence

Raw evidence remains on the controlled GPU host under
`voice-model-outputs/conformance/20260814_cosyvoice3_vllm_sampling_profiles_v1`.
Generated voice audio and extractor outputs are not committed.

- Audio-identity diagnostic SHA-256:
  `3fc6c03ebdc9c77f22bfe5290e27c18cb9d39df2307d997199a99c6feeddcd30`
- Generation-attempt receipt SHA-256:
  `cf71594c0d02b6f7ffb75f75842458540764d22ce95e03bd7c0851bd3bda1852`
- ASR results SHA-256:
  `b3247ef8870a39b9af6f0b309dbe9166c53d42ff3020422d9b5dc263313adf5b`
- Speaker results SHA-256:
  `6cdca713a15e53c3057c382db387f9cb866b57084aa6dee5faeecbdb1a78774f`
- Prosody results SHA-256:
  `ba3ae8195785735b137aa095d0f82efd71f5ac4347376ab11b2d1d3221404bf9`
- Objective report SHA-256:
  `adb6d1644bcbe5acf56a66f78601ca7f7bace8325623c8e976288b1df1473cb7`
- Content report SHA-256:
  `836e2c253c9333973cdaf0d758135519c05ae797b28573e005c081545a09b893`
- Upstream rerun comparison SHA-256:
  `7a48d29c0300cb37660668957a774a36b2eeba3c65f3cc5d8a98bcc62b7f7794`
- Upstream rerun prosody comparison SHA-256:
  `2d185e49cf0e80f2e1ee17fb8ae2010edc39776ace5044a78fe901dc797d9227`
- Blind stage manifest SHA-256:
  `de5e6208d1744a4677a622a1ec2a637b5f92ca2408a554fb3c44bb628e6da738`
- Blind review packet SHA-256:
  `cda967b5c4c54bb3819ca45704e3290f802a54aec53821ff240ab5b38cd1820c`

## Decision and next experiment

Keep all three vLLM profiles diagnostic-only. The unchanged upstream rerun
clears fresh-process WAV drift as the explanation for the observed differences
in this slice. Request seeding is associated only with the longer two-request
prompt, while top-p 0.8 changes every row and introduces a near-silent failure.
Neither profile repairs content faithfulness.

Before another GPU run, extend the sampling evidence to record each actual vLLM
request: row ID, request ordinal, effective seed, top-p, top-k, output token
count, and a token-sequence hash without copying generated token content. Then
preregister prompts on both sides of the frontend split boundary with matched
length controls. This can test request-reset versus sequence-length hypotheses
without claiming causal proof from the present row-level association.
