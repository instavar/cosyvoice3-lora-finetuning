# Four-way persisted-runtime comparison

Date: 2026-08-13

## Result

The preregistered four-way experiment clears merged-weight serialization and
fresh-process PyTorch reload as the source of the previously observed runtime
divergence in this tested slice.

Adapter PyTorch, merged-in-memory PyTorch, and a persisted safetensors model
reloaded into a fresh PyTorch process produced byte-identical WAV files for all
three seeds. Their ASR, speaker, audio, and prosody values were also identical.
The retained merged vLLM export produced different WAVs, higher requested-text
error, more repetition, longer output, different prosody proxy values, and
higher peak GPU allocation.

All four conditions failed every primary content-faithfulness row. This is a
diagnostic localization result, not a quality promotion or causal proof. It
does not establish general PyTorch and vLLM behavior outside the frozen model,
prompt, reference, environment, and seeds.

## Evidence chronology and scope

- Persisted-artifact implementation commit:
  `c53bc71752ce81aeba0250b3fcc54353721f6733`
- Preregistration commit:
  `8fa43a2ab849bf2d61664c1d71afad0c1a662a07`
- Generation plan SHA-256:
  `7cbd70c1b3353e514fa1fdc16bdc1ee0c4e31c16475329dbc13dcd1ae00884ef`
- Evaluator revision and version:
  `fa4fab5a2499f863d903baa45bfba0f090fe18af`, version 0.39
- Prompt count: 1
- Seeds: 42, 314159, 20260812
- Planned and completed rows: 12
- Invalid rows: 0
- Runtime artifact verification before and after generation: passed

The direct result applies to the epoch-12 adapter, one neutral Singapore
English prompt, the frozen reference, the recorded CUDA environment, and the
three seeds. The rule that content failure outranks speed applies generally.
Waveform identity and runtime differences require replication before they are
generalized to other prompts, adapters, hardware, library versions, or exports.

## Conditions and artifact boundaries

| Candidate | Route | Artifact boundary |
| --- | --- | --- |
| Adapter PyTorch | PEFT adapter through PyTorch | Exact source adapter binding |
| Merged PyTorch | `merge_and_unload` in memory, then PyTorch | Derived in memory |
| Reloaded merged PyTorch | Persisted safetensors loaded into a fresh pretrained PyTorch process | Derived artifact binding |
| Merged vLLM | Retained exported weights through vLLM 0.15.1 | Derived export binding |

The persisted PyTorch artifact contains a 1,976,163,632-byte
`merged_model.safetensors` file with SHA-256
`0159101d13869ef7a1d3b45a824534d908990bf693ed9a152e8ea9dc5daaf974`.
Its manifest binds the model class, exporter revision, source adapter digest,
weight digest, and byte count. The fresh process loaded it into the existing
base architecture without pickle.

The runtime artifact manifest separates the exact adapter input from the two
derived merged representations. Derived-artifact comparisons use matched
prompt and seed semantics. They do not claim exact-artifact runtime
equivalence, and `proves_runtime_equivalence` remains false.

## Objective means

| Metric | Adapter PyTorch | Merged PyTorch | Reloaded merged PyTorch | Merged vLLM |
| --- | ---: | ---: | ---: | ---: |
| Requested-text WER | 0.483871 | 0.483871 | 0.483871 | 0.774194 |
| Speaker similarity | 0.746955 | 0.746955 | 0.746955 | 0.815717 |
| Output duration, seconds | 11.866667 | 11.866667 | 11.866667 | 17.586667 |
| Generation time, seconds | 4.316287 | 3.737022 | 3.765503 | 3.207197 |
| Real-time factor | 0.375856 | 0.326060 | 0.328779 | 0.184536 |
| Peak GPU allocation, bytes | 3,632,279,723 | 3,623,755,776 | 3,623,628,971 | 7,243,958,272 |

Reloading the persisted PyTorch model added 0.028481 seconds of mean generation
time relative to the in-memory merge and reduced mean peak allocation by only
126,805 bytes. These small run-level differences do not affect the identical
audio result and are not a performance conclusion.

Relative to reloaded merged PyTorch, vLLM reduced mean generation time by
0.558306 seconds and mean real-time factor by 0.144243. It used about 3.62 GB
more mean peak GPU allocation, emitted audio 5.72 seconds longer on average,
and increased mean WER by 0.290323. Faster generation cannot override failed
content.

## Content-faithfulness gate

| Candidate | Gate | High-WER rows | Reference-overlap rows | Repetition rows |
| --- | --- | ---: | ---: | ---: |
| Adapter PyTorch | failed | 3 | 2 | 1 |
| Merged PyTorch | failed | 3 | 2 | 1 |
| Reloaded merged PyTorch | failed | 3 | 2 | 1 |
| Merged vLLM | failed | 3 | 1 | 2 |

Every row exceeded the preregistered WER threshold. The PyTorch hypotheses also
contained retained-reference overlap in two seeds and repetition excess in one
seed. The vLLM condition had a different corruption pattern, including
repetition excess in two seeds. No condition is production-conformant.

## Exact audio and prosody

For seeds 42, 314159, and 20260812:

- Adapter PyTorch, merged PyTorch, and reloaded merged PyTorch SHA-256 values
  were equal within each seed.
- Each vLLM SHA-256 differed from the three PyTorch values for the same seed.
- Every audio-derived metric and every recorded prosody proxy delta was zero
  between the in-memory and reloaded merged PyTorch conditions.
- Every recorded reloaded-PyTorch-versus-vLLM prosody proxy had a nonzero mean
  delta.

The exact PyTorch agreement is bounded evidence that the implemented
safetensors serialization and reload preserved the tested model behavior.
Signal proxies do not establish better cadence, naturalness, accent, or human
preference.

## Rerun determinism observation

The retained vLLM export produced the same three WAV hashes as the immediately
preceding three-way run. That contrasts with an earlier two-way-to-three-way
rerun where the same declared export and seeds produced different hashes.

Together, these runs show that a frozen seed and unchanged declared artifacts
are not by themselves proof of byte reproducibility, while also showing that
vLLM drift is not universal across every rerun. No general nondeterminism or
causal claim is made.

## Blind listening

Twelve identity-neutral audio files were staged with two counterbalanced rater
schedules. The criteria are speaker identity, Singapore English accent
fidelity, naturalness, and artifact severity. The pack remains unrated.
Perceptual ratings can add evidence but cannot override the content gate.

## Retained evidence

Raw evidence remains on the controlled GPU host under
`voice-model-outputs/conformance/20260813_cosyvoice3_four_way_v1`. Generated
voice audio, merged weights, and large extractor outputs are not committed.

- Runtime artifact manifest SHA-256:
  `36bfc3bd1392399286139211584279971c8a8db21ae813fb73ea6d162b1f0c83`
- Final artifact verification SHA-256:
  `372924efc784608d611d696930e324efcd5eb13a9a86e2981f7795344d644a9a`
- Generation receipt SHA-256:
  `1235dd6afae96f8f01ad68e4ea86e25aba8138cb31e79f7dc52341da3b7209c9`
- ASR results SHA-256:
  `596bacfed5fb5d4c7c10c33c811576024a947fa1a997a05b92c6dee422b1212e`
- Speaker results SHA-256:
  `06ffc442e5aea409947da545616f6336af382f9e1d7ab780680d10671303407b`
- Prosody results SHA-256:
  `08594a036de1ba5c419553e30929841e41dc71dd814473ff1dbec2acf98c5cf2`
- Objective report SHA-256:
  `e69da1206f6b163d4f95b6b8c929ae74ac27b310d5c8582c0e1863f8c67afbf6`
- Content report SHA-256:
  `8accaacc22384df05f967c30e3b193b19dc59d1b1581646416e68178be1ba5fe`
- In-memory-versus-reloaded PyTorch objective comparison SHA-256:
  `4f031d24f0f18833830a28aa61b8bda847e874db63e66b4468e5a5c41793aa55`
- Reloaded-PyTorch-versus-vLLM objective comparison SHA-256:
  `b0ca78bc91fc7d87023db57cf92c916f0fc7d5abd11d02a9554a4d010debffb8`
- In-memory-versus-reloaded PyTorch prosody comparison SHA-256:
  `0baf205a9830d66706a5b0729997dfdce1c49c7754c08133780572586a2e57b8`
- Reloaded-PyTorch-versus-vLLM prosody comparison SHA-256:
  `716f4b66dd994b8ad3596cd0e9f9e00c72a2504aeef395429610789e4f59e342`
- Blind audio manifest SHA-256:
  `cbb709601cbd34956fa792d4cf53d406bd9cb20e77acb88374f2aa572bf60c76`
- Blind review packet SHA-256:
  `90576f1261bedfa9e6215d17c32dff96be7d57fc0618fcc41d9978e82a909baa`

## Decision and next experiment

Keep persisted merged PyTorch as an experimental diagnostic route with
negative content conformance. The bounded evidence clears both in-memory merge
and serialization/reload as the source of the observed divergence, so vLLM
decoding remains the next implicated layer. The evidence does not prove why
the vLLM path differs.

The next runtime experiment should expand the four-way design to pronunciation
and long-form structural prompts, then isolate vLLM sampling and token-decoding
settings one variable at a time. No runtime should be promoted until requested
content passes and the blind review is completed.
