# Three-way adapter, merged PyTorch, and merged vLLM comparison

Date: 2026-08-13

## Result

The preregistered three-way experiment narrows the observed vLLM divergence to
a layer after the in-memory PEFT merge in this tested slice.

Adapter PyTorch and merged-in-memory PyTorch produced byte-identical WAV files
for all three seeds. Their ASR, speaker, audio, and prosody values were also
identical. The retained merged vLLM export produced different WAVs, higher
requested-text error, more repetition, longer output, different prosody proxy
values, and higher peak GPU allocation.

All three conditions failed the primary content-faithfulness gate. This is a
diagnostic localization result, not a quality promotion. It does not separate
export serialization from vLLM decoding and does not establish general runtime
equivalence.

## Evidence chronology and scope

- Implementation commit:
  `20a38a9cc6cfddb56434789147a46a681645bca2`
- Preregistration commit:
  `bdce608d2a782fc48b305ea79c9c1bbca64c8049`
- Generation plan SHA-256:
  `4b3ef499de4fb05b9e40017f7a1559223df18d6206126d78b30d19ad46de3e4b`
- Evaluator revision and version:
  `fa4fab5a2499f863d903baa45bfba0f090fe18af`, version 0.39
- Prompt count: 1
- Seeds: 42, 314159, 20260812
- Planned and completed rows: 9
- Invalid rows: 0

The experiment applies directly to the selected epoch-12 adapter, one neutral
Singapore English prompt, the retained reference, the recorded runtimes, and
the three seeds. The promotion rule that a content failure outranks speed
applies generally. The waveform and metric results do not generalize to other
prompts, adapters, exports, hardware, or vLLM versions without replication.

## Conditions

| Candidate | Route | Artifact boundary |
| --- | --- | --- |
| Adapter PyTorch | PEFT adapter through PyTorch | Exact source adapter binding |
| Merged PyTorch | `merge_and_unload` in memory, then PyTorch | Derived in memory, no persistent post-merge artifact binding |
| Merged vLLM | Retained exported weights through vLLM 0.15.1 | Derived export binding |

The missing persistent fingerprint for the in-memory merged model prevents an
exact-artifact runtime-equivalence claim. Every comparison therefore uses
matched prompt and seed semantics and records
`proves_runtime_equivalence: false`.

## Objective means

| Metric | Adapter PyTorch | Merged PyTorch | Merged vLLM |
| --- | ---: | ---: | ---: |
| Requested-text WER | 0.483871 | 0.483871 | 0.774194 |
| Speaker similarity | 0.746955 | 0.746955 | 0.815717 |
| Output duration, seconds | 11.866667 | 11.866667 | 17.586667 |
| Generation time, seconds | 4.340178 | 3.745605 | 3.255866 |
| Real-time factor | 0.377548 | 0.326880 | 0.187522 |
| Peak GPU allocation, bytes | 3,632,279,723 | 3,623,755,776 | 7,243,958,272 |

Merged PyTorch reduced mean generation time by 0.594573 seconds relative to
adapter PyTorch while emitting the same bytes. This timing difference can
reflect wrapper overhead or warm-runtime effects and is not a quality result.

Merged vLLM reduced real-time factor relative to merged PyTorch by 0.139358,
but used about 3.62 GB more peak allocation and emitted audio 5.72 seconds
longer on average. The three-pair WER delta was +0.290323, with a wide bootstrap
interval that crossed zero. The content gate, not the interval estimate, blocks
promotion because every individual row exceeded its preregistered WER
threshold.

## Content-faithfulness gate

| Candidate | Gate | High-WER rows | Reference-overlap rows | Repetition rows |
| --- | --- | ---: | ---: | ---: |
| Adapter PyTorch | failed | 3 | 2 | 1 |
| Merged PyTorch | failed | 3 | 2 | 1 |
| Merged vLLM | failed | 3 | 1 | 2 |

The identical PyTorch hypotheses include requested-text repetition and retained
reference phrases. vLLM contains a different and more severe corruption pattern
for this slice. Because all conditions fail, the result does not validate the
adapter or the merged route as production quality.

## Exact audio and prosody

For seeds 42, 314159, and 20260812:

- Adapter PyTorch SHA-256 equaled merged PyTorch SHA-256.
- Merged PyTorch SHA-256 did not equal merged vLLM SHA-256.
- Adapter PyTorch SHA-256 did not equal merged vLLM SHA-256.

Every adapter-versus-merged-PyTorch audio-derived metric and prosody proxy delta
was exactly zero. Every recorded merged-PyTorch-versus-vLLM prosody proxy had a
nonzero mean delta. Signal proxies do not establish better cadence,
naturalness, accent, or preference.

## Rerun determinism observation

The retained vLLM export was also generated in the preceding two-way run with
the same three seeds. The rerun preserved each output duration exactly but
changed all three WAV hashes. Deterministic audio comparison recorded small
nonzero level and silence-statistic deltas, so the difference is not only a
path or filename change.

This is bounded evidence that the recorded seed does not guarantee byte-level
vLLM reproducibility in this setup. The broad content pattern remained similar,
but no numerical-determinism claim is made. Cross-run PyTorch outputs are not
used for this conclusion because the earlier run used a different companion
revision and execution boundary.

## Blind listening

Nine identity-neutral audio files were staged with two counterbalanced rater
schedules. The applicable criteria are speaker identity, Singapore English
accent fidelity, naturalness, and artifact severity. The pack remains unrated.
Ratings may describe perception, but they cannot override the content gate.

## Retained evidence

The raw evidence remains on the controlled GPU host under
`voice-model-outputs/conformance/20260813_cosyvoice3_three_way_v1`. Generated
voice audio and large extractor outputs are not committed.

- Generation receipt SHA-256:
  `aab527c500efb3846d02895194e0fabc53e0cf5a34bf95666d7f997d3d944b62`
- ASR results SHA-256:
  `1afa026dafa489ef0575efe2d55670feddac23e0667a0953f877111da82c8e86`
- Speaker results SHA-256:
  `e3e8241815007c7265543d2680d2be661352a9f2eb79fe20e532615f9607a79e`
- Objective report SHA-256:
  `8d029974d9e2cb5f2d332512a3423d9ffdda941bb7e0741bec79196abf440ed5`
- Content report SHA-256:
  `afbd358dba62fb903a9aada4f430c0e0e9477dc860800daf174b39a05e97c86d`
- Adapter-versus-merged-PyTorch objective comparison SHA-256:
  `5efd0a161b2aeb0725b93f72647de4802d9ad3278ed0f013a994ec49912c38de`
- Merged-PyTorch-versus-vLLM objective comparison SHA-256:
  `95a95d3113d3b7b6f7a364ae5c77eb5571467433e75d949d9048b4325c1b1945`
- Adapter-versus-merged-PyTorch prosody comparison SHA-256:
  `e73c5e16b0142eb2fb819e9e1ac99b141e36f87b471a386d6b1c0a55a4e1bbf2`
- Merged-PyTorch-versus-vLLM prosody comparison SHA-256:
  `f6f2de996d0719212a93651b5ad8b9ec1620d72dff3a01b6d391fae162c7eb69`
- Exact WAV comparison SHA-256:
  `141f3fc218112d5cec00bb558135060b05a4ac622f728ac54545b16f859578cb`
- Blind audio manifest SHA-256:
  `5f8f0e17b8d68f65e69c6739f44e4d17e705bb5070b28aa2336be9c4fb0c0edb`
- Blind review packet SHA-256:
  `94e3a3e310c92d489189cbf5f7834721f1b51c8c1fc67b9518e6ac54e485d5df`

## Decision and next experiment

The in-memory merge is not the source of the observed divergence in this
bounded reproduction. Keep the merged-PyTorch mode for future diagnosis and
retain vLLM as experimental negative-conformance evidence.

The next experiment should persist and fingerprint a reloadable merged
PyTorch artifact, load that exact artifact through PyTorch in a fresh process,
and compare it with vLLM. That separates serialization from runtime decoding.
It should also expand beyond the neutral prompt to pronunciation and longer
structural prompts. No runtime is promoted until content faithfulness passes.
