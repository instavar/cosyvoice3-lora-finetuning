# vLLM split-boundary probe result

## Outcome

The preregistered 18-row GPU probe separated frontend request count from text
length and replicated the request-generator mechanism across all three seeds.
Every row had valid audio, exact frozen-plan coverage, calibrated frontend
segment hashes, a matching live vLLM request count, and complete request-level
token receipts.

Within both upstream and request-seeded profiles, combined request one matched
the standalone 64-token prefix in token count and SHA-256 for all three seeds.
Within request-seeded, combined request two also matched the standalone
20-token tail for all three seeds. Within upstream, combined request two
matched the same frontend tail hash but differed from the standalone tail token
hash for all three seeds after request one advanced the process-global
generator.

This confirms the preregistered mechanism for the frozen runtime. It does not
promote quality. Both profiles failed the primary content-faithfulness gate in
eight of nine rows.

## Chronology and frozen evidence

- Instrumentation revision:
  `a4df080810c225f54a4e5142eacc403cc27f310e`.
- Preregistration revision:
  `1fb9b73528521c463dcf5bfa56ac70cda7da3ea9`.
- Receipt-validator revision:
  `67ec542f8f6e3e7c603f9e595f367cc7c25c4032`.
- Evaluator revision:
  `2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0.
- Hosted instrumentation CI: run `31734483431`, passed.
- Hosted preregistration CI: run `31735196684`, passed.
- Hosted validator CI: run `31736680414`, passed.
- Generation-plan file SHA-256:
  `f3cf6fcac69a278708509b84f48c2459cafaa0794cb95414476c9dab890905ee`.
- Protocol SHA-256:
  `15862e5dd260dc4de3586101c23d4a4ba7a5e08852801e48bae15c7de63ef6c6`.
- Prompt-specification file SHA-256:
  `98a24807af374649eb93a58aa11a7650f9e8e47a532e51118b4d9b1a6da27fb8`.
- Retained evidence root:
  `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260814_cosyvoice3_vllm_split_boundary_v1`.

Every sample ran through the exact `--sample-id` selector in its own Python and
vLLM process. The retained reference WAV and transcript matched their frozen
SHA-256 values. The vLLM export verification passed before and after generation
with identical report SHA-256
`62147031262cfd7236ff64c2027dcdc0486d33768e9daa47ea1bcca8db3e611d`.

## Calibration and live request coverage

The actual Fun-CosyVoice3-0.5B frontend was calibrated before synthesis:

| Prompt | Frontend segments | Token counts | Segment identity |
| --- | ---: | --- | --- |
| Prefix | 1 | 64 | prefix hash |
| Tail | 1 | 20 | tail hash |
| Combined | 2 | 64, 20 | exact prefix hash, exact tail hash |

The combined prompt deliberately has no whitespace after the prefix period.
This preserves exact normalized segments and makes it a mechanical boundary
probe rather than a naturalness prompt.

All 18 live observations matched the calibration and vLLM request count. The
tested validator retained:

- raw observations SHA-256:
  `69fa1481421e2bddf5b63aaed68dc5483e1e01981d5fde067c16ab9605245b05`;
- receipt-validation SHA-256:
  `650a4b85a257ba55db05734e7ecc065dba9188c3655d8e17111b592ac6938cb1`;
- complete-coverage SHA-256:
  `3ab23cae09745a832a3afe8f060c9291ebe004a2b16475e971ec3d3918d22cf9`.

The receipt result is exact across the three frozen seeds:

| Profile | Prefix standalone versus combined request 1 | Tail standalone versus combined request 2 |
| --- | --- | --- |
| Upstream global generator | 3 of 3 token hashes equal | 0 of 3 token hashes equal |
| Request-local seed | 3 of 3 token hashes equal | 3 of 3 token hashes equal |

The upstream tail mismatch is not frontend drift. The frontend tail segment
hash and token count stayed exact, while the output-token count and hash
changed after the prefix request consumed global generator state. Request-local
seeding reset the generator for the second request and reproduced the
standalone tail identity.

## Waveform and content evidence

Prefix and tail WAVs were byte-identical between upstream and request-seeded
for every seed. Combined WAVs differed for all three seeds, which is consistent
with the second request-token divergence. Matched waveform results remain
downstream observations and do not make the request receipts a proof of kernel
execution.

The complete evaluator artifacts have these SHA-256 values:

- observations with all extractor and runtime receipts:
  `2ddfd65a1e6d213f8d6152d7ab5f4a355d54e9cec7a99a816eb3f98cb8cba62d`;
- objective report:
  `011cba311c81197c45f54f055cfda0945d434ac2c6aa40a0c78588bab47d670e`;
- matched comparison:
  `b337936cbed42caf32f6f36f0a4f77765d6be7e9100bad89cb6db0b8255e3ced`;
- content-faithfulness report:
  `64821dcd0f751e26752ace45c6d2859316a82282bf95f3a08c3de44cff762b7d`;
- prosody comparison:
  `d911ec75bd4daf8cff6904b88741f8d7f6096c0d8329bc45417ae441edbc62d9`.

| Objective mean | Upstream | Request-seeded |
| --- | ---: | ---: |
| Requested-text WER | 0.447340 | 0.436830 |
| Speaker similarity | 0.855118 | 0.854916 |
| Duration, seconds | 19.120000 | 19.600000 |
| Generation time, seconds | 3.963953 | 4.072297 |
| Real-time factor | 0.215223 | 0.216112 |
| Peak GPU allocation, bytes | 7,226,327,779.56 | 7,226,327,779.56 |

Both candidates had eight failed content rows and one unflagged tail row. The
overall matched WER delta was -0.010511 for request-seeded minus upstream, with
a 95 percent bootstrap interval from -0.036036 to 0.015015. The interval
crosses zero, and every prefix plus most tail and combined rows still exceeded
the frozen content threshold. On combined rows alone, mean WER was 0.472973
upstream and 0.441441 request-seeded. This is a bounded diagnostic difference,
not an adaptation or quality benefit.

Prosody proxies were identical where WAV bytes were identical and changed on
the combined rows. They do not establish improved cadence, lower monotony,
accent fidelity, naturalness, or preference. This mechanical probe does not
need a blind listening pack because the primary content gate already blocks
promotion and its deliberate prompt boundary is unsuitable for naturalness
judgment.

## Decision and generalisability

For this frozen CosyVoice3 and vLLM 0.15.1 path, the earlier long-form
request-seed association is explained by frontend request boundaries rather
than text length alone. A process-global generator carries state from one
frontend request into the next. Supplying the same request-local seed resets
each request and reproduces an identical standalone segment's output-token
identity.

The mechanism applies directly only to the frozen epoch-12 export, frontend,
reference, prompts, seeds, host, and pinned vLLM internals. The evaluation
method generalizes to segmented generation systems: calibrate exact frontend
chunks, isolate processes, capture request-level state, compare identical
segments, and keep content quality as a separate gate. The specific hashes,
WER values, and generator behavior do not generalize to other versions or
models without replication.

The next runtime gap is not whether request seeding takes effect. It does. The
remaining questions are whether a production policy should intentionally reset
each frontend request, whether a per-utterance seed schedule can preserve
controlled diversity across chunks, and whether either policy improves content
under a checkpoint that first passes the content gate.
