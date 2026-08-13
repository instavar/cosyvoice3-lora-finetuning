# CosyVoice3 corrected emotion-route evaluation, 2026-08-13

## Result

The preregistered corrected-route run completed all 12 planned generations:
unchanged CosyVoice3 Base and the selected epoch-12 LoRA adapter, two
instruction-bearing prompts, and three frozen seeds. Every row was runtime-valid
and recorded the requested instruction as applied through
`inference_instruct2`. This resolves the earlier runner defect that silently
sent instructed rows through `inference_zero_shot`.

The corrected route is not a quality success. All 12 rows failed the frozen
requested-text WER threshold of `0.1`; both candidates had mean WER
`0.666667`. A post-hoc evaluator 0.39 diagnostic additionally found exact
instruction-exclusive two-gram overlap in five of six Base hypotheses and
three of six adapter hypotheses. The route can therefore produce audible words
from a style instruction that was meant to condition delivery rather than be
spoken.

No emotion-obedience or perceptual winner is claimed. The blind pack is staged
but unrated. The objective comparison explicitly records
`proves_adaptation_benefit: false`.

## Scope and preregistration

The frozen protocol is
[`evaluation/preregistration/cosyvoice3-base-epoch12-emotion-route-2026-08-13.md`](../evaluation/preregistration/cosyvoice3-base-epoch12-emotion-route-2026-08-13.md).
It bound:

- public runner revision
  `717bcb239e2ba81c1759426b8dd8135ff861f841`;
- evaluator revision
  `286f7e7052740d6fec9a5a4b5c27a0419ed310fd`;
- prompt pack 1.2.0 canonical SHA-256
  `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`;
- generation plan canonical SHA-256
  `87d8a099385e30b3f08e77b14b41b5039e13f8fe9ced961dfb545c8001a76b51`;
- Base artifact-set SHA-256
  `7c9fb1b5d255cd899bc2b562b1f0b58540f73e24f3902ec5276a535183ed7e54`;
- adapter artifact-set SHA-256
  `7b4b30762864daaa60a355b9445e9722b393ef0b3c472157f4905e74c9dfb2da`;
- epoch-12 PEFT tree SHA-256
  `8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`;
- retained reference WAV SHA-256
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`;
  and
- retained transcript SHA-256
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`.

The generation and preregistered content checks ran without seed replacement or
retry. Instruction-overlap attribution was added only after the ASR output
exposed the failure mode. It is post-hoc characterization, not preregistered
confirmation.

## Runtime execution

Generation ran on an NVIDIA RTX 3090 Ti from a clean detached checkout at the
frozen runner revision. Base and adapter ran in separate fresh processes with
the same reference, text, prompts, seeds, frontend, and instruction route. Only
the adapter process loaded the frozen PEFT tree.

All six rows per candidate recorded:

- `instruction_requested: true`;
- `instruction_route: inference_instruct2`;
- `instruction_applied: true`; and
- an applied instruction equal to its generation-plan value.

Base durations ranged from `5.28` to `10.16` seconds. Adapter durations ranged
from `4.96` to `10.32` seconds. Both candidates had invalid-output rate `0` and
clipping fraction `0` across all six rows. This validates corrected-route
execution for these exact artifacts and inputs. It does not prove that the
route obeyed the intended emotion or kept control text out of speech.

## Requested-text and instruction diagnostics

The preregistered content settings were four-grams, two minimum retained
reference hits, repetition excess over `0.05`, and WER over `0.1`. Every row
failed WER:

| Candidate | Mean WER | WER failures | Repetition failures | Reference-overlap failures |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.666667 | 6 of 6 | 0 of 6 | 0 of 6 |
| Epoch 12 | 0.666667 | 6 of 6 | 1 of 6 | 0 of 6 |

The equal mean hides broad row-level variation. Base WER ranged from `0.333333`
to `1.133333`; adapter WER ranged from `0.4` to `0.866667`. Runtime-valid WAVs
therefore cannot be treated as requested-text success.

Evaluator 0.39 at public revision
`fa4fab5a2499f863d903baa45bfba0f090fe18af` was then applied without changing
the frozen four-gram, retained-reference, repetition, or WER settings. The new
instruction diagnostic used two-grams and a one-hit threshold:

| Candidate | Rows with instruction-exclusive overlap | Total exact two-gram hits |
| --- | ---: | ---: |
| Base | 5 of 6 | 7 |
| Epoch 12 | 3 of 6 | 6 |

The report excludes instruction n-grams that also occur in requested speech and
stores only hit hashes, not raw hypotheses. One Base row and three adapter rows
had no exact hit, but that is not proof that instruction speech was absent.
Recognition errors, garbling, paraphrase, and short common phrases limit exact
n-gram attribution.

## Objective comparison

The matched comparison retained all six prompt-and-seed pairs and all nine
required objective slots. Mean adapter-minus-Base deltas were:

| Metric | Delta | Interpretation boundary |
| --- | ---: | --- |
| WER | approximately 0 | both means were 0.666667 |
| duration | -0.84 s | context only |
| generation time | +0.763241 s | adapter was slower |
| peak allocated CUDA memory | -203,699,968 bytes | adapter used less in this run |
| real-time factor | +0.135411 | adapter was slower relative to audio duration |
| silence fraction | -0.038479 | context only |
| ECAPA similarity | +0.175294 | proxy improvement only |

The adapter's higher ECAPA similarity cannot override universal content-gate
failure. ECAPA is a speaker-embedding proxy, not proof of correct words,
naturalness, accent, or emotion obedience.

## Extractors and execution boundary

ASR used the pinned local
`mobiuslabsgmbh/faster-whisper-large-v3-turbo` revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, artifact-set SHA-256
`3433b5ac25f4b005aadfcde370f3615a5d2883fe40d251e823b80204071115d6`.
The first CUDA attempt was preserved as a failure because CTranslate2 could not
load `libcudnn_ops.so.9`. The same pinned model completed on CPU with `int8`.

Speaker scoring used pinned
`speechbrain/spkrec-ecapa-voxceleb` revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, artifact-set SHA-256
`5a8cd13222e7edf1c932b8695e34c6537c15230e8e47aabe9af454284906dd7c`.
It completed on CPU after the generation environment was found not to contain
SpeechBrain. These fallbacks preserve extractor identity but do not validate a
CUDA extractor runtime.

## Blind listening status

The counterbalanced plan contains 12 samples for two pseudonymous raters. It
asks separately about speaker identity, naturalness, emotion and instruction
obedience, and artifact severity. Accent, pronunciation, cadence, long-form
monotony, and listening fatigue are excluded because this focused prompt slice
cannot support those claims.

No ratings exist. The staged pack proves only deterministic assignment and
identity-neutral presentation. It does not prove delivery, reviewer
independence, attention, emotion obedience, or preference.

## Evidence hashes

The retained remote evidence root is
`/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260813_cosyvoice3_emotion_route_v1`.
Important file SHA-256 values are:

- generation observations:
  `4d390e05baba5bdf466ad57524f7ea746896d5b77b6d65619b52e0dfd886cc3a`
- generation attempt receipt:
  `c7751e3d3323e47076a934b90ac8194e1f6374dee936725c733e30e2e8fa1e9b`
- complete observations:
  `d9192464549300ae008a569dd256070c175ef6cc0e1334430dc0ee0c278d4956`
- preregistered content report:
  `fb1a4b483c94d5ee6f8026ef7ba8ea7c59510f206e5c68a67272b55bb8021c73`
- evaluator 0.39 post-hoc content report file:
  `e64ad725a220154f5e6937aa817dec5658ad1075403e290276ed979695d963b6`
- evaluator 0.39 post-hoc report self-hash:
  `88c8a91de38c622e4e5ae6f83c083bf19467de3c19223c5b80434359505912e4`
- objective report:
  `bbf68c5d93d087fbd42be63e6511c54c89e7f673a480590957ffaefb4e4d43c3`
- matched objective comparison:
  `9d2db2256fcf18b3015d8060ab4fc51b10882049fe468f2b8d9803d71884607f`
- listening assignment:
  `2346b6186f03551d1fe1741d89980e104cc35b0f528b07fca1ab860132cdd2f3`
- blind review document:
  `a910ca0f88b6dbf143ce7714f14487bf48b2a98fccfe54d55530a5e15b5ca4bf`
- blind reveal:
  `bc9c2e498a1ba52f72d1ea0a839571c883b00d0e0523efa3cffeb9a8937fb202`
- listening stage manifest:
  `d56a6ed9eb20090972b950c0cee28201c29193e51158b383f5063712c82a609a`

## Decision and applicability

The historical conclusion changes in one important way: the earlier 0.04-second
emotion rows were a runner-route defect, not evidence that
`inference_instruct2` itself could not execute. The corrected route executes but
still fails requested-text fidelity on every tested row and often exposes
instruction words in ASR.

This is strong negative evidence for the exact Base tree, epoch-12 PEFT tree,
two English instructions, shared requested text, three seeds, retained speaker
reference, upstream route, and PyTorch environment. It generalizes to the
evaluation rule that control-route success and valid audio must be gated by
content diagnostics. It does not establish a failure rate for other prompts,
languages, speakers, checkpoints, runtime revisions, or CosyVoice models.

The next promotion gate is not another post-hoc seed search. It is a new frozen
prompt set that separates instruction wording from requested speech, retains
invalid and leaky rows, and obtains blinded instruction-obedience ratings. Any
prompt rewrite or frontend change needs a fresh preregistration and cannot be
combined with this run as if it were the same intervention.
