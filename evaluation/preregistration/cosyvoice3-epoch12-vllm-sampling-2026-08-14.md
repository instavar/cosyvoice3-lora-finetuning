# CosyVoice3 epoch-12 vLLM sampling preregistration

Date: 2026-08-14 Asia/Singapore, cross-checked against GitHub Actions server
timestamp `2026-08-13T17:47:17Z` from run `31727455686`.

## Question

Do a per-request seed and then top-p 0.8 explain any of the retained vLLM
divergence, and do those effects transfer across local context, names and
numbers, and structured long-form text?

## Frozen plan and protocol

The 45-row generation plan is
`cosyvoice3-epoch12-vllm-sampling-plan-2026-08-14.json`, SHA-256
`dee7c76b2de146822ca9284027562af633732e8b0fb650eacb849ace9bdf3183`.
The machine-readable protocol is
`cosyvoice3-epoch12-vllm-sampling-protocol-2026-08-14.json`, SHA-256
`d266a461770cf3a14a4ed53786158c97fc38ee204646edec80cb6849007ce9e9`.

The source prompt pack is evaluator version 1.2.0 with SHA-256
`6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`.
The selected prompts are `local-context`, `names-numbers`, and
`structured-long-form`. Every condition uses seeds 42, 314159, and 20260812.
Failures remain in the matrix and are not replaced.

The producing companion revision is
`19dded017efbd901abf0547330f7296cea757b17`. The evaluator revision is
`2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0.

## Conditions

Each condition starts in a fresh process and executes in this fixed order:

1. Reloaded merged PyTorch comparator.
2. vLLM `upstream-a`: top-k 25, vLLM defaults for top-p and request seed.
3. vLLM `request-seeded`: upstream request plus only the plan-row seed.
4. vLLM `request-seeded-top-p-0.8`: request seed plus only top-p 0.8.
5. vLLM `upstream-b`: an unchanged repeat of condition 2 after the profile
   runs.

The last condition is a falsification control. If `upstream-a` and
`upstream-b` drift as much as a profile transition, the profile transition
must not be interpreted as a clean seed or top-p effect.

## Artifact boundaries

The reloaded merged PyTorch tree is frozen at SHA-256
`6d93ca3d777e3c357dae495cd2cd038c7f040ed53b5a3fdaf43fb61ed6b0221c`.
The retained vLLM export tree is frozen at SHA-256
`624f504ea3558f391279fcbcdc7109a38849ea277bb5ab1ef00323c82a30cc22`.
Both are derived from the epoch-12 adapter tree at SHA-256
`8bd5e1ca24c71a099cec6612230c9c448cbcff0c3f0ee15ad7a15b26e7bd2bb8`.

The full paths, artifact-set hashes, and relations are frozen in the protocol.
Verification must run before and after generation. Derived bindings do not
establish exact-artifact runtime equivalence, and
`proves_runtime_equivalence` remains false.

## Hypotheses and falsifiers

- Request seed effect: `request-seeded` differs from `upstream-a` by more than
  the observed `upstream-a` versus `upstream-b` rerun drift.
- Top-p effect: `request-seeded-top-p-0.8` differs from `request-seeded` by
  more than the unchanged-upstream rerun drift.
- Transfer: any effect appears across more than one prompt category rather
  than only one row or prompt.
- Falsifier: unchanged upstream reruns differ materially or the recorded
  effective request parameters do not match the protocol.
- Falsifier: artifact verification fails, coverage is incomplete, or any
  failed generation is silently replaced.

The experiment can localize a request-parameter association. It cannot prove
vLLM internal sampler behavior, causal sufficiency, deterministic execution,
or PyTorch equivalence. PyTorch still uses CosyVoice repetition-aware sampling,
which these vLLM controls do not reproduce.

## Primary gate

Content faithfulness remains primary. A condition fails promotion if any
planned row is unevaluable or has requested-text WER above 0.10, repeated
four-gram excess above 0.05, or at least two reference-exclusive four-gram
hits. Runtime speed, memory, speaker similarity, signal validity, prosody
proxies, and listening ratings cannot override a failed content gate.

## Secondary evidence

For every valid row, retain exact WAV hashes, audio probes, faster-whisper ASR,
SpeechBrain ECAPA similarity, prosody proxies, generation time, real-time
factor, and peak GPU allocation. Produce matched comparisons for every adjacent
condition and a direct `upstream-a` versus `upstream-b` comparison. Stage a
counterbalanced blind pack only after objective evidence is complete.

Pronunciation, Singapore English accent fidelity, cadence variation,
naturalness, and listening fatigue require human review. ASR recognition of
`paiseh`, a name, number, or address is not evidence of correct pronunciation.

## Scope

Direct findings apply only to the frozen epoch-12 artifacts, reference,
selected prompts, three seeds, vLLM 0.15.1, recorded GPU host, and exact
producer and evaluator revisions. The content-first decision rule and the need
for an unchanged rerun control generalize to other stochastic runtime
diagnostics. Metric values and any profile effect do not generalize without
replication.
