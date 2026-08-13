# vLLM input-batch receipt smoke result

## Outcome

The corrected preregistered two-row GPU smoke passed every instrumentation
acceptance check. Both planned rows produced valid audio and exactly one
complete schema 1.0.0 request receipt from the live
`vllm.v1.worker.gpu_input_batch.InputBatch.add_request` path.

This promotes the request-receipt mechanism from repository-declared to
executed for the frozen vLLM 0.15.1 runtime. It does not promote model quality,
determinism, or runtime equivalence.

## Frozen inputs

- Producing revision:
  `434b7ee8b0d61e7e79ba08e3f05996809ca419a0`
- Instrumentation implementation revision:
  `d86e5c5149ec95c688d4e17dcd77ccb560d50a54`
- Evaluator revision:
  `2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0
- Generation plan SHA-256:
  `2143c7db8fcd0107aceec92d4fa037b076d8b5a159097fd181fe69298171da5b`
- Protocol SHA-256:
  `1d4780e869ecf2bcddcbdc39502fe034a08173aec0581094fa99cbe409dc5e15`
- vLLM export artifact-set SHA-256:
  `db7b3b425b4f8b4353c8111d49f57ebbf3f351a04352eadb3adb2ae80a26c866`
- vLLM export tree SHA-256:
  `624f504ea3558f391279fcbcdc7109a38849ea277bb5ab1ef00323c82a30cc22`

Remote evidence root:
`/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260814_cosyvoice3_vllm_input_batch_smoke_v1`.

Artifact verification passed before and after generation with identical report
SHA-256
`62147031262cfd7236ff64c2027dcdc0486d33768e9daa47ea1bcca8db3e611d`.

## Receipt evidence

The receipt-validation artifact passed with SHA-256
`5d155f73d9316533723328435a721dea117dd7072a02730d015cef596fb0ee5f`.
The combined observation SHA-256 is
`cb99f17eb5735d9452b0f8e1ab16edde0f45cef2349fb654f249f6071a882968`.
The complete-coverage report SHA-256 is
`84e815a8ae307cbc215bcd96e7cd7e7dba98910d4ee189c10a13ced794ecfcc6`.

Both rows recorded temperature 1.0, top-p 1.0, top-k 25, min-p 0.0,
minimum 68 output tokens, maximum 680 output tokens, and 366 yielded output
tokens.

The unchanged upstream row recorded:

- seed source `global_engine_generator`
- engine configuration seed 0
- no request-local generator seed
- token SHA-256
  `81e0ea358d7c2ece499fe26c185d5ce0d66d8c636637e823b7d2d9a3ca7a02a4`
- WAV SHA-256
  `f94a6145c59bc3cb694cd8bebb2aa0b0d7b971a2a99881d0fe98c781950fcff0`

The request-seeded row recorded:

- seed source `supplied_request_seed`
- request-local generator seed 42
- the same token and WAV hashes as the upstream row

The equal hashes are a direct result for this prompt, seed, artifact, and two
fresh processes. They replicate the earlier observation that request seeding
did not change short-prompt outputs. They do not prove general deterministic
execution or show that the generators had equivalent internal state.

## Request identity nuance

The live engine transforms its internal request identifier, so the input-batch
identifier did not equal CosyVoice's wrapper UUID in either row. The receipt is
instead bound by the serial diagnostic scope: the global hooks are active only
until the first yielded token, exactly one input-batch registration is required,
and the outer observation supplies the sample identifier. The implementation
does not retain or claim equality between the two internal identifiers.

This binding is appropriate only for the serial diagnostic runner. It is not a
concurrent-server receipt design.

## Decision and generalisability

The corrected receipt mechanism is ready for a preregistered split-boundary
experiment. That experiment should compare a one-request prefix with the same
prefix followed by a second frontend chunk, use both upstream and
request-seeded profiles, and retain request-level token hashes for every chunk.

Direct receipt values apply only to the frozen vLLM 0.15.1 runtime, artifact,
prompt, seed, and host. The live-path verification method generalizes: fail
closed on missing internal observations, preserve failed smokes, and do not
infer execution merely because an internal API exists in the installation.
