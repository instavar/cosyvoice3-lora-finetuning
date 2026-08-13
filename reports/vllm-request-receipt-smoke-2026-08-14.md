# vLLM request-receipt smoke result

## Outcome

The preregistered two-row GPU smoke failed its instrumentation acceptance
criteria and retained both invalid observations. The proposed
`SamplingStates.add_request` hook was never called. This was a useful
fail-closed result: vLLM 0.15.1 contains that API, but the executed CosyVoice
engine used a different sampling path.

Do not treat the failed receipts as evidence of applied sampling parameters.
They contain zero output tokens, no `applied_sampling` object, and an
`interrupted` status.

## Frozen inputs

- Producing revision:
  `94a8a392bcb44021dbefbd4e867973df869d6397`
- Evaluator revision:
  `2812e200233804fde685c35ea1da1cbf9fe8ef4b`, version 0.41.0
- Generation plan SHA-256:
  `de733193609e350d0f9cf7cc532e6cfe19ec36fba475c6e83e6718d34f4cc5cf`
- Protocol SHA-256:
  `006688246fe885dedba36952b6f0af4b8f1d969e4c79cfa1c9bc5fa653d82331`
- vLLM export artifact-set SHA-256:
  `db7b3b425b4f8b4353c8111d49f57ebbf3f351a04352eadb3adb2ae80a26c866`
- vLLM export tree SHA-256:
  `624f504ea3558f391279fcbcdc7109a38849ea277bb5ab1ef00323c82a30cc22`

Remote evidence root:
`/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260814_cosyvoice3_vllm_request_receipt_smoke_v1`.

Artifact verification passed before and after generation with identical report
SHA-256
`62147031262cfd7236ff64c2027dcdc0486d33768e9daa47ea1bcca8db3e611d`.

## Retained result

Coverage was complete: two expected rows, two observed rows, no missing,
duplicate, unexpected, or mismatched rows. Both rows were invalid. The coverage
report SHA-256 is
`48a0a0dbcfebbb11d6ddb055a1a14ebdc2270ac01be3884c37af58274b1ce524`.
The combined observation SHA-256 is
`012a71484aeb6e9d700a83f1eb9518f677bdd18daea0e8c7c0e8651706681308`.

Both the unchanged upstream profile and request-seeded profile recorded the
same background-thread failure:

```text
expected exactly one vLLM sampling-state capture per request, observed 0
```

The subsequent empty token stream also caused the downstream audio path to
fail on an input shorter than its convolution kernel. That second error is a
consequence of the deliberately retained empty generation, not evidence about
voice quality.

## Root cause

Source inspection after the failure showed two sampling implementations inside
the pinned vLLM installation:

- `vllm.v1.worker.gpu.sample.states.SamplingStates`, the hook assumed by the
  first implementation.
- `vllm.v1.worker.gpu_input_batch.InputBatch`, the path used by the live
  `GPUModelRunner` in this engine.

The live object graph was `LLMEngine` to `InprocClient` to `EngineCore` to
`UniProcExecutor` to `GPUModelRunner`. Its sampler was
`vllm.v1.sample.sampler.Sampler`, and request parameters were materialized in
`GPUInputBatch.add_request`. The similarly named worker sampler and sampling
states API existed on disk but was not on this executed path.

This also corrected the upstream seed model. A request with an explicit seed
gets its own Torch generator. An upstream request with no seed has no
request-local generator and uses vLLM's process-global generator. There is no
runtime-selected upstream request seed to record on this path. The engine
configuration seed can be recorded, but it does not capture the evolving
global generator state.

## Decision and generalisability

The receipt hook must move to `GPUInputBatch.add_request`, record the fields
materialized in that input batch, and distinguish request-local generator state
from the process-global generator path. A new smoke must be preregistered and
executed after the correction. The failed rows must not be replaced in this
evidence root.

The direct root cause applies to the pinned CosyVoice integration and vLLM
0.15.1 object graph. The broader lesson generalizes: the presence and signature
of an internal API do not prove that a configured runtime executes it. Internal
instrumentation needs a live-path smoke before it is used as evidence in a
larger experiment.
