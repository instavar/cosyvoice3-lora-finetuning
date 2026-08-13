# Background-thread failure capture validation

Date: 2026-08-13

## Finding

CosyVoice generation can execute LLM decoding in a worker thread. An uncaught
worker exception does not automatically fail the parent process. Earlier real
runs demonstrated the dangerous result: a roughly 0.04-second WAV could be
written while the log contained the actual LLM failure and the command returned
success.

Duration, peak, and RMS guards catch the observed short-output symptom, but the
symptom is not the cause. A worker could fail after emitting enough samples to
pass those thresholds. Future inference and evaluation now capture uncaught
thread failures while each output stream is consumed.

## Behavior

The shared helper temporarily installs `threading.excepthook`, retains the
previous hook so the traceback still reaches the log, and restores it after the
bounded generation scope. Each captured failure records a bounded thread name,
exception type, and message.

The frozen evaluation runner:

- records the capture method on every observation;
- marks a row invalid when one or more worker failures occur;
- records structured `background_thread_failures` evidence;
- preserves any generated WAV; and
- returns failure unless the caller explicitly selected the existing
  `--allow-invalid-output` evidence-retention mode.

The standalone inference helper also preserves generated audio, then raises in
the parent process when a worker failure was captured.

## OOD controls

Dependency-free tests create a named worker that raises an uncaught exception,
verify its structured record, verify restoration of the previous hook, and
verify the no-failure path. Source-contract tests also require both production
entry points to retain the capture boundary.

The complete companion suite passed 80 tests locally with the pinned evaluator
available on `PYTHONPATH`. Focused Ruff F checks passed and the new helper tests
passed Ruff formatting checks. Implementation commit
`0c6f7b0b17aeb85a71d9fff335e081fe6bde0d9e` passed hosted Instavar Voice
contract run `31682801829` on 2026-08-13.

## Scope and boundary

The capture generalizes to uncaught Python thread exceptions that occur while a
generation stream is being consumed in a fresh isolated process. Because
`threading.excepthook` is process-global, this design relies on the runner's
single-generation process contract and should not be reused inside a concurrent
multi-request server without a request-aware dispatcher.

The guard does not observe exceptions swallowed by worker code, failures before
or after the bounded stream-consumption scope, native crashes, subprocess
failures, hangs, or semantically bad audio produced without an exception. Log
inspection, signal validation, objective metrics, and blind listening remain
separate evidence layers.
