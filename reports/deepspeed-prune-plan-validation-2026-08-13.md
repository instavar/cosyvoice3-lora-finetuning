# DeepSpeed prune-plan validation - 2026-08-13

## Outcome

Commit `dceab3d901bf8bb395945ca43b1fa2685925cd73` replaced implicit
YAML-stem deletion with a two-phase legacy DeepSpeed pruning contract. Plan
creation requires explicit checkpoint tags and never deletes. Execution
requires the exact reviewed plan digest and revalidates the plan, pruner source,
model-directory identity, and every planned filesystem object before staging.

Hosted Instavar Voice contract run `31662101911` completed successfully. GitHub
recorded the run at `2026-08-13T02:51:32Z`.

## Failure modes covered

- unsafe and overlong tags
- metadata-only tags without a checkpoint payload
- malformed, duplicate-key, oversized, non-finite, or unsupported YAML
- undeclared YAML dependencies on clean Python runtimes
- `keep_best` values greater than one
- terminal symlinks and hard-linked regular files
- mismatched plan confirmations and modified plan content
- pruner source drift, model-directory replacement, inode drift, and byte drift
- foreign components added after planning
- aliased lock files and concurrent cooperative pruners
- interrupted component staging and partially completed removal
- unrelated files and unadopted checkpoint namespaces

## Validation

- 72 dependency-free repository tests passed on Python 3.11 and Python 3.14.
- Ruff 0.12.12 passed for the pruner and its tests.
- Python compilation passed for all repository tools and tests.
- The pinned Instavar Voice evaluator accepted the capability manifest and
  lifecycle backend.
- The hosted Instavar Voice contract passed at the commit above.

## Boundary

No real CosyVoice model, GPU, training process, live DeepSpeed checkpoint,
network filesystem, process crash, power failure, or operator review session was
used. Deletion tests operated only on temporary fixtures. The lock is advisory
and cannot stop a writer that ignores it, so operators must stop training and
synchronization jobs before planning and execution. Exact inode binding is a
local-filesystem contract and may not have equivalent semantics on every remote
or distributed filesystem.

The implementation does not infer ownership from numeric names, YAML fields, or
directory discovery. Explicit adoption is still an operator assertion. The
content-bound plan makes that assertion reviewable and rejects drift; it does
not prove that the operator selected the correct checkpoints.
