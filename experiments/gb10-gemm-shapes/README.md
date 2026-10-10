# GB10 GEMM shapes experiment

Experiment ID: `GB10-GEMM-SHAPES-001`.
Last reviewed with the user: 2026-10-10 (Europe/London).
Status: designed; this experiment has not been executed.

This is one experiment using the `flops-ceiling` package to investigate how
GB10's hardware organization relates to GEMM shape and performance. It is
separate from the package's general purpose and implementation specification.

- [intent.md](intent.md): why we are doing this and the questions to answer.
- [spec.md](spec.md): scope, required evidence and completion criteria.
- [plan.md](plan.md): the execution phases and measurement protocol.
- [AGENTS.md](AGENTS.md): instructions for agents working on this experiment.

Inventory, implementation preparation, shape measurements, confirmation,
profiling and synthesis are phases of this same experiment. A phase, datatype,
trial or retry is not a new experiment. All artifacts and run records should
carry the experiment ID and identify their phase and arithmetic mode.

## Workspace and package boundary

The maintained workspace is `experiments/gb10-gemm-shapes/` in this Git
repository. Version the intent, specification, plan, agent instructions and
any experiment-specific Rust/Cargo drivers or analysis source here.

The [package specification](../../docs/SPEC.md)
and API guides describe implemented reusable behavior. This experiment's
requirements do not automatically become the package's scope or roadmap.
Record the actual revision and source used for execution.

Raw measurements, generated analysis, profiler captures and append-only journals
stay outside Git at
`/home/coral/inference-artifacts/flops-ceiling/experiments/gb10-gemm-shapes/`
on the measurement host. Keep each raw attempt in a unique UTC-timestamped
`runs/` directory there, with journal records under
`journals/dgx-spark/<precision>/`, README indexes and RUN IDs. The external
evidence directory and its subdirectories are created when needed; there are
no new measurements yet. Local build and output directories are ignored.
The Git-tracked files here are the maintained experiment documents.

Verified findings will be summarized in the package's existing
[single results report](../../docs/final-report.md)
under this experiment ID, with links to this workspace's evidence. Existing
measurements remain historical references; they are not runs of this experiment.
