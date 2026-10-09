# Agent Instructions

Adapted from `/home/coral/repos/DGXSpark-Small-LLMs/AGENTS.md` and its
benchmarking and experiment-journal guidance for this repository's scope.

- Use Rust for this project and Cargo for dependency management, builds, and
  command execution. The initial target is DGX Spark; focus on BF16, FP32, FP8, and FP4.
- Do not add tests unless the user explicitly asks for tests or the change is a
  regression fix that needs a regression test.
- Keep documentation proportional to implemented behavior and observed results.
  Avoid speculative implementation or troubleshooting documentation.
- Keep package specifications, design notes, usage guides and attribution
  documents in `docs/`. Keep `README.md`, `LICENSE` and `AGENTS.md` at the
  repository root. Keep experiment-specific documents in their external
  experiment workspace, as described below.
- Record important assumptions, tradeoffs, and user corrections in the relevant
  authoritative documentation. Update that documentation when direction changes;
  do not maintain a duplicate correction history.

## Experiment workspaces

- The reusable package's implemented scope is defined in `docs/SPEC.md`.
  An experiment has its own intent, specification and plan; do not replace
  the package's purpose or defaults with an experiment's research questions.
- The GB10 hardware-to-GEMM-shape study is one experiment,
  `GB10-GEMM-SHAPES-001`, maintained outside this package at
  `/home/coral/inference-artifacts/flops-ceiling/experiments/gb10-gemm-shapes/`.
  Read that workspace's `AGENTS.md`, `intent.md`, `spec.md` and `plan.md` when
  working on the study. Inventory, measurement, confirmation and profiling are
  phases of the same experiment, not separate experiments.
- Every time the conversation discusses that experiment's intention, update
  its external `intent.md` before ending the turn, including clarifications,
  reaffirmations and changes. Refresh its review date using the user's timezone
  and reconcile its spec/plan when affected. Do not wait for another request.
  Distinguish the user's agreed direction from proposals and unresolved questions;
  maintain the current account rather than a duplicate correction history.
- Keep experiment `intent.md` files short and limited to the user's explicitly
  stated purpose, questions and scope. Put technical elaboration, assumptions,
  proposed controls and completion criteria in the experiment's `spec.md`, and
  execution steps in its `plan.md`. Label agent proposals as proposals; do not
  present them as user intent.
- Keep experiment-specific drivers, analysis and documents in that workspace.
  Reusable package changes remain normal package work and must document their
  implemented behavior. Consolidate verified results in `docs/final-report.md`
  with the experiment ID and links to external evidence.

## Performance experiments

- Separate warmup from measurement. Complete compilation and initialization
  before timing, and record warmup and measurement boundaries.
- Declare repetition counts or convergence criteria before running. Label a
  single pass preliminary; repeat measurements and report variability before
  declaring a performance winner.
- Compare configurations with matched workloads, inputs, timing boundaries,
  warmup, and cache conditions. Change one setting at a time where possible and
  disclose other changes needed for the comparison.
- Record the exact command, repository revision, hardware, driver, CUDA and Rust
  toolchain versions, dependencies, precision, workload dimensions, configuration,
  UTC timestamps, and exit or error state.
- State the FLOP-counting method and timing scope. Distinguish measured kernel
  throughput, estimated model-operation rates, and advertised hardware peaks.
  Historical SGLang estimates are not measurements of this project's TFLOPS ceiling.
- Collect supported GPU telemetry, including temperature, utilization, and power.
  Record unavailable measurements explicitly; missing measurements are not zero.
- Preserve raw measurements and failed attempts in unique artifact directories.
  Keep large raw artifacts outside Git and disposable worktrees, and record their
  locations in `docs/final-report.md`.

## Experiment records

- Keep **one maintained results report in the repository**:
  `docs/final-report.md`. Consolidate verified conclusions, comparison scope,
  limitations and external evidence locations there. Keep usage/API guides
  focused on implemented behavior rather than duplicating result tables.
- Store detailed results, append-only experiment journals, imported reports and
  research notes outside the repository, under
  `/home/coral/inference-artifacts/flops-ceiling/` on this host. Keep them out of
  Git; local scratch output belongs in ignored `artifacts/`, `experiments/` or
  `results/` directories. Do not recreate `docs/experiments/` or `docs/research/`.
- Organize external journals by hardware and precision with a `README.md` index.
- Store each experiment attempt or follow-up in its own UTC-timestamped file:
  `<YYYY-MM-DDTHH-MM-SSZ>-<short-slug>.md`.
- Assign the next directory-local ID (`RUN-0001`, `RUN-0002`, and so on), place it
  immediately below the title, and list it in the index. Never reuse or renumber
  IDs; keep the recorded count and next ID in the index.
- Record observed build, GPU discovery, kernel compilation, execution, and
  validation failures before ending the work session, even if unresolved.
  Include the redacted command and error, environment, evidence-based diagnosis,
  attempted fix, verification result, status, and reusable lesson.
- Preserve chronology. Link a new follow-up record to an earlier attempt instead
  of adding later findings to the earlier file. Never fabricate historical runs
  or record credentials or other secrets.
- Keep imported reports in the external archive as historical references.
  Preserve their original dates, run IDs, results, and limitations; do not count
  them as experiments performed in this repository.
