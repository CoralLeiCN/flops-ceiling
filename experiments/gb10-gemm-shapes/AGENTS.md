# Instructions for this experiment

These instructions apply to `GB10-GEMM-SHAPES-001` in this workspace. The
`flops-ceiling` package has its own general instructions and specification.

- Read `intent.md`, `spec.md` and `plan.md` before working on this experiment.
- Whenever the user discusses this experiment's intention, update `intent.md`
  before ending the turn. Capture clarifications, reaffirmations and changes,
  and refresh the review date using the user's timezone. Update the spec/plan
  when their scope or procedure is affected; do not wait for another request.
- Keep a current account, not a duplicate correction history. Distinguish the
  user's agreed intent from agent proposals and unresolved questions.
- Keep `intent.md` short and limited to the user's explicitly stated purpose,
  questions and scope. Put technical elaboration, assumptions, proposed controls
  and completion criteria in `spec.md`, and execution steps in `plan.md`.
  Label agent proposals as proposals; do not present them as user intent.
- Treat inventory, preparation, measurement, confirmation, profiling and
  synthesis as phases of one experiment. Keep its ID on every run/record.
- Do not rewrite the reusable package's purpose or scope to match this study.
  Version experiment documents, drivers and analysis source here. Package changes
  should implement reusable capabilities and follow that package's instructions.
- Use Rust and Cargo. Separate compilation/initialization/warmup from timing,
  predeclare repetitions, match comparison conditions, validate outputs and
  collect supported telemetry. Preserve failures as well as successful runs.
- Store raw measurements, generated analysis, profiler captures and journals at
  `/home/coral/inference-artifacts/flops-ceiling/experiments/gb10-gemm-shapes/`
  on the measurement host. Preserve unique UTC-timestamped raw attempts under
  that external workspace's `runs/`. Use its
  `journals/dgx-spark/<precision>/` for append-only records with README indexes,
  directory-local sequential RUN IDs immediately below each title, recorded
  counts and next IDs. Never renumber IDs or fabricate past measurements.
- Record exact commands, source revision/diff, toolchain/library/device data,
  timing boundaries, exit state, and evidence-based diagnosis/fix/verification
  for failures. Never store credentials. Missing telemetry is unavailable,
  not zero.
- Keep generated evidence outside Git and disposable worktrees; the maintained
  intent, specification, plan, instructions and source belong in this repository.
  Use the existing [package results report](../../docs/final-report.md) for
  verified results and external evidence locations. Historical runs remain
  historical.
