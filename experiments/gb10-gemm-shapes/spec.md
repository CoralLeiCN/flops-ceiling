# Experiment specification: GB10 GEMM shapes

Experiment ID: `GB10-GEMM-SHAPES-001`.
Status: prospective requirements; not a description of implemented package APIs.
Purpose: answer the questions in [intent.md](intent.md) through one experiment.
The measurement procedure is maintained in [plan.md](plan.md).

This specification develops the technical approach to the user's stated intent.
The diagnostic questions and controls below are design proposals, not additional
user-stated goals. Keep technical elaboration here and execution steps in the plan.

## Required answers and evidence

| Question | Required evidence |
| --- | --- |
| What hardware does GB10 provide? | Sourced SM/CUDA-core counts and Tensor Core capabilities; queried device/cache/resource properties; calculations labeled as derived |
| How does output shape use the SMs? | Actual algorithm/tile/grid information, fixed-K M/N comparisons, residency/scheduling observations where available |
| How does K affect performance? | Fixed-output K comparisons, split-K choices, latency and memory/pipeline evidence |
| Does aspect ratio matter at equal useful work? | Equal M×N and K with different M/N ratios, matched arithmetic and timing conditions |
| Which shapes perform best, and why? | Repeated per-mode rankings, confirmation and diagnostic evidence; remaining causal uncertainty stated explicitly |

## Coverage

The target is this host's DGX Spark / GB10. Start with all natively supported
Tensor Core modes, including integer and scaled variants. As a diagnostic control,
the proposed design adds full FP32 CUDA-core GEMM after the Tensor Core measurements
within the same experiment. Dense, structured-sparse, binary and different-backend
comparisons remain distinct result groups.

The following is a capability checklist, **not a support claim**. Enumerate
legal A/B pairs, accumulator variants, scale formats/block sizes and sparsity
against the installed toolchain and device. A native instruction must be
distinguished from emulation, scalar execution and unsupported library layouts.

| Family to resolve | Variants to inventory | Package capability at design time |
| --- | --- | --- |
| TF32 | FP32 storage, TF32 multiplication, FP32 accumulation | cuBLASLt GEMM implemented |
| FP16 | FP32 and native FP16 accumulation where supported | GEMM missing |
| BF16 | Supported accumulator/output combinations | FP32 accumulation/BF16 output implemented |
| FP8 | E4M3, E5M2, legal ordered A/B pairs and accumulator variants | E4M3×E4M3, FP32 accumulation/BF16 output implemented |
| FP6 | E2M3, E3M2 and legal mixed low-bit pairs | Missing; device/library route requires verification |
| FP4 | Unscaled E2M1 where supported | Unscaled GEMM missing |
| Scaled floating point | MXFP8, MXFP6, MXFP4, NVFP4; legal pairs and distinct scale/block layouts | NVFP4 GEMM implemented; MXFP4 register probe only |
| INT8 / UINT8 | Signed, unsigned and legal mixed pairs, INT32 accumulation | GEMM missing |
| INT4 / UINT4 | Native signed/unsigned and mixed pairs if supported | Missing; capability unresolved |
| Binary operations | Native AND/XOR-popcount forms if supported | Missing; separate operation semantics |
| FP64 Tensor Core arithmetic | Native GB10 support must be established | Unresolved; scalar/emulated FP64 does not qualify |

Each registry entry records hardware evidence, library exposure, backend/layout,
implementation status and probe outcome. Classify unsupported modes with
evidence, and keep implementation gaps or unresolved support separate. A
failed cuBLAS heuristic search does not prove absence of hardware support.
The experiment cannot claim all-mode measurement coverage while supported
rows lack a validated complete GEMM implementation.

## Measurement contract

- Compare shapes within a fixed arithmetic/scale/output/backend/sparsity row.
  Use common shape and timing rules across rows; disclose necessary differences.
- Validate correctness before tuning and after measurement. Use exact integer
  references, numerical floating-point references, and format-specific scale
  and sparse-metadata checks. No silent datatype substitution or fallback.
- Separate setup, encoding, transfers, allocation, tuning, graph construction
  and validation from timed GEMM. Record warmup and repetition boundaries.
- Record useful operations, latency, variability and supported telemetry.
  Floating rates use TFLOPS, integer rates TOPS; sparse dense-equivalent and
  binary conventions must be explicitly labeled.
- Preserve source/toolchain, exact commands, run order, failed attempts and raw
  evidence. Record cache conditions and hardware settings. Profile separately
  from the runs used for performance rankings.

The plan sets the bounded shape matrix, repetition counts, confirmation rule
and diagnostic selection. Capability discovery freezes the eligible row list
and run count before timing; it does not create a second experiment.

## Boundary with the reusable package

The package supplies reusable Rust APIs, CLI commands and kernels. This
experiment owns its research questions, mode registry, shape selection,
ordering, measurement driver, analysis and conclusions.

Version experiment-specific code and requirements in this workspace. If a required
capability is useful in the package, implement it as a separately identifiable
package change and document only its implemented behavior in the package spec.
For capabilities better served by a dedicated harness, use this experiment's
Rust/Cargo driver. Neither route changes the package's general purpose.

Do not represent this specification as an already supported package API or
automatically broaden package defaults because the experiment needs a mode.

## Completion and outputs

Produce one synthesis answering the intent's questions: hardware inventory,
mode coverage, best tested square/rectangular shapes with variability, and an
explanation supported by kernel/profiler evidence. Explicitly list unresolved
support or causal questions rather than claiming they were answered.

Complete the planned phases under the same experiment ID. Preserve repeated
runs and diagnostic captures as evidence within that experiment. Close the
bounded study with its findings and limitations; any suggested expansion is
an unexecuted recommendation until incorporated into its documented scope.

Keep raw measurements, generated analysis, profiler captures and journals in
the external archive identified in [README.md](README.md). Consolidate verified
conclusions into the existing [package results report](../../docs/final-report.md),
with the experiment ID and evidence location. Do not create a competing results
report in the package. Existing benchmark results retain their original identity.
