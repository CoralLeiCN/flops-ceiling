# Experiment plan: GB10 GEMM shapes

Experiment ID: `GB10-GEMM-SHAPES-001`.
Designed on 2026-10-09 (Europe/London). **Status: design only; not executed.**
This is the proposed procedure for one experiment with the purpose in
[intent.md](intent.md) and technical design in [spec.md](spec.md). The following
phases belong to that experiment; batches, confirmation rounds and profiler
captures are not separate experiments. Proposed implementation prerequisites
have not been fulfilled by writing this plan.

## Questions and controlled comparisons

Find the best tested GEMM shapes per arithmetic mode and investigate how their
performance relates to GB10's SM organization, kernel tiling and memory system.

| Question | Comparison | Evidence needed |
| --- | --- | --- |
| What hardware are we scheduling work onto? | Published specifications versus device queries | SM/core inventory, Tensor Core capabilities, cache and resource limits |
| When is there enough parallel work? | Change M or N while holding K fixed | Output tile count, actual grid, residency, partial waves |
| How much does reduction length matter? | Change K with M=N fixed | Kernel duration, pipeline activity, traffic and split-K changes |
| Does aspect ratio matter beyond total work? | Same M×N and K, different M/N ratios | Same useful operation count, different tiling and access patterns |
| Which shape is fastest for each mode? | Common square and rectangular search | Repeated throughput and latency, correctness, clocks |

## 1. Establish the hardware and mode inventory

Before timing, archive the revision, source diff, Cargo.lock, binary hash,
driver, CUDA toolkit/runtime, loaded cuBLASLt and Rust versions. Query this
device again. The existing records report **48 SMs, compute capability 12.1
and 24 MiB L2**; NVIDIA publishes **6,144 CUDA cores**, giving 128 cores/SM.
Keep queried, published and calculated values distinct. Source Tensor Cores
per SM/total specifically for GB10 or mark them unresolved. Do not import B200
or another Blackwell chip's resource counts into the GB10 model.

Also record supported warp/block limits, shared memory and registers per SM,
memory capacity/bandwidth information, and current SM clocks. The CUDA-core
count alone does not supply a Tensor Core throughput formula. Any theoretical
Tensor Core rate needs the operation's format, instruction work/rate, clock
and dense/sparse convention, with sources and assumptions.

Build a machine-readable capability registry from the
[coverage checklist](spec.md#coverage). Each entry identifies A/B encodings,
accumulation/output type, scale format and block size, sparsity, backend/layout,
hardware evidence, library exposure, implementation status, probe command and
validation outcome. The checklist is not a claim of support for every mode.

Enumerate the supported instruction and library combinations against the
**installed** versions, including supported structured-sparse variants. A generic
Blackwell feature list or a PTX instruction accepting a target is insufficient
evidence of native Tensor Core execution. Use algorithm descriptors plus a
small validated device probe and native-instruction/profiler evidence where
needed. A cuBLASLt heuristic returning no candidates establishes failure of
that library configuration, not absence of the datatype in hardware.

Every entry must finish as supported and validated, unsupported with evidence,
or unresolved/implementation missing. Keep supported gaps visible until they
have a complete GEMM path. Do not label a four-format run as all-mode coverage.
Include legal mixed A/B and accumulator variants as separate registry rows;
kernel instruction shapes and tuning choices are configurations within a row.

Use cuBLASLt wherever it exposes the mode. A missing library path may require
a Rust/cuTile or Rust-generated PTX GEMM, reported as a separate backend cohort.
A register-only probe can establish instruction behavior but cannot replace
complete GEMM shape measurements. Do not rank backend differences as isolated
datatype effects.

## 2. Prepare and validate the required implementation

The current CLI implements only `tf32`, `bf16`, `fp8` (E4M3) and `nvfp4` for
Tensor Core GEMM. Prepare the missing registry rows before claiming complete
coverage, using this experiment's Rust/Cargo harness and reusable package
extensions where appropriate. Requirements include independent A/B formats,
accumulator and output choices, scaling descriptors/encoders, integer reference
checks and required layouts. These are experiment requirements, not existing
CLI flags or a redefinition of the package's scope.

Generate floating operands from one deterministic signed-uniform FP32 source
in [-1, 1] for matching shape/seed, then encode each mode. Supply this through
the experiment's input path; do not silently change package defaults. Specify
and freeze
rounding, scale selection and block layout in the registry before execution;
record decoded values and quantization error. Use bounded signed values [-7, 7]
and unsigned values [0, 15] for integer families, with deterministic bit values
for binary modes. Different arithmetic families are separate numerical workloads.

For the main floating-point comparison, use FP32 accumulation and FP32 output
where the hardware/library combination supports them. Provide explicit output
selection for this experiment instead of changing package defaults. Keep native
FP16-accumulation/output cases and
any required alternative outputs in separate, explicitly named groups. Use
INT32 accumulation/output for arithmetic integer modes. Keep output and scale
choices fixed within every shape comparison; changing them starts a new row.

Require Tensor Core implementation evidence for every timed row. Generalize
the existing TF32 capability check to the other modes in the harness or a reusable
package extension, with format-appropriate checks; do not assume the current
CUDA header has a unique flag for every
low-bit format. Preserve rejected candidates and prohibit silent conversion
to another input format or scalar fallback in a Tensor Core result.

Before performance runs, validate 128×128×256 and 128×256×512, checking every
output. Include signed/zero inputs, varied finite scales and format-specific
edge cases. For floating point, compare with CPU reference sums of the decoded
encoded/scaled operands and separately report error from the original source
values where quantization is involved. Retain the TF32 check against original
FP32 inputs. For integer modes, use INT64 reference sums and inputs bounded so
INT32 cannot overflow; require exact agreement. Binary operations need an exact
bitwise/popcount reference. Freeze each new mode's numerical acceptance rules
before throughput measurement; do not relax them to admit a faster kernel.

For sparse rows, validate the sparsity metadata and compare with a reference
using the same pruned matrix. Packing, quantization, scale construction and
sparsification are outside GEMM timing and must be recorded as such.

## 3. Common shape matrix

Run these **22 distinct M×N×K shapes** for every supported, validated Tensor Core
registry row, followed by the full FP32 control on the same shape matrix.
Order below defines the initial case list; remove overlaps between groups.
These shapes are selected to test hardware hypotheses, not predicted winners.

| Group | Shapes | Why these shapes |
| --- | --- | --- |
| Square size sweep (8) | S³ for S = 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288 | Parallelism and footprint across sizes; includes historical controls and a larger bound |
| Fixed-K scheduling sweep (6) | 1024×N×4096 for N = 1280, 1536, 1792; plus M×1024×4096 for M = 1280, 1536, 1792 | Change one output dimension and bracket an illustrative 48-SM wave |
| Fixed-output K sweep (4 additional) | 4096×4096×K for K = 1024, 2048, 8192, 12288; reuse 4096³ | Isolate reduction length at fixed output area |
| Equal-work aspect ratios (2 additional) | 2048×8192×4096 and 8192×2048×4096; reuse 4096³ | Same M×N, K and useful operations with different aspect ratios |
| Small rectangular controls (2) | 1536×1024×1024 and 1024×1536×1024 | Recheck earlier TF32 candidates across all modes and footprints |

For illustration, 128×256 logical output tiles give **40, 48 and 56 tiles**
in each fixed-K sweep. With one active tile per SM, that brackets a wave across
48 SMs. Actual kernels may use different tiles, multiple resident blocks,
split-K, clusters or persistent scheduling. Decode the selected configuration
and map library tile coordinates to logical M/N before applying this model.
If selection changes, the comparison also measures the library's response to
shape; it does not isolate wave occupancy by itself.

Record actual allocated A/B/D bytes, padded scale bytes and workspace alongside
L2 capacity. Equal dimensions have different footprints across formats; fitting
the nominal matrices in L2 does not prove residency. The primary workload reuses
operands without a cache flush. Cold-cache/streaming behavior is outside this
protocol and must not be inferred from these results. If it becomes necessary
to answer the agreed question, revise this experiment's scope and predeclare
an additional control phase under the same experiment ID before running it.

## 4. Timing, ordering and confirmation

- Compile and initialize before timing. Use device 0, seed 42, alpha=1, beta=0,
  graph timing, a 64 MiB workspace cap and up to 32 requested cuBLASLt heuristic
  candidates. Record actual counts. Disclose a different backend's tuning set.
- Tune each case/round: 10 warmup GEMMs, one untimed 20-GEMM graph replay and
  three measured batches of 20. Select only numerically valid candidates.
- Final timing: 100 warmup GEMMs plus one untimed replay of the measurement
  graph, followed by five measured trials. Use 1000 GEMMs/trial when all three
  dimensions are <=3072; use 100 otherwise. Keep this rule identical across
  modes and rounds. Report per-GEMM time as well as throughput.
- Run three rounds sequentially, with no competing GPU benchmark. Rotate the
  22-shape list by 0, 7 and 14 positions. For each shape position j in round r
  (both zero-based), cycle the frozen, documented registry order starting at
  (j+r) modulo the row count. Apply this rotation to the Tensor Core rows, then
  run the full FP32 control phase with the same three shape-order rotations.
  Do not pool the later FP32 control with earlier measurements or treat its
  different thermal period as an isolated datatype effect.
- Record temperature, utilization, power and SM clocks approximately every
  second and at trial boundaries; mark unavailable readings explicitly. Keep
  clock/power/fan settings fixed and record other GPU users. Archive failures;
  do not silently discard slow trials or retry into a winning result.
- Final validation uses 256 CPU reference outputs plus all-output finiteness
  checks for floating point; use corresponding exact checks for integer/bit
  operations. Preserve tuning and post-measurement validation separately.

For P validated Tensor Core registry rows plus one full FP32 control row,
discovery is **66(P+1) results and 330(P+1) trials**. Freeze the row list and
resulting run count after capability inventory and
before measurements. Rows awaiting implementation remain explicitly uncovered.
Historical results motivate controls but are not pooled with this cohort.

Select the two shapes with highest mean throughput per row, plus 4096³
(deduplicated), for three further rounds of five trials with the same settings.
Rotate their order and the mode order within their cohorts. This confirmation
phase of the same experiment adds at most 45(P+1) trials. Report
discovery and confirmation separately, with mean, sample standard deviation
and individual round means. Report the top two as a leading group if their
confirmation means differ by less than 2%, or if the leading shape does not win
all three confirmation rounds. The 2% rule is a practical tie threshold for this
experiment, not a statistical confidence interval. A larger gap also needs
scrutiny against observed variability. Complete the defined phases and report
their findings. Any necessary extension is a documented scope/protocol revision
within this experiment before execution, not an automatic new experiment.

Count dense arithmetic as 2MNK useful operations: floating point in TFLOPS,
integer in TOPS. For 2:4 sparse rows report both executed-nonzero operations
and dense-equivalent rates, separately from dense GEMM. Report binary operations
under their explicitly defined bit-operation convention, not floating TFLOPS.
Sparse and binary rows are companion comparisons, not pooled into a dense
floating-point ranking.

## 5. Explain the result with selected-kernel evidence

For each row, decode algorithm ID/configuration, tile geometry, split-K,
workspace and numerical implementation metadata. Profile, in separate runs,
4096³, the best square, the best rectangle and the largest adjacent throughput
change in the fixed-K sweep (both sides); deduplicate repeated shapes.
Collect actual grid/block geometry, registers/shared memory, achieved occupancy,
Tensor Core activity, L2/DRAM traffic and kernel durations where supported.
Unavailable counters stay unresolved. Include workspace/reduction kernels.

The current CLI retunes at each invocation. A saved-algorithm replay path is
therefore a prerequisite for profiling the exact measured cuBLASLt choice.
Validate the saved configuration with the matching library/shape before replay.
For causal shape comparisons, also hold an algorithm/tile fixed across adjacent
shapes wherever it is valid. Otherwise label the shape and kernel changes
together. Profiler timings are diagnostic and do not enter the unprofiled
throughput ranking.

The explanatory output should relate observed work distribution to the 48 SMs,
and actual cache/traffic behavior to the encoded footprints. Include the full
FP32 CUDA-core control in the explanation; matching a dimension to 6,144 cannot
substitute for these measurements.

## 6. Synthesize one experiment's answers

Answer every question in [intent.md](intent.md), marking unresolved evidence
explicitly. Summarize the sourced hardware inventory, coverage matrix, best
tested shapes, variation and explanations under `GB10-GEMM-SHAPES-001` in the
[package's existing results report](../../docs/final-report.md).
Version the maintained registry, protocol and driver/analysis source in this
experiment workspace. Archive each run's exact commands, probe failures,
registry/source/toolchain snapshot, raw trials, telemetry and profiler captures
in the external evidence directory identified in [README.md](README.md), following
[AGENTS.md](AGENTS.md). This plan is not a results report or evidence of a run.

Primary references for inventory and interpretation:

- [DGX Spark hardware specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): published GB10/Spark specifications.
- [CUDA 13.0 cuBLAS reference](https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html): datatype/layout/scaling combinations and algorithm capabilities; verify against the loaded library.
- [PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html): matrix instruction forms and target restrictions; instruction listing is not a measured GEMM result.
- [NVIDIA GEMM performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): tiling, reduction length and scheduling-wave models; its examples are not GB10 measurements.
