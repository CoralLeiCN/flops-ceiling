# DGX Spark performance: final report

Consolidated on 2026-09-29 from measurements on 2026-09-27–28.
Build and correctness verification updated on 2026-10-03; a cuBLASLt shape
comparison was added on 2026-10-06, with theoretical and measured FP32
candidate-shape follow-ups on 2026-10-07 and an explicit FP32/TF32 comparison
on 2026-10-08 UTC. September results retain their original measurement cohorts
and are reported separately from the new comparisons.
This is the repository's single maintained results report. Detailed journals,
imported reports, research notes and raw artifacts are stored outside Git.

The Rust benchmarks produce validated BF16, FP8, NVFP4 and full FP32 results,
with an explicit cuBLASLt TF32 mode added for the Oct 8 UTC comparison.
Observed NVFP4 rates include approximately **493 TFLOPS for repeated register
instructions** in September and **344 TFLOPS for an 8192³ cuBLASLt GEMM** in
the October shape comparison. These are different workloads and do not
establish sustained hardware ceilings or inference throughput.

## Implementations and measurement scope

There are three implementations behind two APIs:

| Implementation | API | Supported arithmetic |
| --- | --- | --- |
| cuBLASLt | GEMM | BF16, FP8 E4M3, NVFP4, full FP32, TF32 |
| Native Rust cuTile, optional feature | GEMM | BF16, FP8 E4M3, NVFP4, full FP32 |
| Rust-generated PTX | Registers | BF16, FP8 E4M3, NVFP4, MXFP4, full FP32 |

Tensor register modes support dense and structured 2:4 sparse instructions;
FP32 registers use dense scalar FMA. GEMM accumulates in FP32, with BF16 output
except for full FP32 and TF32 modes, which use FP32 storage/output. cuBLASLt
FP32 uses PEDANTIC compute without TF32 substitution. Register kernels reuse uniform operands;
complete GEMM also moves matrices and scales through the memory hierarchy.

The FP32 shape comparisons on Oct 6–7 all used the **same cuBLASLt backend and
`CUBLAS_COMPUTE_32F_PEDANTIC` arithmetic mode**, including 3072³, 4096³ and
the rectangular candidates. Their throughput differences cannot be attributed
to switching between FP32 and TF32. TF32 is a separate arithmetic mode that
cuBLASLt supports through Tensor Core execution. The Oct 8 UTC implementation
adds `--precision tf32` / `Precision::Tf32` with explicit verification of
TF32 Tensor Core algorithm flags; the earlier cohorts remain full FP32.
FP32 output alone does not identify the multiplication precision. See the
[FP32/TF32 and backend definitions](benchmarking.md#fp32-tf32-and-backend-selection)
for the distinction and recorded configuration fields. The specific bottleneck
behind the 3072³ deficit remains unisolated.

## September TFLOPS (no profiler attached)

Values are means ± sample standard deviation. No Nsight Systems or Nsight
Compute was attached to these measurements. The cohorts have different trial
counts and configurations; the table is not a controlled cross-precision comparison.

| Precision | Dense register TFLOPS (no profiler attached) | cuBLASLt 4096³ TFLOPS (no profiler attached) | Measurement cohort |
| --- | ---: | ---: | --- |
| BF16 | 122.44 ± 0.62 | 94.93 ± 1.53 | Sep 27, 3 rounds × 5 trials |
| FP8 E4M3 | 244.92 ± 1.31 | 190.66 ± 3.80 | Sep 27, 3 rounds × 5 trials |
| NVFP4 | 493.47 ± 0.88 | 334.17 ± 1.44 | Sep 28, 1 session × 3 trials |
| FP32 | 29.691 ± 0.002 | 15.79 ± 0.05 | Sep 28, 1 session × 3 trials |

All register cases use 384 blocks, 128 threads/block and 16 independent chains.
BF16/FP8 use 700 loop iterations, 100 warmups and 100 launches/trial. Their GEMM
runs tune up to eight candidates and use 10 warmups plus an untimed graph replay,
then 20 GEMMs/trial. Inputs use seed 42, a 64 MiB workspace limit and 256 reference
validation samples. Candidate tuning is excluded from measurement.

The Sep 28 cases use 10 warmups and 300 launches/GEMMs per trial. NVFP4 registers
use 700 loop iterations; FP32 uses 20,000. GEMM selects the first validated
heuristic and uses graph timing with an extra untimed replay. The earlier FP32
700-loop cohort measured 27.86 ± 0.33 register TFLOPS and 16.11 ± 0.12 GEMM TFLOPS
over 15 trials; the different configurations do not establish an improvement.

Longer NVFP4 register batches on Sep 27 measured **490.49 ± 2.36 dense** and
**981.91 ± 4.80 sparse dense-equivalent TFLOPS** over three alternating rounds,
10 trials/round and 1,000 launches/trial, with 100 warmups. BF16/FP8 sparse
register cohorts measured **245.30 ± 1.04** and **490.57 ± 2.10** respectively
over 15 trials. Sparse counts include implicit zeros; executed nonzero arithmetic
is half the reported dense-equivalent count.

For cuBLASLt NVFP4, an earlier longer 4096³ comparison pooled **322.72 ± 3.20**
TFLOPS for installed cuBLASLt 13.1.1.3 and **323.97 ± 3.66** for isolated
cuBLASLt 13.8.0.4. Each library had two five-trial runs, with 10,000 GEMMs/trial.
It did not establish a sustained library-version win. The short-session 334.17
result above should remain separate from these longer measurements.

Native Rust cuTile measured **250.86 ± 0.66 NVFP4 TFLOPS at 4096³** in five
100-GEMM trials, and **1.69 ± 0.01 FP32 TFLOPS at 1024³** in three tuned rounds
of three 20-GEMM trials. Smaller FP32 tiles removed observed spills, but cuTile
remains an initial implementation. BF16/FP8 cuTile correctness and diagnostic
captures succeeded; there is no complete matched performance matrix for all
three implementations.

## cuBLASLt shape comparison, 2026-10-06

The three tested square sizes have different winners by precision: **6144³
for BF16, 8192³ for FP8 and NVFP4, and 4096³ for full FP32**. Every precision
retained its shape ranking in all three rounds. Values below are means ± sample
standard deviation across 15 trials, including variation between rounds.
Percentage changes use this comparison's own 4096³ baseline.

| Precision | 4096³ TFLOPS | 6144³ TFLOPS | 8192³ TFLOPS | Best tested shape; change from 4096³ |
| --- | ---: | ---: | ---: | --- |
| BF16 | 94.04 ± 1.05 | **98.24 ± 1.08** | 95.45 ± 0.41 | 6144³; +4.48% |
| FP8 E4M3 | 178.38 ± 0.90 | 185.79 ± 0.49 | **190.12 ± 0.95** | 8192³; +6.58% |
| NVFP4 | 325.26 ± 2.51 | 334.62 ± 1.50 | **344.09 ± 0.51** | 8192³; +5.79% |
| Full FP32 | **15.56 ± 0.11** | 13.48 ± 0.04 | 14.95 ± 0.19 | 4096³; no larger-shape improvement |

The winning larger shapes improved over 4096³ by 4.11–5.04% for BF16,
5.69–7.18% for FP8 and 5.06–6.33% for NVFP4 across the three individual
rounds. FP32 instead fell 13.35% at 6144³ and 3.91% at 8192³ in the pooled
comparison. These are throughput comparisons between different workloads,
not latency improvements for an unchanged GEMM or a proof of optimal size.
They do not establish an improvement or regression against the September cohort.

The predeclared protocol used three rounds per precision and five trials per
shape per round. Shape order rotated as 4096/6144/8192, 6144/8192/4096 and
8192/4096/6144; precision groups ran sequentially as BF16, FP8, NVFP4 and FP32.
All cases used seed 42, a 64 MiB workspace limit, graph timing, 100 final
warmup GEMMs and one additional untimed 300-GEMM graph replay, followed by
five measured batches of 300 GEMMs. Each shape/round was independently tuned
with up to 32 requested heuristic candidates, 10 candidate warmups, an untimed
tuning graph replay and three 20-GEMM tuning batches. Actual returned candidate
counts were three for BF16; six for FP8 at 4096³ and four at its larger sizes;
six for NVFP4; and eight for FP32. All returned lists were shorter than the
requested limit.

CUDA-event time covers repeated GEMMs, including any library workspace
operations, with useful work counted as `2*M*N*K`. Benchmark compilation,
initialization, encoding, transfers, allocation, tuning, warmup, graph setup,
validation and telemetry queries are outside event timing. Operands and
allocations are reused without an explicit cache flush. Arithmetic/output
formats are unchanged: FP32 accumulation with BF16 output for the tensor
formats, and PEDANTIC full FP32 arithmetic/output for FP32. No profiler was
attached. All **36 shape/round results, 180 trials and 195 candidate validation
checks** succeeded; each validation scans all outputs for finiteness and checks
256 outputs against CPU FP64 reference sums. Large-matrix validation is sampled.

Host-side queries of the saved algorithms, using the same cuBLASLt version,
confirmed BF16 output tile IDs of 128×208 and NVFP4 tile IDs of 128×128, both
with split-K=1. FP8 selected 128×128 at 4096³ and either 192×176 or 128×256 at
the larger sizes. FP32 selected 128×256, with split-K=1 at 4096³ and split-K=2
at both larger sizes. These configuration changes do not isolate the cause of
the performance differences. In particular, NVFP4 used 128×128 tiles yet
8192³ outperformed 6144³: the illustrative full-wave argument is insufficient
to identify the best size. Actual residency and scheduling waves were not
profiled, and matching a dimension to the CUDA core count is not an optimization.

Execution ran from **00:30:20 to 00:51:29 UTC on 2026-10-06**, at revision
`6e58a5c7997cbfe56df59da7768fd3682ba910f6`, with benchmark source and dependencies
unchanged. The device/library identity was GB10, SM12.1, 48 SMs, 24 MiB L2,
driver 580.173.02, CUDA runtime 13.0.96, cuBLASLt 13.1.1.3 and Rust/Cargo 1.98.1.
The release build completed before GPU measurements. Per-trial snapshots and
1,266 approximately one-second device-wide samples recorded 33–83 °C,
4.71–96.72 W and 208–2535 MHz SM clocks across setup and execution. These ranges
include idle periods; they are not kernel-only average power. Memory clocks
and power limits were unavailable. The continuous trace uses BST (UTC+01:00);
command boundaries are UTC and benchmark telemetry uses Unix milliseconds.

Clocks were automatic, desktop graphics processes were present, and host-side
analysis helpers ran during the sweep without launching GPU kernels. No power,
clock or fan settings were changed. This single session does not establish
thermal equilibrium or variability across independent sessions. The raw archive
contains exact commands, manifests, all trials, source/lockfile snapshots,
binary hash, telemetry, independent Rust audits and decoded algorithm attributes.
The metadata helper's initial C ABI build/type errors and their verified fixes
are preserved in the external mixed-precision journal; no benchmark fix or GPU
rerun was required. Evidence locations are listed below.

## FP32 theoretical follow-up, 2026-10-07

A CPU-only analysis derived candidate shapes; that analysis itself established
no new GPU measurement or performance winner. Its subsequent GPU comparison is
recorded below. Assuming the observed 128×256 output tile,
one tile per block, split-K=1 and `r` resident blocks/SM, the nominal tile count
is `B=ceil(M/128)*ceil(N/256)` and wave efficiency is
`B/(48*r*ceil(B/(48*r)))`. Actual residency and block scheduling remain unknown.

**4096×4608×4096** is a useful first rectangular candidate: its 576 tiles give
full modeled waves for `r=1,2,3,4`, while retaining the baseline M and K.
**4096×3840×4096** gives 480 tiles and full waves for `r=1,2`. For a square-only
search at `r=1`, aligned dimensions `S=3072,6144,9216,...` have full waves;
**3072³** was proposed as an unmeasured candidate. These conclusions depend on using the
assumed kernel configuration, which shape selection alone cannot force through
the current heuristic-based CLI. **4096³ remains the measured FP32 winner among
the previously tested square sizes.**

At `r=1`, the 4096³ baseline already has 96.97% modeled wave efficiency, so
eliminating its tail would improve throughput only 3.125% if all other
efficiencies stayed fixed. Full waves cannot explain or remove all kernel
inefficiency. The full derivation also distinguishes ideal whole-GEMM reuse
from reuse within a tile, accounts for the 24 MiB L2 limit, and separates
clock-dependent FMA capacity from measured throughput. Cache behavior, K-depth,
split-K reduction, kernel selection and clocks prevent a unique optimum from
being inferred from CUDA core or SM counts. Calculations and assumptions are
archived under the evidence locations below; the detailed note is FP32 RUN-0004.

## FP32 candidate-shape measurements, 2026-10-07

**Both rectangular candidates improved throughput over a fresh 4096³ baseline
in all four rounds.** The highest pooled mean was 4096×4608×4096 at +2.13%,
but its 0.11% advantage over 4096×3840×4096 does not establish a clear winner
between the rectangles: their order reversed in the last two rounds. The
square candidate 3072³ was slower in every round. Values are means ± sample
standard deviation across 20 trials per shape, including between-round variation.

| Shape M×N×K | Full FP32 TFLOPS | Change from this session's 4096³ | Selected output tile; split-K |
| --- | ---: | ---: | --- |
| 4096×4096×4096 | 15.71 ± 0.31 | baseline | 128×256; 1 |
| 4096×4608×4096 | **16.05 ± 0.26** | **+2.13%** | 128×256; 1 |
| 4096×3840×4096 | 16.03 ± 0.21 | +2.02% | 128×256; 1 |
| 3072×3072×3072 | 13.62 ± 0.07 | −13.30% | 256×128; 1 |

Per-round gains were 1.31–3.05% for 4096×4608×4096 and 0.84–2.78% for
4096×3840×4096. Both rectangles retained the baseline's selected algorithm ID
20, 128×256 tile, stage ID 26, split-K=1, custom option/swizzle 0 and zero
workspace use. This is consistent with the conditional wave argument, but
does not isolate scheduling from cache reuse or clocks. The 3072³ kernel
instead selected a 256×128 tile, still with 288 nominal output tiles and
complete waves under the one-block/SM assumption. Its lower throughput again
shows that complete waves alone do not identify an optimum. Actual residency
and cache traffic were not profiled.

A read-only follow-up of the saved trials found that 3072³ took **4.256 ms**
per GEMM versus **8.750 ms** for 4096³: it finishes sooner, but performs only
42.19% as much arithmetic in 48.65% as much time. Its lower rate is therefore
a throughput-efficiency loss, not a longer latency. At 3072³, the tuner also
tried unsplit 128×256 in every round; its median tuning times were
4.267/4.286/4.275/4.259 ms versus 4.238/4.253/4.273/4.253 ms for selected
256×128. Changing tile orientation alone is not supported as a remedy.
Mean post-trial SM-clock snapshots were higher for 3072³ (2337 MHz) than
4096³ (2243 MHz); these are snapshots, not in-kernel clock averages, and do
not support a simple lower-clock explanation. Shorter K (3072 versus 4096)
can amortize per-tile setup/finalization less effectively, and changed matrix
dimensions/leading dimensions can change cache reuse; neither contribution
has been isolated. These CPU-only findings are preserved in FP32 RUN-0011.

The predeclared protocol used four rounds of five trials per shape, rotating
the four shapes ABCD/BCDA/CDAB/DABC in the table's order. Each shape/round
independently tuned the eight returned candidates (32 requested). All other
benchmark settings matched the Oct 6 FP32 protocol: PEDANTIC full FP32 output
and arithmetic, seed 42, 64 MiB workspace limit, graph timing, 10 candidate
warmups, an untimed tuning graph replay and three 20-GEMM tuning trials,
then 100 final warmups, an untimed 300-GEMM graph replay and five measured
300-GEMM batches. CUDA-event throughput counts `2*M*N*K`; setup, tuning,
warmup, validation and telemetry remain outside timing. Operands/allocations
were reused without a cache flush. All **16 shape/round results, 80 trials and
128 candidate validations** passed, including all-output finiteness scans and
256 CPU FP64 reference samples per validation. No benchmark source or
dependency changes were required; no tests were added.

Execution ran **22:13:15–22:19:08 UTC on 2026-10-07** at revision
`6e58a5c7997cbfe56df59da7768fd3682ba910f6`, on the same GB10, driver 580.173.02,
CUDA runtime 13.0.96, cuBLASLt 13.1.1.3 and Rust/Cargo 1.98.1 environment.
Compilation finished before GPU measurements. Per-trial snapshots and 353
approximately one-second UTC telemetry samples recorded 35–83 °C, 0–96%
utilization, 4.79–97.51 W and 208–2489 MHz SM clocks across setup/execution;
memory clocks and power limits were unavailable. These include idle/setup
periods, not kernel-only average power. Automatic clocks and desktop graphics
remained active; no power, clock or fan settings changed. The baseline's round
means varied from 15.46 to 16.22 TFLOPS, so this session neither establishes
thermal equilibrium nor ranks the rectangles across independent sessions.
Different workloads are being compared for throughput, not identical-workload
latency. The prior 4096³ result is not pooled into this cohort.

The sandbox's initial `nvidia-smi` discovery failure was resolved by host GPU
access and recorded in FP32 RUN-0005. All four benchmark commands and the Rust
result/algorithm audits succeeded without retry. Exact commands, build logs,
source/lockfiles, environment, raw results and telemetry are in the external
archive below; FP32 RUN-0006–RUN-0009 record the rounds and RUN-0010 the analysis.

## Full FP32 versus TF32 shape measurements, 2026-10-08 UTC

`--precision tf32` now requests cuBLASLt `CUBLAS_COMPUTE_32F_FAST_TF32` with
FP32 input/output storage and FP32 accumulation. Full FP32 remains PEDANTIC.
Both modes generate exactly the same original FP32 inputs for a matching
shape/seed, without host rounding. TF32 candidates must advertise Tensor Core
arithmetic, TF32 inputs and FP32 accumulation; other candidates are recorded
and excluded. The selected flags were `0x40202` for TF32 and `0x80201` for
full FP32: respectively HMMA/TF32-input/FP32-accumulator and
FMA/FP32-input/FP32-accumulator. These are library algorithm descriptors;
no profiler was attached to these measurements. See the
[implemented mode and validation details](benchmarking.md#comparing-full-fp32-and-tf32).

Across the bounded search of **17 distinct TF32 shapes**, the best tested
repeated-GEMM shapes were **1536×1024×1024 and 1024×1536×1024**, both near
48 TFLOPS and about 3× their matched full-FP32 rates. Their difference is
smaller than the observed variation; neither is a demonstrated unique optimum.
The smaller-shape follow-up below has separate timing settings and is not pooled
with the larger-shape results.

The larger-shape comparison used **three rounds with five trials per
shape/mode/round**, each trial timing 100 GEMMs. The table reports the mean
and sample standard deviation over all 15 trials. Mode order alternated within
pairs and shape order rotated between rounds. Rates use useful `2*M*N*K`
FLOPs per GEMM and CUDA-event elapsed time. Each pair has matched dimensions,
input values, seed 42, graph timing and warmup; different shapes compare
throughput on different workloads. The 12288³ extension ran TF32 only.

| Shape, M×N×K | Full FP32 TFLOPS | TF32 TFLOPS | TF32 / FP32 |
| --- | ---: | ---: | ---: |
| 3072³ | 13.66 ± 0.30 | 42.16 ± 0.92 | 3.09× |
| 4096³ | 15.69 ± 0.39 | 40.52 ± 0.34 | 2.58× |
| 4096×3840×4096 | **16.03 ± 0.34** | 41.32 ± 0.48 | 2.58× |
| 4096×4608×4096 | 16.02 ± 0.24 | 42.22 ± 0.78 | 2.64× |
| 6144³ | 13.46 ± 0.06 | 42.85 ± 1.22 | 3.18× |
| 8192³ | 14.97 ± 0.09 | 41.58 ± 0.67 | 2.78× |
| 12288³ | Not measured | **43.33 ± 0.14** | — |

12288³ had the highest TF32 average in this cohort, but did not win every
round: 3072³ led round 1 and 6144³ led round 3. The predeclared condition for
extending to 16384³ was a 12288³ win in all three rounds, so that extension
was not run. The two FP32 rectangles are effectively tied at this variability.
TF32 gains of 2.58–3.18× are matched-shape ratios of means, not an assertion
that the arithmetic has equivalent accuracy.

All cases requested 32 heuristic candidates with a 64 MiB workspace cap and
tuned the returned candidates. Candidate timing followed 10 warmup GEMMs
and an untimed graph replay, then three batches of 20 GEMMs. Final timing
followed 100 warmup GEMMs and one untimed replay of the measurement graph.
Compilation, initialization, allocation, transfers, tuning, graph construction,
validation and telemetry were outside the CUDA-event timing window. Operands
and allocations were reused with no L2 flush. Clocks were automatic and
desktop graphics remained active; no clock, power or fan setting changed.

The selected TF32 kernels used split-K 1. Their tile descriptors varied between
128×128, 128×256 and 256×128. Full FP32 used split-K 1 on the smaller members
of this cohort and split-K 2 at 6144³ and 8192³. This directly establishes
different library algorithm choices; it does not establish the cause of each
timing difference. Matching a dimension to the CUDA-core count still does not
identify the best shape.

The larger cohort ran **23:02:31–23:27:00 UTC**, at revision
`6e58a5c7997cbfe56df59da7768fd3682ba910f6` plus the archived implementation
patch/source snapshot. The device was GB10, SM12.1, 48 SMs, 24 MiB L2,
driver 580.173.02, CUDA runtime 13.0.96, cuBLASLt 13.1.1.3, Rust/Cargo 1.98.1.
Its 1,466 approximately one-second telemetry samples spanned 35–85 °C,
0–96% utilization, 4.95–96.79 W and 208–2502 MHz SM clocks, including idle
and setup periods. Memory clocks and power limits were unavailable.

### Smaller-shape follow-up

Because the large-shape results plateaued near 40–43 TFLOPS, a separately
declared follow-up compared eleven smaller shapes in both modes. It retained
three rounds and five trials per round, but increased each measured graph to
**1000 GEMMs** to lengthen the small cases' timing windows. All other numerical,
tuning and warmup settings above were retained; shape order rotated by four
positions per round and mode order alternated. Each entry again reports the
mean ± sample standard deviation of 15 trials.

| Shape, M×N×K | Full FP32 TFLOPS | TF32 TFLOPS | TF32 / FP32 |
| --- | ---: | ---: | ---: |
| 768³ | 12.17 ± 0.23 | 35.88 ± 0.36 | 2.95× |
| 1024³ | 13.97 ± 0.17 | 35.22 ± 0.38 | 2.52× |
| 1152³ | 12.62 ± 0.23 | 42.32 ± 0.78 | 3.35× |
| 1280³ | 11.73 ± 0.15 | 33.97 ± 0.36 | 2.90× |
| 1536³ | 13.19 ± 0.12 | 43.01 ± 0.60 | 3.26× |
| 2048³ | 14.33 ± 0.20 | 34.83 ± 0.55 | 2.43× |
| 2304³ | 13.10 ± 0.05 | 36.67 ± 0.35 | 2.80× |
| 3072³ | 13.56 ± 0.04 | 42.77 ± 0.58 | 3.16× |
| 1024×1536×1024 | 16.03 ± 0.06 | **47.86 ± 0.81** | 2.98× |
| 1536×1024×1024 | **16.05 ± 0.04** | **47.95 ± 0.65** | 2.99× |
| 1536×1536×1024 | 14.59 ± 0.05 | 43.40 ± 0.55 | 2.97× |

1536×1024×1024 had the highest TF32 mean, winning rounds 1 and 2; the swapped
M/N shape won round 3. The 0.10 TFLOPS difference between their overall means
does not support preferring one reliably. Both selected a 128×256 tile and
split-K 1 in all rounds. Each has 48 output tiles at that tile size and a
16 MiB combined A+B+D footprint, below the device's 24 MiB L2 capacity.
That is consistent with the cache/tile-wave motivation, but cache residency,
occupancy and the exact performance bottleneck were not profiled. These are
**warm repeated-GEMM rates**, not measurements of streaming matrices from DRAM.
No global optimum or stable ranking across independent sessions is established.

This follow-up ran **23:30:22–23:36:00 UTC** with the same benchmark binary.
Its 338 device-wide telemetry samples recorded 44–83 °C, 0–96% utilization,
5.11–99.72 W and 208–2535 MHz SM clocks, including setup/idle periods;
memory clocks and power limits remained unavailable. The two cohorts cover
17 distinct TF32 shapes and 16 full-FP32 shapes, with **105 performance results
and 525 measured trials**. All 789 returned performance candidates and all
105 selected post-measurement results passed numerical validation; no TF32
fallback candidates were returned. Timing/FLOP arithmetic, matching settings,
three rounds per shape/mode, compute modes, flags and the common binary hash
passed the saved Rust audit.

TF32 post-measurement relative RMSE ranged from **2.37e-4 to 2.89e-4**
(about 0.024–0.029%) against original FP32 inputs, versus **3.17e-7 to 1.17e-6**
for full FP32. These are 256 sampled reference outputs per performance case,
plus a finiteness check of every output; they are not universal error bounds.
Eight small FP32/TF32 correctness cases also checked every output in graph and
stream timing. Six additional BF16/FP8/NVFP4 results passed complete-output
checks. Two BF16 algorithm-23 candidates failed the existing error threshold
and were excluded before selection; the selected alternatives passed. Their
diagnostics are retained, and no tolerance was relaxed. The expected cuTile
TF32 rejection was also checked.

The release build, optional cuTile all-target check and formatting checks passed.
All GPU comparison commands completed successfully. The large driver then
exited 101 in its host audit because it treated `round-*.log` files as result
directories. An entry-type check fixed the helper; the recovered audit passed
without rerunning GPU measurements. TF32 RUN-0006/RUN-0007 preserve the failed
attempt and resolution. Exact commands, dependencies, source patches, timing
boundaries, validations and telemetry are retained in the archives below;
the performance tables do not include the short correctness-check timings.

## Nsight measurements

Both profilers worked in temporary containers on Sep 28. Systems required
`--cap-add=PERFMON` for GPU metrics; Compute required `--cap-add=SYS_ADMIN`.
Host policy stayed at `RmProfilingAdminOnly=1`. Containers used `--gpus all`,
networking disabled, read-only CUDA/Nsight mounts and writable artifact output.
All temporary containers stopped after the runs. Profiling remains optional.

Each Compute capture contains three measured kernels, with nine replay passes
per NVFP4 kernel and ten per FP32 kernel. These percentages describe the separate
Compute executions, not the executions behind the TFLOPS table above.

| Workload | Tensor pipeline activity | FMA pipeline activity | Achieved occupancy |
| --- | ---: | ---: | ---: |
| NVFP4 registers | 98.627 ± 0.005% | Not collected | 37.64% |
| NVFP4 cuBLASLt 4096³ | 57.134 ± 0.736% | Not collected | 22.35% |
| FP32 registers | 0% | 95.090 ± 0.004% | 50.22% |
| FP32 cuBLASLt 4096³ | 0% | 56.683 ± 0.161% | 16.17% |

Activity metrics are `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
and `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed`. Occupancy uses
`sm__warps_active.avg.pct_of_peak_sustained_active`. High activity is not directly
a percentage of advertised TFLOPS or proof of the remaining performance headroom.
BF16/FP8 hardware pipeline counters have not yet been collected in this project.

Systems sampled device-wide GPU metrics at 1,000 Hz and traced 900 arithmetic
kernels per workload. Its sample means include gaps between trials:

| Workload | GPU window | Sampled Tensor Active | Sampled SMs Active |
| --- | ---: | ---: | ---: |
| NVFP4 registers | 0.515 s | 98.89% | 99.84% |
| NVFP4 cuBLASLt | 0.433 s | 62.87% | 86.52% |
| FP32 registers | 0.964 s | 0.00% | 99.02% |
| FP32 cuBLASLt | 8.209 s | 0.00% | 98.81% |

All four Systems reports contain **"Not all CUDA events might have been
collected."** The expected kernel counts are present, but complete event coverage
remains unresolved. Unified Memory tracing is unsupported in this configuration.
cuBLASLt metric samples have occasional intervals up to about 2.57 ms, so sample
means are not exact time-weighted utilization. GPU metrics are device-wide;
desktop graphics processes were present. The initial analyzer did not inspect
these diagnostics; the later audit corrected that omission.

FP32 cuBLASLt also has 900 workspace-sized 2 KiB clears, one per GEMM. Accounting
for them resolved an initial offline analysis assertion; no GPU rerun was needed.
Other captures have no memcpy/memset activity in the measured window.

An earlier four-round paired comparison tested Systems CUDA tracing without
GPU metrics sampling: NVFP4 dense registers measured 487.397 ± 3.726 TFLOPS
with no profiler attached and 487.384 ± 3.734 with Systems attached. For 4096³
cuBLASLt the values were 327.757 ± 3.936 and 326.620 ± 2.613. The largest GEMMs
had mixed timing-change signs across pairs. These observations do not establish
the overhead of Compute replay or Systems GPU metrics sampling.

## FLOP counting and correctness

GEMM counts `2*M*N*K` useful FLOPs. Register tensor work counts
`warps * iterations * accumulators * 2*16*8*K`; scalar FP32 counts
`threads * iterations * accumulators * 2`. TFLOPS divides the count by measured
device milliseconds times `1e9`. Sparse register rates use dense-equivalent K.

Hardware counters independently confirmed:

- NVFP4 registers: **281,857,228,800 operations/launch**, exactly the formula.
- NVFP4 4096³ GEMM: **137,438,953,472 operations**, exactly `2*4096³`.
- FP32 registers: **15,728,640,000 thread FFMA instructions**, yielding
  **31,457,280,000 benchmark FLOPs**. Another 786,432 FADDs implement the checksum
  and are excluded from the arithmetic-loop FLOP count.
- FP32 cuBLASLt: **137,556,393,984 executed FP32 operations**, slightly above
  useful GEMM work because of additional arithmetic instructions.

All three selected kernels per workload agree on counts. All target correctness
checks passed. GEMM checks every output for finiteness and samples numerical
agreement with CPU FP64 reference sums; this is not exhaustive large-matrix
validation. Registers check every thread's checksum for uniform inputs, including
signed/scaled probes. Changing B from 1 to 2 preserved the FLOP count and produced
the predicted checksums 2,869,216 and 5,736,416. Some widely separated scale and
initialization magnitudes fail the strict reference due to MMA rounding; the API
rejects those cases instead of accepting a throughput result.

## Power, temperature and reliability

Each result retains timestamped temperature, utilization, power and SM-clock
snapshots. GEMM samples before setup and after each trial; registers sample
before warmup and after measurement/validation. Memory-clock readings were
unavailable, not zero. Sep 28 measurements with no profiler attached recorded:

| Workload | Temperature snapshot range | Power snapshot range |
| --- | ---: | ---: |
| NVFP4 registers | 36–44 °C | 5.00–11.58 W |
| NVFP4 cuBLASLt | 37–51 °C | 5.26–39.50 W |
| FP32 registers | 40–51 °C | 13.37–49.54 W |
| FP32 cuBLASLt | 41–63 °C | 16.26–95.94 W |

These snapshots can miss short workloads and do not establish average load
power, peak temperature or TFLOPS/W. Clocks were automatic; three trials in one
session do not measure independent-run variability or thermal equilibrium.
Sep 28 NVFP4 measurements total only about 0.4–0.5 seconds of timed work per
workload. Small sample SD is not an accuracy bound.

NVFP4 cuBLASLt averaged 0.411294 ms/GEMM with no profiler attached, versus
0.498795 ms/kernel under Compute. Launch mode, timing scope, replay, caches and
clocks differ, so the roughly 21.3% difference is not a controlled estimate of
profiler overhead. Compilation, allocation, input preparation, warmup, tuning
and validation are outside event timing. Warm operands are reused without
explicit cache flushing. Register and GEMM timings include different work.

## Build and correctness follow-up, 2026-10-03

Fetching the locked Cargo dependencies resolved the earlier offline cuTile
check failure (`aho-corasick v1.1.5` was absent from the cache). With the
documented GCC include path, the all-target cuTile Cargo check and release
binary build passed. No benchmark source or dependency versions were changed.

GPU validation on GB10 passed all 16 GEMM cases: BF16, FP8, NVFP4 and FP32,
each using cuBLASLt and cuTile in both stream and graph timing. Every case used
M=N=128, K=256 and checked all 16,384 outputs against the CPU reference.
All 32 candidate validations passed. Each case used two tuning candidates,
two warmups and three three-GEMM tuning trials per candidate, followed by three
warmups and three five-GEMM measurement trials. Graph mode additionally replayed
its graph once before measurement.

All nine register modes also passed: dense and sparse BF16, FP8, NVFP4 and
MXFP4, plus dense FP32. Each used default unit operands/scales and initialization,
384 blocks, 128 threads/block, 16 chains, 700 iterations, three warmup launches
and three trials of three launches. All four checksum checks per mode passed
for every one of the 49,152 threads; no local or shared memory was reported.
This was one short correctness sweep. Its rates are preliminary and do not
extend the performance comparisons above. Temperature, utilization, power and
SM clocks were recorded; memory clocks remained unavailable. GPU access required
execution outside the sandbox. Git metadata remained empty outside the sandbox
as well, so change-specific review and branch membership remain unverified.

## Environment, reproduction and evidence

The latest cohorts used DGX Spark / NVIDIA GB10, SM12.1, 48 SMs, 24 MiB L2,
driver 580.173.02, CUDA runtime 13.0.96, cuBLASLt 13.1.1.3, Rust/Cargo 1.98.1,
Nsight Systems 2025.3.2 and Compute 2025.3.1. Register PTX 9.0/sm_121a is loaded
through CUDA Driver JIT. cuTile is pinned to
`cc720f182f38bf46527753caa340e78d6d5fa3cc`, using isolated tileiras/ptxas 13.4.92.
Earlier runs retain their specific toolchains in the archived manifests.
The September and Oct 3 runs had no usable Git revision; source snapshots,
Cargo.lock and binary SHA-256 hashes identify those measured builds. The Oct 6
shape comparison records its Git revision above as well as the source and binary
evidence.

See [GEMM usage](benchmarking.md), [register usage](register-ceiling.md) and
[profiler commands](profiling.md). Exact historical commands, UTC timestamps,
exit states, inputs, launch settings, raw trials and diagnostics remain in:

| Evidence | External archive |
| --- | --- |
| Oct 8 UTC explicit TF32 implementation, small correctness checks, large-shape comparison, recovered host audit and raw telemetry | `/home/coral/inference-artifacts/flops-ceiling/2026-10-08T23-01-58Z-tf32-comparison` |
| Oct 8 UTC smaller FP32/TF32 shapes, separately declared 1000-GEMM batches and raw telemetry | `/home/coral/inference-artifacts/flops-ceiling/2026-10-08T23-30-22Z-tf32-small-shapes` |
| TF32 implementation, protocols, rounds, failures/recovery and result analysis journals | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/tf32` |
| Oct 7 CPU-only analysis of the 3072³ deficit: saved candidate tuning times, latency and post-trial clocks, RUN-0011 | `/home/coral/inference-artifacts/flops-ceiling/2026-10-07T22-28-46Z-fp32-3072-explanation` |
| Oct 7 FP32 candidate-shape measurements: predeclared protocol, commands, 80 trials, telemetry, Rust audit and decoded algorithms | `/home/coral/inference-artifacts/flops-ceiling/2026-10-07T22-13-14Z-fp32-candidate-shapes` |
| Oct 7 FP32 protocol, four rounds and measured analysis, RUN-0005–RUN-0010 | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/fp32` |
| Oct 7 FP32 theoretical shape analysis: reproducible Rust calculator, model output and environment; no new GPU run | `/home/coral/inference-artifacts/flops-ceiling/2026-10-07T21-48-23Z-fp32-shape-theory` |
| Oct 7 detailed theoretical derivation and assumptions, FP32 RUN-0004 | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/fp32/2026-10-07T21-48-23Z-fp32-shape-theory.md` |
| Oct 6 cuBLASLt shape comparison: protocol, exact commands, source snapshot, raw results, telemetry and Rust analysis | `/home/coral/inference-artifacts/flops-ceiling/2026-10-06T00-30-20Z-cublaslt-shape-sweep` |
| Oct 6 per-precision round journals, each RUN-0001–RUN-0003 | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/{bf16,fp8,nvfp4,fp32}` |
| Oct 3 build recovery and GPU correctness sweep: commands, logs, source snapshot and raw results | `/home/coral/inference-artifacts/flops-ceiling/2026-10-03T18-19-45Z-review-recovery` |
| Review follow-ups and Oct 6 protocol, metadata-helper recovery and analysis journals | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/mixed` |
| Archived journals, imported reports, research notes and repository-local output | `/home/coral/inference-artifacts/flops-ceiling/2026-09-29T21-59-16Z-report-consolidation` |
| Sep 28 counter captures, baselines and raw reports | `/home/coral/inference-artifacts/flops-ceiling/2026-09-28T22-02-50Z-container-nsight` |
| BF16/FP8/FP32 extension and cuTile follow-up | `/home/coral/inference-artifacts/flops-ceiling/2026-09-27T23-26-46Z-dtype-extension` |
| Paired Systems tracing comparison | `/home/coral/inference-artifacts/flops-ceiling/2026-09-27T23-02-33Z-nsight-comparison` |
| Longer NVFP4 register rerun | `/home/coral/inference-artifacts/flops-ceiling/2026-09-27T22-55-59Z-nvfp4-rerun` |
| Initial cuTile and cuBLASLt library comparisons | `/home/coral/inference-artifacts/flops-ceiling/2026-09-27T19-25-39Z-rust-library` |

The consolidation archive preserves original paths and checksums in
`migration-manifest.json`; no GPU experiments were rerun to prepare this report.
Imported inference estimates are historical references, not this library's TFLOPS
measurements. Update this report as conclusions change and retain detailed attempts outside Git.

The register methodology credits [secYOUre/nvfp4bench](https://github.com/secYOUre/nvfp4bench),
particularly its [peak MMA kernel](https://github.com/secYOUre/nvfp4bench/blob/main/src/peak_mma.cu).
Related packed-FP4 UE8M0 results must not be silently equated with our NVFP4
UE4M3 scaling. Attribution remains in [the specification](SPEC.md).
Metric interpretation follows NVIDIA's [Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [Systems GPU metrics guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics).
