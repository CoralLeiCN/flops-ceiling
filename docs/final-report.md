# DGX Spark performance: final report

Consolidated on 2026-09-29 from measurements on 2026-09-27–28.
Build and correctness verification updated on 2026-10-03; the performance
tables below retain their original measurement cohorts.
This is the repository's single maintained results report. Detailed journals,
imported reports, research notes and raw artifacts are stored outside Git.

The Rust benchmarks produce validated BF16, FP8, NVFP4 and full FP32 results.
The strongest NVFP4 observations are approximately **493 TFLOPS for repeated
register instructions** and **334 TFLOPS for a 4096³ cuBLASLt GEMM** in the latest
short session. These are different workloads and do not establish sustained
hardware ceilings or inference throughput.

## Implementations and measurement scope

There are three implementations behind two APIs:

| Implementation | API | Supported arithmetic |
| --- | --- | --- |
| cuBLASLt | GEMM | BF16, FP8 E4M3, NVFP4, full FP32 |
| Native Rust cuTile, optional feature | GEMM | BF16, FP8 E4M3, NVFP4, full FP32 |
| Rust-generated PTX | Registers | BF16, FP8 E4M3, NVFP4, MXFP4, full FP32 |

Tensor register modes support dense and structured 2:4 sparse instructions;
FP32 registers use dense scalar FMA. GEMM accumulates in FP32, with BF16 output
except for FP32 inputs, which also use FP32 output. cuBLASLt FP32 uses PEDANTIC
compute without TF32 substitution. Register kernels reuse uniform operands;
complete GEMM also moves matrices and scales through the memory hierarchy.

## TFLOPS (no profiler attached)

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
No usable Git revision was available; source snapshots, Cargo.lock and binary
SHA-256 hashes identify measured builds.

See [GEMM usage](benchmarking.md), [register usage](register-ceiling.md) and
[profiler commands](profiling.md). Exact historical commands, UTC timestamps,
exit states, inputs, launch settings, raw trials and diagnostics remain in:

| Evidence | External archive |
| --- | --- |
| Oct 3 build recovery and GPU correctness sweep: commands, logs, source snapshot and raw results | `/home/coral/inference-artifacts/flops-ceiling/2026-10-03T18-19-45Z-review-recovery` |
| Review follow-up journal and directory-local run IDs | `/home/coral/inference-artifacts/flops-ceiling/journals/dgx-spark/mixed` |
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
