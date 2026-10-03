# Nsight profiling on DGX Spark

Both Rust APIs support measurement capture: use
`registers::benchmark_with_profiler_range(&config)` for register-only kernels or
`flops_ceiling::benchmark_with_profiler_range(&config)` for GEMM. Both CLIs accept
`--profile-range` for BF16, FP8 E4M3, FP32 and FP4. GEMM capture works with
cuBLASLt and the optional cuTile backend. CUDA profiler start/stop calls surround the measured trials.
Setup, compilation, algorithm selection, warmup and validation stay outside the
capture. Cleanup stops capture on an early benchmark error as well.

Register telemetry is outside capture. GEMM telemetry between trials is outside
CUDA-event timing but inside the capture range; those inter-trial gaps include
telemetry collection and should not be attributed solely to kernel launch overhead.

This enables external Nsight tools; it does not start a profiler itself.
Manifest, JSON results and CSV summaries record `profiler_range`. It records the
requested range, not whether an external tool successfully collected data.
Run the ordinary `benchmark()` / CLI without an attached profiler to measure
**TFLOPS (no profiler attached)**. Nsight replay, instrumentation and capture
overhead can change timings.

Use these labels in new reports and comparison tables:

| Label | Meaning |
| --- | --- |
| TFLOPS (no profiler attached) | Benchmark throughput measured without an attached profiler, including Nsight Systems or Nsight Compute |
| TFLOPS (Nsight Systems attached) | Diagnostic throughput measured while Nsight Systems profiles the benchmark |
| TFLOPS (Nsight Compute attached) | Diagnostic throughput measured while Nsight Compute profiles the benchmark; replay can strongly affect timing |

Earlier reports call the first case "unprofiled TFLOPS" or "baseline TFLOPS".
Choose labels from the actual invocation: `--profile-range` only enables capture
boundaries and does not detect whether a profiler is attached. The CLI's generic
TFLOPS output and JSON/CSV fields therefore do not infer attachment status.

The same commands select the new formats with `--formats bf16`, `fp8` or
`fp32` for registers, or `--precision bf16`, `fp8` or `fp32` for GEMM.
Keep `--sparsities dense` for FP32. To trace a multi-format sweep, use
`--capture-range-end=repeat` and retain every generated report. Profiling remains
optional; the normal APIs do not invoke profiler start/stop.

## What to inspect

| Measurement | Tool / section | What it tells us |
| --- | --- | --- |
| Kernel duration, launch latency, GPU gaps, synchronization | Nsight Systems CUDA timeline | Whether the CPU keeps the GPU busy and how much time is spent between launches |
| Sampled Tensor Active, SMs Active, warp and issue activity | Nsight Systems GPU metrics, enabled separately | How pipeline activity varies through the capture; requires counter access |
| Tensor pipeline activity, precision-specific operation counters, instruction mix | Nsight Compute `ComputeWorkloadAnalysis`, `InstructionStats` | Whether tensor execution approaches its available throughput and how much other work executes |
| Registers, theoretical/achieved occupancy, active warps | `LaunchStats`, `Occupancy` | How accumulator count and launch geometry constrain concurrent work |
| Eligible warps, issue activity, dependency/pipeline stalls | `SchedulerStats`, `WarpStateStats` | Whether dependencies or an already-busy execution pipeline limit issuing instructions |
| L1/L2 traffic, cache hits, local-memory loads/stores | `MemoryWorkloadAnalysis` | Whether the register kernel spills or generates unexpected memory traffic; also useful for GEMM |
| DRAM read/write activity attributed to GPU or CPU | Compute `mcc__*` counters | Activity at Spark's memory controller, including the CPU/GPU split |
| SM clocks, temperature, power, GPU utilization | Existing before/after telemetry | Context for timing changes; snapshots do not measure instantaneous in-kernel peaks |

For tensor register modes, start with tensor activity, issue activity, register
usage and occupancy. For FP32, inspect the FP32/FMA execution pipeline instead
of Tensor Core activity. Higher occupancy alone does not establish better performance.
Memory traffic includes the final checksum stores and FP32 initial operand loads;
the arithmetic loop has no operand loads. Full GEMM needs separate attention to memory movement.

The installed Compute 2025.3.1 GB10B metric catalog advertises these base names:
`sm__ops_path_tensor_src_fp4_dst_fp32`, `sm__pipe_tensor_cycles_active`,
`sm__inst_executed_pipe_tensor`, `smsp__warps_eligible`, and
`lts__t_sector_hit_rate`. Query names for the installed tool/device before
requesting custom metrics. Catalog availability is not a successful measurement.
For the added types, the same catalog lists
`sm__ops_path_tensor_src_bf16_dst_fp32` for BF16 and
`sm__ops_path_tensor_src_fp4_fp6_fp8_dst_fp32` for the grouped low-precision
Tensor Core path used by FP8. The latter is not an FP8-only attribution.
FP32 can use `sm__pipe_fma_cycles_active` and
`sm__sass_thread_inst_executed_op_ffma_pred_on`, alongside the corresponding
`fadd` and `fmul` instruction counters. A scalar FMA counts two FLOPs; do not
multiply warp-level and thread-level counters interchangeably. Container captures
have now measured the NVFP4 and FP32 counters for both Rust register kernels and
cuBLASLt. BF16/FP8 entries remain discovered names, without hardware-counter
measurements in this project. The successful container approach is documented below.

The queried GB10B catalog uses `mcc__*` memory-controller counters rather than
`dram__*`, including `mcc__dram_throughput_srcnode_gpu_op_read` and
`mcc__dram_throughput_srcnode_gpu_op_write`. Do not interpret L2 traffic as
measured DRAM bandwidth or substitute unavailable counters with zero.
Tensor utilization percentages are not directly a percentage of advertised
sparse NVFP4 TFLOPS. Preserve this project's explicit dense/sparse FLOP accounting.

Section meanings and replay behavior are described in NVIDIA's
[Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
The [Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
describes CUDA timeline and capture-range options.

## Capture a timeline

Build first; choose one format, sparsity and input combination per report.
Use new output names for each invocation.

```sh
cargo build --release --locked --bin register-ceiling
mkdir -p artifacts
nsys profile --trace=cuda --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output artifacts/nsys-nvfp4-dense \
  target/release/register-ceiling --profile-range \
  --formats nvfp4 --sparsities dense --warmup 10 \
  --trials 3 --launches-per-trial 100 \
  --output artifacts/nsys-nvfp4-dense-target

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace \
  artifacts/nsys-nvfp4-dense.nsys-rep
```

This example requests 300 measured `register_mma` launches. Check the trace's
kernel count and profiler diagnostics as well as the target's checksum validation.
Keep captured timings separate from measurements with no profiler attached.

`--capture-range-end=stop` captures the first range only. The example deliberately
uses one configuration and one round; run a separate capture for another case.
Use the profiler's repeated-range options explicitly when profiling a sweep.

For cuBLASLt GEMM, the same control is available on `flops-ceiling`:

```sh
cargo build --release --locked --bins
nsys profile --trace=cuda --sample=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output artifacts/nsys-cublaslt-4096 \
  target/release/flops-ceiling --profile-range \
  --backend cublaslt --precision nvfp4 --shapes 4096x4096x4096 \
  --selection heuristic --timing graph --warmup 100 \
  --iterations 1000 --trials 3 --output artifacts/nsys-cublaslt-4096-target
```

`--cuda-graph-trace=node` records kernels inside graphs and can add more overhead
than graph-level tracing. Keep this setting with the results. For a profiling
overhead comparison, run fresh measurements with identical inputs, repetitions,
timing mode and algorithm selection, alternating runs with and without the
profiler attached.
`--selection heuristic` avoids an independent timing-based retune. Compare all
public algorithm configuration attributes through
`cublasLtMatmulAlgoConfigGetAttribute`, including tile, stages, split-K, reduction,
swizzle, custom option and inner/cluster shapes, along with workspace size.
Our recorded opaque descriptor bytes varied in an internal word even though all
nine public attributes matched. The [cuBLAS API](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgo-t)
supports restoring those descriptors with the same library version for inspection;
raw byte equality alone is not a reliable configuration comparison. Prior tuned
results are not a matched baseline for this heuristic comparison.

## Capture kernel counters

```sh
ncu --profile-from-start off --kernel-name register_mma --launch-count 1 \
  --clock-control none --cache-control none --pipeline-boost-state dynamic \
  --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section LaunchStats --section Occupancy --section SchedulerStats \
  --section WarpStateStats --section MemoryWorkloadAnalysis \
  --section InstructionStats --export artifacts/ncu-nvfp4-dense \
  target/release/register-ceiling --profile-range \
  --formats nvfp4 --sparsities dense --warmup 10 --trials 3 \
  --output artifacts/ncu-nvfp4-dense-target
```

This requests one measured kernel, excluding warmup and probes. It can require
multiple replay passes; a single capture is preliminary. The explicit clock,
cache and pipeline options avoid locking clocks, flushing caches or forcing
stable tensor boost. Uncontrolled clocks/cache state can vary between passes;
the tool warns about this, and the exact settings must accompany the report.
Check the profiler's exit status as well as the target program's status.

## Container access on DGX Spark

The installed GB10 / driver 580.173.02 stack restricts host counter access with
`RmProfilingAdminOnly=1`. A host-launched capture can fail with
`ERR_NVGPUCTRPERM` even when the target passes validation. See NVIDIA's
[counter permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

The working container configuration uses `--gpus all` with these capabilities:

| Tool | Container capability | Collection |
| --- | --- | --- |
| Nsight Systems 2025.3.2 | `--cap-add=PERFMON` | CUDA timeline and sampled GPU metrics |
| Nsight Compute 2025.3.1 | `--cap-add=SYS_ADMIN` | Kernel hardware counters; `PERFMON` alone was insufficient on this stack |

Mount the installed CUDA toolkit and Nsight tools read-only, use an output
location outside Git, and run the benchmark under the profiler as the container
entrypoint. The archived commands use a pinned image, disabled networking and
writable artifact output. The host profiling policy does not need to change.
The Rust `--profile-range` boundaries work in the container as on the host.

Systems GPU metrics require
`--gpu-metrics-devices=all --gpu-metrics-frequency=1000` in addition to CUDA
tracing. The timeline examples above do not enable that sampling. Compute
collection can use multiple replay passes; capture only the kernels needed for
the diagnostic. Run separately without an attached profiler for throughput.

Inspect Systems' `DIAGNOSTIC_EVENT` records as well as kernel counts, target
validation and exit codes. A successful exit and the expected kernel count do
not establish complete event coverage. cuBLASLt may include workspace clears
inside a GEMM, so valid captures need not contain kernels alone. Device-wide
sample means can include host gaps and uneven sampling intervals.

The [final report](final-report.md#nsight-measurements) contains the measured
Tensor/FMA activity, operation-count checks and unresolved trace warnings.
Its [evidence locations](final-report.md#environment-reproduction-and-evidence)
point to exact commands, raw profiler reports and the archived investigation
that established this container setup. Profiling remains optional.
