# Package specification

This package uses Rust to measure NVIDIA GPU TFLOPS, initially targeting DGX
Spark with BF16, FP32, FP8, and FP4 benchmarks. This document describes the
reusable package's implemented behavior. Experiment-specific intent,
specifications and plans are maintained in external experiment workspaces.

## Benchmark scope

- The GEMM API measures complete matrix multiplication, with numerical
  validation and explicit timing boundaries. Both cuBLASLt and the optional
  Rust cuTile backend support BF16, FP8 E4M3, NVFP4 and FP32. FP32 has FP32
  output; the other inputs use BF16 output. All accumulate in FP32.
  cuBLASLt additionally supports explicit TF32 multiplication with FP32
  storage/accumulation/output, matched FP32 inputs, validation against original
  inputs, and algorithm flags verifying the advertised TF32 Tensor Core
  implementation (not profiler counters). TF32 is
  not implemented for cuTile or the register API.
- Every precision shares tuning, graph/stream timing, structured artifacts,
  telemetry and optional Nsight capture. Full FP32 is never labeled TF32;
  cuBLASLt uses its pedantic FP32 compute mode.
- The separate register API measures repeated BF16, FP8 E4M3 and FP4 tensor
  instructions, plus full FP32 scalar FMA on CUDA cores. Tensor modes support
  dense and structured 2:4 sparsity; FP32 supports dense only. NVFP4 UE4M3
  and MXFP4 UE8M0 block scaling remain specific to FP4. Uniform A/B values,
  block scales, accumulator initialization, launch geometry and timing counts
  are configurable through Rust and the CLI.
- Sparse instruction throughput uses dense-equivalent FLOP counts. Register-only
  results are reported separately from full GEMM throughput.

## Prior work and attribution

The register-only benchmark credits
**[secYOUre/nvfp4bench](https://github.com/secYOUre/nvfp4bench)** for its measurement
approach and instruction-level reference implementation, particularly
[`src/peak_mma.cu`](https://github.com/secYOUre/nvfp4bench/blob/main/src/peak_mma.cu).
That work demonstrates repeated register-resident FP4 MMA instructions with
independent accumulators to measure dense and sparse tensor-core throughput.

Our implementation follows that approach through a separate Rust API that
generates PTX and launches it through the CUDA Driver API. It includes NVFP4
UE4M3 scaling, checksum validation, configurable repetitions, and structured
result records. The original register-only methodology is credited to
`secYOUre/nvfp4bench`.

See [GEMM methodology](benchmarking.md),
[register-only API and methodology](register-ceiling.md), and the
[final performance report](final-report.md)
for implementation details and measured results.
