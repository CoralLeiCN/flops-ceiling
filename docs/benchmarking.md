# Running and reusing the benchmark

The Rust library and CLI measure dense GEMM with FP32 accumulation. Both
cuBLASLt and the optional Rust cuTile backend support the following inputs:

| `--precision` / Rust `Precision` | Inputs | Output | Register API |
| --- | --- | --- | --- |
| `bf16` / `Bf16` | BF16 | BF16 | Dense and 2:4 sparse MMA |
| `fp8` / `Fp8` | FP8 E4M3 | BF16 | Dense and 2:4 sparse MMA |
| `nvfp4` / `Nvfp4` | E2M1, E4M3 block-16 scales | BF16 | Dense and 2:4 sparse MMA |
| `fp32` / `Fp32` | Full FP32 | FP32 | Dense scalar FMA on CUDA cores |

Every GEMM precision shares shape/seed controls, validated candidate selection,
tuning, stream/graph timing, warmup, repeated trials, telemetry, JSON/CSV and
optional Nsight capture. cuTile tunes up to eight tile configurations. Its
kernels are initial implementations, not claims of optimal throughput.

FP32 cuBLASLt uses `CUBLAS_COMPUTE_32F_PEDANTIC`, which retains full FP32
arithmetic instead of substituting TF32. cuTile uses FP32 inputs directly in
its matrix operation. Results record `output_precision` and `compute_mode`
(schema 2; these fields default to empty strings when reading older records).
See NVIDIA's [compute-type definitions](https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html#cublascomputetype-t)
and [Tile IR matrix operations](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html#cuda-tile-mmaf).

The separate [register throughput API](register-ceiling.md) measures reused
register operands. Its instruction TFLOPS has a different scope from GEMM.

## Build and run

Use Linux, Rust 1.89 or newer, an NVIDIA driver, and a CUDA 13.x toolkit containing
`libcudart` and `libcublasLt`. CUDA 13.0 and Rust 1.98.1 were used for the multi-precision DGX Spark runs.
Set `CUDA_HOME` when the toolkit is not installed at `/usr/local/cuda`.

```sh
cargo build --release --locked
cargo run --release --locked -- --precision nvfp4 --output artifacts/nvfp4-001
cargo run --release --locked -- --precision fp8 --output artifacts/fp8-001
cargo run --release --locked -- --precision bf16 --output artifacts/bf16-001
cargo run --release --locked -- --precision fp32 --output artifacts/fp32-001
```

The default shapes are `64x4096x4096,512x4096x4096,4096x4096x4096`. Add the
rectangular comparison with `--shapes 4096x4096x4096,4096x14336x4096`. M and N
must be positive multiples of 16, and K a positive multiple of 32. Backend
support may impose further restrictions; unsupported configurations return
errors and never produce throughput results.

Each invocation creates a **new** output directory and refuses to overwrite an
existing one. It writes `manifest.json`, `results.jsonl`, `summary.csv`, a copy
of the available `Cargo.lock`, and `errors.json` if a shape fails. Every trial,
candidate's validation and tuning measurements, library version, loaded library
path, device identity, and available telemetry are retained. A failure gives a
nonzero exit code even if other shapes succeed. Missing telemetry and unknown
Git revisions are recorded as unavailable. Keep large runs outside the checkout.

## Compare tuning, graphs, and library versions

```sh
target/release/flops-ceiling --selection heuristic --timing stream --output artifacts/heuristic-stream
target/release/flops-ceiling --selection tune --timing stream --output artifacts/tuned-stream
target/release/flops-ceiling --selection tune --timing graph --output artifacts/tuned-graph
```

All modes reuse allocated inputs, outputs, workspace, and descriptors. Heuristic
mode takes the first candidate that passes validation. Tune mode requests up to
32 candidates, validates each, performs 5 warmup launches, and chooses the lowest
median of three 20-GEMM tuning batches. Selection and compilation are excluded
from final measurements. The final measurement uses 10 warmup launches and five
100-GEMM trials by default. Graph mode captures the whole batch and performs one
additional untimed replay before measuring it. `--help` lists count and workspace
overrides. A single trial has no sample standard deviation and is preliminary.

Tuning may choose different algorithms between stream and graph runs. Compare
the public algorithm attributes and workspace size before attributing a difference
entirely to launch overhead. Recorded `opaque_config` can be restored with the
same cuBLASLt version to query those attributes; private bytes may vary even
when the public configuration is identical.
For a comparison using the first heuristic in both modes, use
`--selection heuristic` for both. Match hardware state and inputs; repeat in
alternating order before naming a winner.

To compare a newer cuBLASLt without replacing system CUDA, install NVIDIA's
package into an isolated directory and change the runtime library search path:

```sh
python3 -m pip install --target /tmp/flops-cublas 'nvidia-cublas==13.8.0.4'
LD_LIBRARY_PATH=/tmp/flops-cublas/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  target/release/flops-ceiling --output artifacts/new-cublas
```

Python is only a package installer here; the benchmark calls the C API from
Rust. Check `device.cublas_version` and `device.loaded_libraries` in the result
to confirm which library actually loaded. Toolkit and cuBLAS component version
numbers differ. This project's run used CUDA Runtime 13.0 with cuBLAS 13.8.0.4
and driver 580.173.02. Other combinations require their own verification.

## Native Rust cuTile kernels

```sh
python3 -m pip install --target /tmp/flops-tile \
  'nvidia-cuda-tileiras==13.4.92' 'nvidia-cuda-nvcc==13.4.92' 'nvidia-nvvm==13.4.92'
cargo build --release --locked --features cutile
PATH=/tmp/flops-tile/nvidia/cu13/bin:$PATH \
CUTILE_TILEIRAS_PATH=/tmp/flops-tile/nvidia/cu13/bin/tileiras \
  target/release/flops-ceiling --backend cutile --precision bf16 --output artifacts/cutile-bf16-001
```

The feature pins cuTile Rust to revision
`cc720f182f38bf46527753caa340e78d6d5fa3cc`. Its build needs libclang and standard C
headers. On the measured host, bindgen needed
`BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/aarch64-linux-gnu/13/include'`;
use the installed compiler's corresponding path if this issue occurs elsewhere.

For an offline build check, first populate Cargo's cache while online. The
following commands were verified on the DGX Spark host:

```sh
cargo fetch --locked
BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/aarch64-linux-gnu/13/include' \
  cargo check --locked --offline --all-targets --features cutile
```

The offline check requires the optional feature's dependencies to be cached too;
a missing crate download is a dependency-availability failure before compilation.

The matching `ptxas` supplied by `nvidia-cuda-nvcc` must accompany `tileiras`.
Installing the package with its dependencies supplies both. A newer `tileiras`
combined with the system's older `ptxas` failed in our first attempt. FP4 needs
Tile IR 13.3 or newer. The system toolkit and driver do not need to be replaced
just to try an isolated assembler; the resulting cubin must still load and pass
validation on the actual device.

NVFP4 uses packed E2M1 operands, row-major E4M3 block scales and block-scaled
MMA. BF16, FP8 and FP32 use an unscaled matrix operation with an explicitly
typed FP32 accumulator. The output type follows the table above. Select another
format with `--precision fp8`, `fp32` or `nvfp4`. Each tries output tiles 64×64, 128×64, 64×128, and 128×128,
each with K tiles 128 and 256 for BF16/FP8/NVFP4. FP32 uses 16×16, 16×32,
32×16 and 32×32 output tiles, each with K=8 or 16; the larger tensor tiles
caused extensive local-memory spills in the FP32 compiler output. Because
output storage matches cuBLASLt's
column-major result, its tile tuple is `[tile_N, tile_M, tile_K]`. These values
occupy the first three `opaque_config` words; `algorithm_id=-1` means no cuBLAS
algorithm is involved. Failed candidates are retained as errors and excluded
from selection. The kernel is an initial implementation, with performance to be
judged from the recorded measurements.

## Library API

Add a path dependency, or a Git dependency once the repository is hosted:

```toml
[dependencies]
flops-ceiling = { path = "/path/to/flops-ceiling" }
```

```rust
use flops_ceiling::{benchmark, BenchmarkConfig, Precision, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = benchmark(&BenchmarkConfig {
        precision: Precision::Nvfp4,
        shape: Shape { m: 4096, n: 4096, k: 4096 },
        ..Default::default()
    })?;
    println!("{:.2} TFLOPS", result.tflops.mean);
    Ok(())
}
```

The executable example is `examples/benchmark.rs`. The library returns a
serializable `BenchmarkResult`; it does not write artifact files itself. Each
call owns and cleans up its CUDA resources and restores the caller's current
device. Run benchmarks sequentially to avoid GPU contention. The application's
loader must be able to find CUDA libraries; use `LD_LIBRARY_PATH` if they are
not registered in the system library search path.

## What a result means

- Useful work is `2*M*N*K` per GEMM, counting a multiply-add as two operations.
  TFLOPS is that work divided by CUDA-event time. It is dense throughput, without
  a sparsity multiplier. Padded tile work is not added to useful FLOPs.
- Timing excludes input creation/encoding, host transfers, allocation, tuning,
  graph setup, and validation. Host submission-plus-synchronization time is also
  reported. It is not an inference or quantization-inclusive benchmark.
- Inputs and output allocations are reused, with no explicit L2 flush. This is
  a warm, repeated-operand workload; it does not establish cold-cache throughput.
- Inputs are deterministic, varied, signed, and directly encoded. NVFP4 block
  scales vary over 0.375, 0.75, 1, and 1.5. This checks GEMM on encoded values,
  not the quality of quantizing a model. Use `--seed` for a different case.
- Validation scans every output for NaN/Inf, and by default compares 256 corner,
  tile-boundary, and distributed outputs with CPU FP64 sums of the decoded
  inputs. Relative RMSE must be below 0.01; each sampled absolute error must be
  at most `0.01*abs(reference) + 0.005*reference_RMS` (floor 1e-12) for BF16
  output. FP32 output has stricter limits: relative RMSE below `1e-5`, and
  sample error at most `1e-5*abs(reference) + 5e-6*reference_RMS` (same floor). Set
  `--validation-samples M*N` for exhaustive checks on small cases. Sampling is
  not exhaustive correctness proof for large matrices.
- Output is poisoned with NaNs before candidate validation, preventing an
  unwritten output from passing. The chosen output is checked again after timing.
- Telemetry snapshots occur outside timing. They are not a continuous power or
  thermal trace. Automatic GPU clocks and operating conditions can change results.

CUTLASS/cuDNN/b12x are not backends in this version. Their imported numbers and
research are references, not measurements made by this library. See the
[final report](final-report.md) for observed results, limitations and external
artifact locations. Detailed experiment output stays outside Git.

For optional Nsight capture, use `benchmark_with_profiler_range(&config)` or the
CLI's `--profile-range`. See [profiling commands and measurement boundaries](profiling.md).
