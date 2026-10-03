# flops-ceiling

This project uses Rust to benchmark the TFLOPS performance of NVIDIA devices.
The first target is DGX Spark, with a focus on BF16, FP32, FP8, and FP4.

For GPU kernel development in Rust, see NVIDIA's
[Introducing CUDA Rust: Two Tracks for Writing GPU Kernels](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/).

The register-only FP4 benchmark follows the measurement approach of
[secYOUre/nvfp4bench](https://github.com/secYOUre/nvfp4bench).
See the [project specification and attribution](docs/SPEC.md).

[Build, run, and reuse the library](docs/benchmarking.md) ·
[Register throughput API](docs/register-ceiling.md) ·
[Nsight profiling](docs/profiling.md) ·
[Final performance report](docs/final-report.md)

Licensed under the [Apache License 2.0](LICENSE).
See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for attribution.
