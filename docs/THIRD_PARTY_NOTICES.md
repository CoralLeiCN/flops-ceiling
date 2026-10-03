# Third-party notices

The kernel pattern in [src/tile.rs](../src/tile.rs) is adapted from NVIDIA's
[cuTile Rust NVFP4 example](https://github.com/NVlabs/cutile-rs/blob/cc720f182f38bf46527753caa340e78d6d5fa3cc/cutile-examples/examples/nvfp4.rs),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, under the
[Apache License 2.0](../licenses/Apache-2.0.txt). Changes include BF16 output,
borrowed allocations, transposed output storage, and integration with this
project's validation, timing, and tile selection.

Dependencies retain their respective licenses. NVIDIA CUDA libraries and tools
are separately installed dependencies and are not redistributed here.
