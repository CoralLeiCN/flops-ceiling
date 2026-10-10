# Experiment intent

`GB10-GEMM-SHAPES-001` · Reviewed 2026-10-10 (Europe/London).

Use one experiment, versioned in this repository and separate from the
`flops-ceiling` package, to:

1. Understand the number of streaming multiprocessors (SMs) and CUDA cores in GB10.
2. Understand how those hardware specifications relate to matrix shape and GEMM performance.
3. Find the best-performing GEMM shapes.

Start with all GB10-supported Tensor Core modes, including integer and scaled variants.
