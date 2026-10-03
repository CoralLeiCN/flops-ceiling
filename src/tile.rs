// NVFP4 kernel pattern adapted from NVIDIA's cuTile Rust example.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Optional Rust-authored GEMM kernels. Compilation precedes graph capture.
use crate::{Error, Precision, Result, Shape, ffi::Handle};
use cutile::{
    core::bf16,
    cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn},
    prelude::*,
};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn dense<E: ElementType, O: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        output: &mut Tensor<O, { [BM, BN] }>,
        lhs: &Tensor<E, { [-1, -1] }>,
        rhs: &Tensor<E, { [-1, -1] }>,
    ) {
        let id = get_tile_block_id();
        let a = lhs.partition(shape![BM, BK]);
        let b = rhs.partition(shape![BN, BK]);
        let mut accum: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for k in 0i32..num_tiles(&a, 1) {
            accum = mma(a.load([id.0, k]), b.load([id.1, k]).transpose(), accum);
        }
        let rounded: Tile<O, { [BM, BN] }> = convert_tile(accum);
        output.store(rounded);
    }

    #[cutile::entry()]
    fn nvfp4<const BM: i32, const BN: i32, const BK: i32, const KP: i32, const KS: i32>(
        output: &mut Tensor<bf16, { [BM, BN] }>,
        lhs: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        rhs: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        lhs_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        rhs_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let id = get_tile_block_id();
        let a = lhs.partition(shape![BM, KP]);
        let b = rhs.partition(shape![BN, KP]);
        let sa = lhs_scales.partition(shape![BM, KS]);
        let sb = rhs_scales.partition(shape![BN, KS]);
        let mut accum: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for k in 0i32..num_tiles(&a, 1) {
            let av = a.load([id.0, k]).unpack(shape![BM, BK]);
            let bv = b.load([id.1, k]).unpack(shape![BN, BK]).transpose();
            let ascale = sa.load([id.0, k]);
            let bscale = sb.load([id.1, k]).transpose();
            accum = mmaf_scaled(av, bv, accum, ascale, bscale);
        }
        let rounded: Tile<bf16, { [BM, BN] }> = convert_tile(accum);
        output.store(rounded);
    }
}

pub(crate) struct Kernel {
    stream: Arc<Stream>,
    precision: Precision,
    a: Handle,
    b: Handle,
    a_scale: Handle,
    b_scale: Handle,
    output: Handle,
    shape: Shape,
    device: usize,
}
impl Kernel {
    /// All pointers are borrowed from Plan. Plan synchronizes before freeing
    /// them, and all accesses use its single stream, so no aliased writes race.
    pub(crate) unsafe fn new(
        device: i32,
        precision: Precision,
        stream: Handle,
        shape: Shape,
        a: Handle,
        b: Handle,
        a_scale: Handle,
        b_scale: Handle,
        output: Handle,
    ) -> Result<Self> {
        let device = Device::new(device as usize).map_err(|e| Error(e.to_string()))?;
        let ordinal = device.ordinal();
        let stream = unsafe { Stream::borrow_raw(stream, &device) };
        Ok(Self {
            stream,
            precision,
            a,
            b,
            a_scale,
            b_scale,
            output,
            shape,
            device: ordinal,
        })
    }
    pub(crate) fn launch(&self, tile: [usize; 3]) -> Result<()> {
        match self.precision {
            Precision::Bf16 => return self.launch_dense::<bf16, bf16>(tile),
            Precision::Fp8 => return self.launch_dense::<f8e4m3fn, bf16>(tile),
            Precision::Fp32 => return self.launch_dense::<f32, f32>(tile),
            Precision::Nvfp4 => (),
        }
        let [bm, bn, bk] = tile;
        // Writing B*A^T in row-major order gives the same column-major result
        // bytes as cuBLASLt's A*B^T. Inputs and values are identical.
        let out = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                self.output as u64,
                self.device,
                vec![self.shape.n as i32, self.shape.m as i32],
                vec![self.shape.m as i32, 1],
            )
        };
        let op = kernels::nvfp4(
            out.partition([bm, bn]),
            self.view::<f4e2m1fnx2>(self.b, self.shape.n, self.shape.k / 2),
            self.view::<f4e2m1fnx2>(self.a, self.shape.m, self.shape.k / 2),
            self.view::<f8e4m3fn>(self.b_scale, self.shape.n, self.shape.k / 16),
            self.view::<f8e4m3fn>(self.a_scale, self.shape.m, self.shape.k / 16),
        )
        .generics(
            [bm, bn, bk, bk / 2, bk / 16]
                .map(|v| v.to_string())
                .to_vec(),
        );
        // No device allocations or synchronization in the prepared launch path. The
        // first launch JIT-compiles; callers do it before all timed work.
        unsafe { op.execute(&ExecutionContext::new(self.stream.clone())) }
            .map_err(|e| Error(e.to_string()))?;
        Ok(())
    }

    fn view<T: cutile::cuda_core::DType>(
        &self,
        ptr: Handle,
        rows: usize,
        cols: usize,
    ) -> Arc<Tensor<T>> {
        // Plan owns the correctly typed buffers and outlives every launch.
        Arc::new(unsafe {
            Tensor::borrow_raw_parts(
                ptr as u64,
                self.device,
                vec![rows as i32, cols as i32],
                vec![cols as i32, 1],
            )
        })
    }

    fn launch_dense<E: cutile::cuda_core::DType, O: cutile::cuda_core::DType>(
        &self,
        tile: [usize; 3],
    ) -> Result<()> {
        let [bm, bn, bk] = tile;
        let out = unsafe {
            Tensor::<O>::borrow_raw_parts(
                self.output as u64,
                self.device,
                vec![self.shape.n as i32, self.shape.m as i32],
                vec![self.shape.m as i32, 1],
            )
        };
        let op = kernels::dense(
            out.partition([bm, bn]),
            self.view::<E>(self.b, self.shape.n, self.shape.k),
            self.view::<E>(self.a, self.shape.m, self.shape.k),
        )
        .generics(vec![
            E::DTYPE.as_str().into(),
            O::DTYPE.as_str().into(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
        ]);
        unsafe { op.execute(&ExecutionContext::new(self.stream.clone())) }
            .map_err(|e| Error(e.to_string()))?;
        Ok(())
    }
}
