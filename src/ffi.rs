//! Private subset of the CUDA Runtime / cuBLASLt C ABI (CUDA 13.0 headers).
//! Opaque objects never escape the safe library API. Enum values are C ints.
#![allow(non_snake_case)]
use std::ffi::{c_char, c_int, c_void};
pub type Handle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Algo {
    pub data: [u64; 8],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Heuristic {
    pub algo: Algo,
    pub workspace_size: usize,
    pub state: c_int,
    pub waves: f32,
    pub reserved: [c_int; 4],
}

unsafe extern "C" {
    pub fn cudaGetErrorString(code: c_int) -> *const c_char;
    pub fn cudaGetDevice(device: *mut c_int) -> c_int;
    pub fn cudaSetDevice(device: c_int) -> c_int;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    pub fn cudaDeviceGetPCIBusId(id: *mut c_char, len: c_int, device: c_int) -> c_int;
    pub fn cudaRuntimeGetVersion(version: *mut c_int) -> c_int;
    pub fn cudaDriverGetVersion(version: *mut c_int) -> c_int;
    pub fn cudaProfilerStart() -> c_int;
    pub fn cudaProfilerStop() -> c_int;
    pub fn cudaMalloc(ptr: *mut Handle, bytes: usize) -> c_int;
    pub fn cudaFree(ptr: Handle) -> c_int;
    pub fn cudaMemcpy(dst: Handle, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
    pub fn cudaMemsetAsync(dst: Handle, value: c_int, bytes: usize, stream: Handle) -> c_int;
    pub fn cudaStreamCreateWithFlags(stream: *mut Handle, flags: u32) -> c_int;
    pub fn cudaStreamDestroy(stream: Handle) -> c_int;
    pub fn cudaStreamSynchronize(stream: Handle) -> c_int;
    pub fn cudaEventCreate(event: *mut Handle) -> c_int;
    pub fn cudaEventRecord(event: Handle, stream: Handle) -> c_int;
    pub fn cudaEventSynchronize(event: Handle) -> c_int;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> c_int;
    pub fn cudaEventDestroy(event: Handle) -> c_int;
    pub fn cudaStreamBeginCapture(stream: Handle, mode: c_int) -> c_int;
    pub fn cudaStreamEndCapture(stream: Handle, graph: *mut Handle) -> c_int;
    pub fn cudaGraphInstantiateWithFlags(exec: *mut Handle, graph: Handle, flags: u64) -> c_int;
    pub fn cudaGraphLaunch(exec: Handle, stream: Handle) -> c_int;
    pub fn cudaGraphExecDestroy(exec: Handle) -> c_int;
    pub fn cudaGraphDestroy(graph: Handle) -> c_int;
    pub fn cublasLtGetVersion() -> usize;
    pub fn cublasLtCreate(handle: *mut Handle) -> c_int;
    pub fn cublasLtDestroy(handle: Handle) -> c_int;
    pub fn cublasLtMatmulDescCreate(desc: *mut Handle, compute: c_int, scale: c_int) -> c_int;
    pub fn cublasLtMatmulDescDestroy(desc: Handle) -> c_int;
    pub fn cublasLtMatmulDescSetAttribute(
        desc: Handle,
        attr: c_int,
        value: *const c_void,
        size: usize,
    ) -> c_int;
    pub fn cublasLtMatrixLayoutCreate(
        desc: *mut Handle,
        dtype: c_int,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> c_int;
    pub fn cublasLtMatrixLayoutDestroy(desc: Handle) -> c_int;
    pub fn cublasLtMatmulPreferenceCreate(pref: *mut Handle) -> c_int;
    pub fn cublasLtMatmulPreferenceDestroy(pref: Handle) -> c_int;
    pub fn cublasLtMatmulPreferenceSetAttribute(
        pref: Handle,
        attr: c_int,
        value: *const c_void,
        size: usize,
    ) -> c_int;
    pub fn cublasLtMatmulAlgoGetHeuristic(
        lt: Handle,
        op: Handle,
        a: Handle,
        b: Handle,
        c: Handle,
        d: Handle,
        pref: Handle,
        requested: c_int,
        results: *mut Heuristic,
        returned: *mut c_int,
    ) -> c_int;
    pub fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const Algo,
        attr: c_int,
        value: Handle,
        size: usize,
        written: *mut usize,
    ) -> c_int;
    pub fn cublasLtMatmul(
        lt: Handle,
        op: Handle,
        alpha: *const c_void,
        a: *const c_void,
        ad: Handle,
        b: *const c_void,
        bd: Handle,
        beta: *const c_void,
        c: *const c_void,
        cd: Handle,
        d: Handle,
        dd: Handle,
        algo: *const Algo,
        workspace: Handle,
        workspace_bytes: usize,
        stream: Handle,
    ) -> c_int;
}
