//! Private CUDA Driver ABI, checked against the installed CUDA 13.0 cuda.h.
#![allow(non_snake_case)]
use super::{DeviceInfo, KernelInfo};
use crate::{Error, Result};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    ptr::null_mut,
    rc::Rc,
};

type Handle = *mut c_void;
unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuGetErrorString(error: i32, text: *mut *const c_char) -> i32;
    fn cuDriverGetVersion(version: *mut i32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, len: i32, device: i32) -> i32;
    fn cuDeviceGetPCIBusId(name: *mut c_char, len: i32, device: i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    fn cuDevicePrimaryCtxRetain(context: *mut Handle, device: i32) -> i32;
    fn cuDevicePrimaryCtxRelease_v2(device: i32) -> i32;
    fn cuCtxPushCurrent_v2(context: Handle) -> i32;
    fn cuCtxPopCurrent_v2(context: *mut Handle) -> i32;
    fn cuModuleLoadDataEx(
        module: *mut Handle,
        image: *const c_void,
        count: u32,
        options: *mut i32,
        values: *mut Handle,
    ) -> i32;
    fn cuModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char) -> i32;
    fn cuModuleUnload(module: Handle) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: i32, function: Handle) -> i32;
    fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> i32;
    fn cuMemFree_v2(ptr: u64) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuStreamCreate(stream: *mut Handle, flags: u32) -> i32;
    fn cuStreamSynchronize(stream: Handle) -> i32;
    fn cuStreamDestroy_v2(stream: Handle) -> i32;
    fn cuEventCreate(event: *mut Handle, flags: u32) -> i32;
    fn cuEventRecord(event: Handle, stream: Handle) -> i32;
    fn cuEventSynchronize(event: Handle) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> i32;
    fn cuEventDestroy_v2(event: Handle) -> i32;
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
    fn cuLaunchKernel(
        function: Handle,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared: u32,
        stream: Handle,
        params: *mut Handle,
        extra: *mut Handle,
    ) -> i32;
}

fn check(status: i32, op: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let mut detail = std::ptr::null();
    // SAFETY: the driver provides a process-lifetime string on success.
    let detail = unsafe {
        cuGetErrorString(status, &mut detail);
        if detail.is_null() {
            "unknown error".into()
        } else {
            CStr::from_ptr(detail).to_string_lossy().into_owned()
        }
    };
    Err(Error(format!("{op}: CUDA driver {status}: {detail}")))
}

// A synchronous, thread-confined owner. Partial construction is also cleaned up.
// Drop synchronizes its stream before destroying any resource it could access.
pub(super) struct Driver {
    device: i32,
    retained: bool,
    pushed: bool,
    module: Handle,
    functions: [Handle; 3],
    stream: Handle,
    start: Handle,
    end: Handle,
    output: u64,
    operands: u64,
    len: usize,
    _thread: PhantomData<Rc<()>>,
}
impl Driver {
    pub fn profiler_range(&self) -> Result<ProfilerRange<'_>> {
        // SAFETY: this driver's retained context is current on this thread.
        unsafe {
            check(cuProfilerStart(), "start profiler capture")?;
        }
        Ok(ProfilerRange {
            driver: self,
            active: true,
        })
    }

    pub fn new(ordinal: i32) -> Result<(Self, DeviceInfo)> {
        let mut d = Self {
            device: 0,
            retained: false,
            pushed: false,
            module: null_mut(),
            functions: [null_mut(); 3],
            stream: null_mut(),
            start: null_mut(),
            end: null_mut(),
            output: 0,
            operands: 0,
            len: 0,
            _thread: PhantomData,
        };
        // SAFETY: all output pointers refer to initialized, correctly sized storage.
        unsafe {
            check(cuInit(0), "cuInit")?;
            check(cuDeviceGet(&mut d.device, ordinal), "cuDeviceGet")?;
            let mut name = [0 as c_char; 128];
            let mut pci = [0 as c_char; 32];
            check(
                cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, d.device),
                "device name",
            )?;
            check(
                cuDeviceGetPCIBusId(pci.as_mut_ptr(), pci.len() as i32, d.device),
                "PCI bus",
            )?;
            let attr = |key| -> Result<i32> {
                let mut v = 0;
                check(
                    cuDeviceGetAttribute(&mut v, key, d.device),
                    "device attribute",
                )?;
                Ok(v)
            };
            let major = attr(75)?;
            let minor = attr(76)?;
            if (major, minor) != (12, 1) {
                return Err(Error(format!(
                    "register-only kernel targets GB10 SM121; found SM{major}{minor}"
                )));
            }
            let mut version = 0;
            check(cuDriverGetVersion(&mut version), "driver version")?;
            let info = DeviceInfo {
                ordinal,
                name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                pci_bus_id: CStr::from_ptr(pci.as_ptr()).to_string_lossy().into_owned(),
                compute_capability: format!("{major}.{minor}"),
                multiprocessors: attr(16)? as u32,
                l2_bytes: attr(38)?,
                max_threads_per_block: attr(1)? as u32,
                max_grid_x: attr(5)? as u32,
                driver_api_version: version,
            };
            let mut context = null_mut();
            check(
                cuDevicePrimaryCtxRetain(&mut context, d.device),
                "retain context",
            )?;
            d.retained = true;
            check(cuCtxPushCurrent_v2(context), "push context")?;
            d.pushed = true;
            check(cuStreamCreate(&mut d.stream, 1), "create stream")?;
            check(cuEventCreate(&mut d.start, 0), "create start event")?;
            check(cuEventCreate(&mut d.end, 0), "create end event")?;
            Ok((d, info))
        }
    }
    pub fn prepare(&mut self, ptx: &str, len: usize) -> Result<KernelInfo> {
        let ptx = CString::new(ptx).map_err(|e| Error(e.to_string()))?;
        let mut log = vec![0_u8; 16 * 1024];
        let mut options = [5, 6]; // CU_JIT_ERROR_LOG_BUFFER, ..._SIZE_BYTES
        let mut values = [log.as_mut_ptr().cast(), log.len() as Handle];
        // SAFETY: PTX and option buffers outlive synchronous module compilation.
        unsafe {
            let status = cuModuleLoadDataEx(
                &mut self.module,
                ptx.as_ptr().cast(),
                2,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            );
            if let Err(e) = check(status, "JIT PTX") {
                let end = log.iter().position(|x| *x == 0).unwrap_or(log.len());
                return Err(Error(format!(
                    "{e}\n{}",
                    String::from_utf8_lossy(&log[..end])
                )));
            }
            for (function, name) in
                self.functions
                    .iter_mut()
                    .zip([c"register_mma", c"probe_scale", c"probe_negative"])
            {
                check(
                    cuModuleGetFunction(function, self.module, name.as_ptr()),
                    "get kernel",
                )?;
            }
            let attr = |key| -> Result<i32> {
                let mut v = 0;
                check(
                    cuFuncGetAttribute(&mut v, key, self.functions[0]),
                    "kernel attribute",
                )?;
                Ok(v)
            };
            let info = KernelInfo {
                registers_per_thread: attr(4)?,
                local_bytes_per_thread: attr(3)?,
                shared_bytes_per_block: attr(1)?,
                max_threads_per_block: attr(0)? as u32,
            };
            if info.local_bytes_per_thread != 0 || info.shared_bytes_per_block != 0 {
                return Err(Error(format!("kernel is not register-only: {info:?}")));
            }
            let bytes = len
                .checked_mul(4)
                .ok_or_else(|| Error("output size overflow".into()))?;
            check(
                cuMemAlloc_v2(&mut self.output, bytes),
                "allocate checksum buffer",
            )?;
            self.len = len;
            Ok(info)
        }
    }
    pub fn upload_operands(&mut self, config: &super::Config) -> Result<()> {
        let values: Vec<f32> = (0..3)
            .flat_map(|case| {
                let (a, b, _, _) = super::values::input_values(config, case);
                [a, b]
            })
            .collect();
        // Loaded once per FP32 kernel, outside its arithmetic loop. Runtime
        // data prevents constant-folding FMA into addition when A/B equal one.
        unsafe {
            check(
                cuMemAlloc_v2(&mut self.operands, values.len() * 4),
                "allocate operands",
            )?;
            check(
                cuMemcpyHtoD_v2(self.operands, values.as_ptr().cast(), values.len() * 4),
                "upload operands",
            )?;
        }
        Ok(())
    }
    pub fn launch(&self, blocks: u32, threads: u32, iterations: u32, case: usize) -> Result<()> {
        let mut out = self.output;
        let mut iters = iterations;
        let mut operands = self.operands + case as u64 * 8;
        let function = *self
            .functions
            .get(case)
            .ok_or_else(|| Error("invalid kernel case".into()))?;
        let mut params: [Handle; 3] = [
            (&mut out as *mut u64).cast(),
            (&mut iters as *mut u32).cast(),
            (&mut operands as *mut u64).cast(),
        ];
        // SAFETY: matching kernel parameter ABI; driver copies parameters before
        // returning. The owned output has blocks*threads elements and stays live.
        unsafe {
            check(
                cuLaunchKernel(
                    function,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    null_mut(),
                ),
                "launch register arithmetic",
            )
        }
    }
    pub fn sync(&self) -> Result<()> {
        unsafe { check(cuStreamSynchronize(self.stream), "synchronize stream") }
    }
    pub fn begin(&self) -> Result<()> {
        unsafe { check(cuEventRecord(self.start, self.stream), "record start") }
    }
    pub fn end_ms(&self) -> Result<f64> {
        let mut ms = 0_f32;
        unsafe {
            check(cuEventRecord(self.end, self.stream), "record end")?;
            check(cuEventSynchronize(self.end), "wait for end")?;
            check(
                cuEventElapsedTime(&mut ms, self.start, self.end),
                "elapsed time",
            )?;
        }
        if !ms.is_finite() || ms <= 0.0 {
            return Err(Error("nonpositive or nonfinite device time".into()));
        }
        Ok(f64::from(ms))
    }
    pub fn download(&self) -> Result<Vec<f32>> {
        self.sync()?;
        let mut output = vec![0_f32; self.len];
        unsafe {
            check(
                cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), self.output, self.len * 4),
                "download checksums",
            )?;
        }
        Ok(output)
    }
}
/// Stops capture on both normal completion and an early benchmark error.
pub(super) struct ProfilerRange<'a> {
    driver: &'a Driver,
    active: bool,
}
impl ProfilerRange<'_> {
    pub fn stop(mut self) -> Result<()> {
        self.driver.sync()?;
        // SAFETY: the borrowed Driver keeps its CUDA context current and alive.
        unsafe {
            check(cuProfilerStop(), "stop profiler capture")?;
        }
        self.active = false;
        Ok(())
    }
}
impl Drop for ProfilerRange<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.driver.sync();
            // SAFETY: the guard is dropped before its borrowed Driver.
            unsafe {
                cuProfilerStop();
            }
        }
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cuStreamSynchronize(self.stream);
            }
            if self.output != 0 {
                cuMemFree_v2(self.output);
            }
            if self.operands != 0 {
                cuMemFree_v2(self.operands);
            }
            if !self.start.is_null() {
                cuEventDestroy_v2(self.start);
            }
            if !self.end.is_null() {
                cuEventDestroy_v2(self.end);
            }
            if !self.module.is_null() {
                cuModuleUnload(self.module);
            }
            if !self.stream.is_null() {
                cuStreamDestroy_v2(self.stream);
            }
            if self.pushed {
                let mut context = null_mut();
                cuCtxPopCurrent_v2(&mut context);
            }
            if self.retained {
                cuDevicePrimaryCtxRelease_v2(self.device);
            }
        }
    }
}
