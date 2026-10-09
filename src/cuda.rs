use crate::{ffi::*, input, *};
use std::{ffi::CStr, ptr::null_mut, time::Instant};

fn cuda(status: i32, op: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: CUDA returns a process-lifetime NUL-terminated error string.
    let detail = unsafe { CStr::from_ptr(cudaGetErrorString(status)) }.to_string_lossy();
    Err(Error(format!("{op}: CUDA {status}: {detail}")))
}
fn blas(status: i32, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error(format!("{op}: cuBLAS status {status}")))
    }
}

// All C handles and allocations stay private to this synchronous entry point.
// Every queued operation is synchronized before resource destruction.
struct Object {
    ptr: Handle,
    destroy: unsafe extern "C" fn(Handle) -> i32,
}
impl Object {
    fn new(
        create: impl FnOnce(*mut Handle) -> i32,
        destroy: unsafe extern "C" fn(Handle) -> i32,
        is_blas: bool,
        name: &str,
    ) -> Result<Self> {
        let mut ptr = null_mut();
        let status = create(&mut ptr);
        if is_blas {
            blas(status, name)?;
        } else {
            cuda(status, name)?;
        }
        Ok(Self { ptr, destroy })
    }
}
impl Drop for Object {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.ptr);
        }
    }
}
struct Buffer {
    ptr: Handle,
    len: usize,
}
impl Buffer {
    fn new(len: usize) -> Result<Self> {
        let mut ptr = null_mut();
        if len > 0 {
            unsafe {
                cuda(cudaMalloc(&mut ptr, len), "cudaMalloc")?;
            }
        }
        Ok(Self { ptr, len })
    }
    fn upload(bytes: &[u8]) -> Result<Self> {
        let buffer = Self::new(bytes.len())?;
        unsafe {
            cuda(
                cudaMemcpy(buffer.ptr, bytes.as_ptr().cast(), bytes.len(), 1),
                "upload",
            )?;
        }
        Ok(buffer)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}
struct DeviceGuard(i32);
impl Drop for DeviceGuard {
    fn drop(&mut self) {
        unsafe {
            cudaSetDevice(self.0);
        }
    }
}

struct Plan {
    precision: Precision,
    #[cfg(feature = "cutile")]
    tile: Option<crate::tile::Kernel>,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    workspace: Buffer,
    _a_scale: Buffer,
    _b_scale: Buffer,
    op: Object,
    ad: Object,
    bd: Object,
    cd: Object,
    lt: Object,
    stream: Object,
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            cudaStreamSynchronize(self.stream.ptr);
        }
    }
}
impl Plan {
    fn new(c: &BenchmarkConfig, a: &input::Operand, b: &input::Operand) -> Result<Self> {
        // SAFETY: descriptor APIs receive correctly sized scalar attributes and
        // valid buffers. Column-major TN views represent A[M,K] * B[N,K]^T.
        unsafe {
            let lt = Object::new(
                |p| cublasLtCreate(p),
                cublasLtDestroy,
                true,
                "cublasLtCreate",
            )?;
            let stream = Object::new(
                |p| cudaStreamCreateWithFlags(p, 1),
                cudaStreamDestroy,
                false,
                "stream create",
            )?;
            let a_device = Buffer::upload(&a.bytes)?;
            let b_device = Buffer::upload(&b.bytes)?;
            let upload_scales = |operand: &input::Operand, rows| -> Result<Buffer> {
                if c.backend == Backend::Cutile && c.precision == Precision::Nvfp4 {
                    let cols = c.shape.k / 16;
                    let scales: Vec<u8> = (0..rows)
                        .flat_map(|r| {
                            (0..cols).map(move |s| operand.scales[input::scale_offset(r, s, cols)])
                        })
                        .collect();
                    Buffer::upload(&scales)
                } else {
                    Buffer::upload(&operand.scales)
                }
            };
            let a_scale = upload_scales(a, c.shape.m)?;
            let b_scale = upload_scales(b, c.shape.n)?;
            let fp32 = c.precision.fp32_storage();
            let output = Buffer::new(c.shape.m * c.shape.n * if fp32 { 4 } else { 2 })?;
            let workspace = Buffer::new(c.workspace_bytes)?;
            let compute = match c.precision {
                Precision::Fp32 => CUBLAS_COMPUTE_32F_PEDANTIC,
                Precision::Tf32 => CUBLAS_COMPUTE_32F_FAST_TF32,
                _ => CUBLAS_COMPUTE_32F,
            };
            let op = Object::new(
                |p| cublasLtMatmulDescCreate(p, compute, 0),
                cublasLtMatmulDescDestroy,
                true,
                "matmul descriptor",
            )?;
            fn attr<T>(op: Handle, key: i32, value: &T) -> Result<()> {
                unsafe {
                    blas(
                        cublasLtMatmulDescSetAttribute(
                            op,
                            key,
                            (value as *const T).cast(),
                            size_of::<T>(),
                        ),
                        "matmul attribute",
                    )
                }
            }
            attr(op.ptr, 3, &1_i32)?; // TRANSA = T
            attr(op.ptr, 4, &0_i32)?; // TRANSB = N
            if matches!(c.precision, Precision::Fp8 | Precision::Nvfp4) {
                attr(op.ptr, 17, &a_scale.ptr)?;
                attr(op.ptr, 18, &b_scale.ptr)?;
                if c.precision == Precision::Nvfp4 {
                    attr(op.ptr, 31, &1_i32)?; // VEC16_UE4M3
                    attr(op.ptr, 32, &1_i32)?;
                }
            }
            let dtype = match c.precision {
                Precision::Fp32 | Precision::Tf32 => 0,
                Precision::Bf16 => 14,
                Precision::Fp8 => 28,
                Precision::Nvfp4 => 33,
            };
            let layout = |dtype, rows, cols, ld| {
                Object::new(
                    |p| cublasLtMatrixLayoutCreate(p, dtype, rows, cols, ld),
                    cublasLtMatrixLayoutDestroy,
                    true,
                    "matrix layout",
                )
            };
            let ad = layout(dtype, c.shape.k as u64, c.shape.m as u64, c.shape.k as i64)?;
            let bd = layout(dtype, c.shape.k as u64, c.shape.n as u64, c.shape.k as i64)?;
            let cd = layout(
                if fp32 { 0 } else { 14 },
                c.shape.m as u64,
                c.shape.n as u64,
                c.shape.m as i64,
            )?;
            #[cfg(feature = "cutile")]
            let tile = if c.backend == Backend::Cutile {
                Some(crate::tile::Kernel::new(
                    c.device,
                    c.precision,
                    stream.ptr,
                    c.shape,
                    a_device.ptr,
                    b_device.ptr,
                    a_scale.ptr,
                    b_scale.ptr,
                    output.ptr,
                )?)
            } else {
                None
            };
            Ok(Self {
                precision: c.precision,
                #[cfg(feature = "cutile")]
                tile,
                a: a_device,
                b: b_device,
                output,
                workspace,
                _a_scale: a_scale,
                _b_scale: b_scale,
                op,
                ad,
                bd,
                cd,
                lt,
                stream,
            })
        }
    }
    fn sync(&self) -> Result<()> {
        unsafe { cuda(cudaStreamSynchronize(self.stream.ptr), "stream synchronize") }
    }
    fn launch(&self, algo: &Algo) -> Result<()> {
        #[cfg(feature = "cutile")]
        if let Some(tile) = &self.tile {
            return tile.launch([
                algo.data[0] as usize,
                algo.data[1] as usize,
                algo.data[2] as usize,
            ]);
        }
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            blas(
                cublasLtMatmul(
                    self.lt.ptr,
                    self.op.ptr,
                    (&alpha as *const f32).cast(),
                    self.a.ptr,
                    self.ad.ptr,
                    self.b.ptr,
                    self.bd.ptr,
                    (&beta as *const f32).cast(),
                    self.output.ptr,
                    self.cd.ptr,
                    self.output.ptr,
                    self.cd.ptr,
                    algo,
                    self.workspace.ptr,
                    self.workspace.len,
                    self.stream.ptr,
                ),
                "cublasLtMatmul",
            )
        }
    }
    fn heuristics(&self, count: usize) -> Result<Vec<Heuristic>> {
        unsafe {
            let pref = Object::new(
                |p| cublasLtMatmulPreferenceCreate(p),
                cublasLtMatmulPreferenceDestroy,
                true,
                "preference",
            )?;
            blas(
                cublasLtMatmulPreferenceSetAttribute(
                    pref.ptr,
                    1,
                    (&self.workspace.len as *const usize).cast(),
                    size_of::<usize>(),
                ),
                "workspace preference",
            )?;
            let mut results = vec![Heuristic::default(); count];
            let mut returned = 0;
            blas(
                cublasLtMatmulAlgoGetHeuristic(
                    self.lt.ptr,
                    self.op.ptr,
                    self.ad.ptr,
                    self.bd.ptr,
                    self.cd.ptr,
                    self.cd.ptr,
                    pref.ptr,
                    count as i32,
                    results.as_mut_ptr(),
                    &mut returned,
                ),
                "algorithm heuristics",
            )?;
            results.truncate(returned as usize);
            if results.is_empty() {
                return Err(Error(
                    "cuBLASLt returned no algorithms for this precision/shape/device/workspace"
                        .into(),
                ));
            }
            Ok(results)
        }
    }
    fn validation(&self, expected: &[(usize, f64)]) -> Result<Validation> {
        self.sync()?;
        let mut bytes = vec![0_u8; self.output.len];
        unsafe {
            cuda(
                cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.output.ptr,
                    self.output.len,
                    2,
                ),
                "validation download",
            )?;
        }
        let fp32 = self.precision.fp32_storage();
        let output: Vec<f32> = if fp32 {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_ne_bytes(b.try_into().unwrap())) << 16))
                .collect()
        };
        let all_finite = output.iter().all(|x| x.is_finite());
        // TF32 is checked against the original, unrounded FP32 inputs. These
        // benchmark tolerances permit its input rounding error without relaxing
        // full-FP32 validation or quantizing away the accuracy comparison.
        let (relative_limit, rms_limit) = match self.precision {
            Precision::Fp32 => (1e-5, 5e-6),
            Precision::Tf32 => (1e-3, 2e-3),
            _ => (0.01, 0.005),
        };
        let norm = expected.iter().map(|(_, x)| x * x).sum::<f64>();
        let rms = (norm / expected.len() as f64).sqrt();
        let mut squared_error = 0.0;
        let mut max_scaled_error = 0.0_f64;
        for &(i, reference) in expected {
            let got = f64::from(output[i]);
            let error = (got - reference).abs();
            squared_error += error * error;
            max_scaled_error = max_scaled_error
                .max(error / (relative_limit * reference.abs() + rms_limit * rms).max(1e-12));
        }
        let relative_rmse = (squared_error / norm.max(1e-30)).sqrt();
        Ok(Validation {
            samples: expected.len(),
            all_outputs_finite: all_finite,
            relative_rmse,
            max_scaled_error,
            passed: all_finite && relative_rmse < relative_limit && max_scaled_error <= 1.0,
        })
    }
    fn poison(&self) -> Result<()> {
        unsafe {
            cuda(
                cudaMemsetAsync(self.output.ptr, 255, self.output.len, self.stream.ptr),
                "poison output",
            )
        }
    }
}

struct Timer {
    start: Object,
    end: Object,
}
impl Timer {
    fn new() -> Result<Self> {
        let event = || unsafe {
            Object::new(
                |p| cudaEventCreate(p),
                cudaEventDestroy,
                false,
                "event create",
            )
        };
        Ok(Self {
            start: event()?,
            end: event()?,
        })
    }
    fn measure(
        &self,
        plan: &Plan,
        algo: &Algo,
        count: usize,
        graph: Option<&Object>,
    ) -> Result<(f64, f64)> {
        let host = Instant::now();
        unsafe {
            cuda(
                cudaEventRecord(self.start.ptr, plan.stream.ptr),
                "start event",
            )?;
        }
        if let Some(graph) = graph {
            unsafe {
                cuda(cudaGraphLaunch(graph.ptr, plan.stream.ptr), "graph launch")?;
            }
        } else {
            for _ in 0..count {
                plan.launch(algo)?;
            }
        }
        unsafe {
            cuda(cudaEventRecord(self.end.ptr, plan.stream.ptr), "end event")?;
            cuda(cudaEventSynchronize(self.end.ptr), "event synchronize")?;
            let host_ms = host.elapsed().as_secs_f64() * 1000.0 / count as f64;
            let mut elapsed = 0.0;
            cuda(
                cudaEventElapsedTime(&mut elapsed, self.start.ptr, self.end.ptr),
                "event elapsed",
            )?;
            if !elapsed.is_finite() || elapsed <= 0.0 {
                return Err(Error("invalid CUDA event elapsed time".into()));
            }
            Ok((elapsed as f64 / count as f64, host_ms))
        }
    }
}
fn graph(plan: &Plan, algo: &Algo, count: usize) -> Result<Object> {
    unsafe {
        cuda(
            cudaStreamBeginCapture(plan.stream.ptr, 0),
            "begin graph capture",
        )?;
        let launches = (0..count).try_for_each(|_| plan.launch(algo));
        let mut raw = null_mut();
        let end_status = cudaStreamEndCapture(plan.stream.ptr, &mut raw);
        let graph = (!raw.is_null()).then_some(Object {
            ptr: raw,
            destroy: cudaGraphDestroy,
        });
        launches?;
        cuda(end_status, "end graph capture")?;
        let graph = graph.ok_or_else(|| Error("capture produced no graph".into()))?;
        Object::new(
            |p| cudaGraphInstantiateWithFlags(p, graph.ptr, 0),
            cudaGraphExecDestroy,
            false,
            "graph instantiate",
        )
    }
}

fn device_info(device: i32) -> Result<DeviceInfo> {
    unsafe {
        let attr = |key| -> Result<i32> {
            let mut v = 0;
            cuda(
                cudaDeviceGetAttribute(&mut v, key, device),
                "device attribute",
            )?;
            Ok(v)
        };
        let mut bus = [0 as std::ffi::c_char; 32];
        cuda(
            cudaDeviceGetPCIBusId(bus.as_mut_ptr(), 32, device),
            "PCI bus id",
        )?;
        let mut runtime = 0;
        let mut driver = 0;
        cuda(cudaRuntimeGetVersion(&mut runtime), "runtime version")?;
        cuda(cudaDriverGetVersion(&mut driver), "driver API version")?;
        let loaded_libraries = std::fs::read_to_string("/proc/self/maps")
            .unwrap_or_default()
            .lines()
            .filter_map(|line| line.split_whitespace().last())
            .filter(|s| {
                s.contains("libcublasLt.so")
                    || s.contains("libcudart.so")
                    || s.contains("libcuda.so")
            })
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(DeviceInfo {
            ordinal: device,
            pci_bus_id: CStr::from_ptr(bus.as_ptr()).to_string_lossy().into_owned(),
            compute_capability: format!("{}.{}", attr(75)?, attr(76)?),
            multiprocessors: attr(16)?,
            l2_bytes: attr(38)?,
            cuda_runtime_version: runtime,
            cuda_driver_api_version: driver,
            cublas_version: cublasLtGetVersion(),
            loaded_libraries,
        })
    }
}

struct ProfilerRange<'a> {
    plan: &'a Plan,
    active: bool,
}
impl<'a> ProfilerRange<'a> {
    fn start(plan: &'a Plan) -> Result<Self> {
        // SAFETY: the plan's CUDA device is current and warmup has synchronized.
        unsafe {
            cuda(cudaProfilerStart(), "start profiler capture")?;
        }
        Ok(Self { plan, active: true })
    }
    fn stop(mut self) -> Result<()> {
        self.plan.sync()?;
        // SAFETY: the borrowed plan and current device outlive the capture.
        unsafe {
            cuda(cudaProfilerStop(), "stop profiler capture")?;
        }
        self.active = false;
        Ok(())
    }
}
impl Drop for ProfilerRange<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.plan.sync();
            // SAFETY: the plan and device guard are still alive on this thread.
            unsafe {
                cudaProfilerStop();
            }
        }
    }
}

pub(crate) fn run(c: &BenchmarkConfig, profiler_range: bool) -> Result<BenchmarkResult> {
    let started = unix_ms();
    let mut original = 0;
    unsafe {
        cuda(cudaGetDevice(&mut original), "get current device")?;
    }
    let _guard = DeviceGuard(original);
    unsafe {
        cuda(cudaSetDevice(c.device), "set device")?;
    }
    let device = device_info(c.device)?;
    let telemetry_before = telemetry(&device.pci_bus_id);
    let a = input::generate(c.shape.m, c.shape.k, c.precision, c.seed);
    let b = input::generate(
        c.shape.n,
        c.shape.k,
        c.precision,
        c.seed ^ 0xa5a5_a5a5_5a5a_5a5a,
    );
    let expected = input::reference(c.shape, &a, &b, c.validation_samples, c.seed);
    let plan = Plan::new(c, &a, &b)?;
    drop(a);
    drop(b);
    let heuristics = if c.backend == Backend::Cutile {
        // BM indexes N and BN indexes M because output is column-major.
        // FP32 lowers to scalar arithmetic on GB10. Tensor-sized tiles cause
        // excessive register pressure and local-memory spills for that path.
        let tiles = if c.precision == Precision::Fp32 {
            [
                [16, 16, 8],
                [16, 32, 8],
                [32, 16, 8],
                [32, 32, 8],
                [16, 16, 16],
                [16, 32, 16],
                [32, 16, 16],
                [32, 32, 16],
            ]
        } else {
            [
                [64, 64, 128],
                [128, 64, 128],
                [64, 128, 128],
                [128, 128, 128],
                [64, 64, 256],
                [128, 64, 256],
                [64, 128, 256],
                [128, 128, 256],
            ]
        };
        tiles
            .into_iter()
            .take(c.candidates)
            .map(|[m, n, k]| Heuristic {
                algo: Algo {
                    data: [m, n, k, 0, 0, 0, 0, 0],
                },
                ..Default::default()
            })
            .collect()
    } else {
        plan.heuristics(c.candidates)?
    };
    let timer = Timer::new()?;
    let mut candidates = Vec::new();
    let mut selected = None;
    let mut best_ms = f64::INFINITY;
    for (index, h) in heuristics.iter().enumerate() {
        let mut algorithm_id = -1;
        let mut written = 0;
        if c.backend == Backend::Cublaslt {
            unsafe {
                blas(
                    cublasLtMatmulAlgoConfigGetAttribute(
                        &h.algo,
                        0,
                        (&mut algorithm_id as *mut i32).cast(),
                        size_of::<i32>(),
                        &mut written,
                    ),
                    "algorithm ID",
                )?;
            }
        }
        let mut candidate = Candidate {
            index,
            algorithm_id,
            opaque_config: h.algo.data,
            workspace_bytes: h.workspace_size,
            numerical_impl_flags: None,
            validation: None,
            tuning_ms_per_gemm: vec![],
            error: None,
        };
        let attempt = (|| -> Result<f64> {
            blas(h.state, "heuristic state")?;
            if c.backend == Backend::Cublaslt {
                let mut flags = 0_u64;
                let mut bytes = 0;
                // SAFETY: the capability has uint64_t type in cublasLt.h.
                unsafe {
                    blas(
                        cublasLtMatmulAlgoCapGetAttribute(
                            &h.algo,
                            CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                            (&mut flags as *mut u64).cast(),
                            size_of::<u64>(),
                            &mut bytes,
                        ),
                        "numerical implementation flags",
                    )?;
                }
                if bytes != size_of::<u64>() {
                    return Err(Error(
                        "unexpected numerical implementation flags size".into(),
                    ));
                }
                candidate.numerical_impl_flags = Some(flags);
                if c.precision == Precision::Tf32
                    && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK == 0
                        || flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 == 0
                        || flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F == 0)
                {
                    return Err(Error(format!(
                        "candidate is not TF32 Tensor Core arithmetic with FP32 accumulation (flags {flags:#x})"
                    )));
                }
            }
            plan.poison()?;
            plan.launch(&h.algo)?;
            let validation = plan.validation(&expected)?;
            let passed = validation.passed;
            candidate.validation = Some(validation);
            if !passed {
                return Err(Error("candidate failed numerical validation".into()));
            }
            if c.selection == Selection::Heuristic {
                return Ok(0.0);
            }
            for _ in 0..c.tune_warmup {
                plan.launch(&h.algo)?;
            }
            plan.sync()?;
            let graph = if c.timing == Timing::Graph {
                Some(graph(&plan, &h.algo, c.tune_iterations)?)
            } else {
                None
            };
            if let Some(g) = &graph {
                unsafe {
                    cuda(
                        cudaGraphLaunch(g.ptr, plan.stream.ptr),
                        "tuning graph warmup",
                    )?;
                }
                plan.sync()?;
            }
            for _ in 0..c.tune_trials {
                candidate.tuning_ms_per_gemm.push(
                    timer
                        .measure(&plan, &h.algo, c.tune_iterations, graph.as_ref())?
                        .0,
                );
            }
            let mut sorted = candidate.tuning_ms_per_gemm.clone();
            sorted.sort_by(f64::total_cmp);
            Ok(sorted[sorted.len() / 2])
        })();
        match attempt {
            Ok(ms) if ms < best_ms => {
                best_ms = ms;
                selected = Some(index);
            }
            Ok(_) => {}
            Err(e) => {
                candidate.error = Some(e.to_string());
                plan.sync()?;
            }
        }
        candidates.push(candidate);
        if c.selection == Selection::Heuristic && selected.is_some() {
            break;
        }
    }
    let selected = selected.ok_or_else(|| {
        Error(format!(
            "no validated algorithm: {}",
            serde_json::to_string(&candidates).unwrap_or_default()
        ))
    })?;
    let algo = &heuristics[selected].algo;
    for _ in 0..c.warmup {
        plan.launch(algo)?;
    }
    plan.sync()?;
    let graph = if c.timing == Timing::Graph {
        Some(graph(&plan, algo, c.iterations)?)
    } else {
        None
    };
    if let Some(g) = &graph {
        unsafe {
            cuda(
                cudaGraphLaunch(g.ptr, plan.stream.ptr),
                "measurement graph warmup",
            )?;
        }
        plan.sync()?;
    }
    let flops = 2.0 * c.shape.m as f64 * c.shape.n as f64 * c.shape.k as f64;
    let mut trials = Vec::new();
    let capture = if profiler_range {
        Some(ProfilerRange::start(&plan)?)
    } else {
        None
    };
    for _ in 0..c.trials {
        let (gpu, host) = timer.measure(&plan, algo, c.iterations, graph.as_ref())?;
        trials.push(Trial {
            gpu_ms_per_gemm: gpu,
            host_ms_per_gemm: host,
            tflops: flops / gpu / 1e9,
            telemetry_after: telemetry(&device.pci_bus_id),
        });
    }
    if let Some(capture) = capture {
        capture.stop()?;
    }
    let validation_after = plan.validation(&expected)?;
    if !validation_after.passed {
        return Err(Error(format!(
            "post-measurement validation failed: {validation_after:?}"
        )));
    }
    let tflops = Statistics::of(&trials.iter().map(|t| t.tflops).collect::<Vec<_>>());
    Ok(BenchmarkResult {
        schema_version: 3,
        output_precision: if c.precision.fp32_storage() {
            "fp32"
        } else {
            "bf16"
        }
        .into(),
        compute_mode: match (c.backend, c.precision) {
            (Backend::Cublaslt, Precision::Fp32) => "CUBLAS_COMPUTE_32F_PEDANTIC",
            (Backend::Cublaslt, Precision::Tf32) => "CUBLAS_COMPUTE_32F_FAST_TF32",
            (Backend::Cublaslt, _) => "CUBLAS_COMPUTE_32F",
            (Backend::Cutile, Precision::Nvfp4) => "mmaf_scaled; FP32 accumulation",
            (Backend::Cutile, _) => "mmaf; FP32 accumulation; native input type",
        }
        .into(),
        backend: match c.backend {
            Backend::Cublaslt => "cublaslt",
            Backend::Cutile => "cutile",
        }
        .into(),
        config: c.clone(),
        profiler_range,
        started_unix_ms: started,
        finished_unix_ms: unix_ms(),
        device,
        telemetry_before,
        timing_scope: format!(
            "CUDA events around repeated dense GEMMs; {} output, FP32 accumulation{}; same allocations and operands reused, no explicit L2 flush; excludes encoding, transfers, allocation, heuristic search, tuning, graph construction and validation. Host timing includes submission and event synchronization. Graph mode has one additional untimed graph replay.",
            if c.precision.fp32_storage() {
                "FP32"
            } else {
                "BF16"
            },
            if c.precision == Precision::Fp32 && c.backend == Backend::Cublaslt {
                " (cuBLASLt PEDANTIC, no TF32)"
            } else if c.precision == Precision::Tf32 {
                " (cuBLASLt TF32 Tensor Core multiplication)"
            } else {
                ""
            }
        ),
        input_description: format!(
            "Deterministic signed, directly encoded finite inputs (generator v1). {}; global alpha=1, beta=0. Validation uses FP64 CPU sums of the decoded inputs, checks every output is finite and samples corners, tile boundaries and distributed coordinates. This measures GEMM, not quantization quality.",
            match c.precision {
                Precision::Fp32 =>
                    "FP32 uniform signed inputs without reduced-precision rounding; validation relative RMSE < 1e-5, scaled error <= 1 using 1e-5*abs(reference)+5e-6*reference_RMS",
                Precision::Tf32 =>
                    "FP32 uniform signed inputs identical to the full-FP32 mode; cuBLASLt rounds internally for TF32 multiplication; validation against original FP32 inputs: relative RMSE < 1e-3, scaled error <= 1 using 1e-3*abs(reference)+2e-3*reference_RMS",
                Precision::Bf16 => "BF16 RNE-encoded signed inputs",
                Precision::Fp8 => "FP8 E4M3 finite signed inputs, unit scalar scales",
                Precision::Nvfp4 if c.backend == Backend::Cutile =>
                    "NVFP4 E2M1 with varying E4M3 block-16 scales in row-major layout",
                Precision::Nvfp4 => "NVFP4 E2M1 with varying E4M3 block-16 scales in 128x4 layout",
            }
        ),
        useful_flops_per_gemm: flops,
        selected_candidate: selected,
        candidates,
        validation_after,
        trials,
        tflops,
    })
}
