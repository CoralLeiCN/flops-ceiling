//! Register-resident tensor and FP32 FMA throughput on DGX Spark (SM121).
//!
//! Rust generates PTX and launches it through the CUDA Driver API. This is an
//! instruction microbenchmark, not a full GEMM or a cuTile Rust kernel. The hot
//! loop has no operand-memory traffic; a checksum store per thread is timed.
//!
//! ```no_run
//! use flops_ceiling::registers::{benchmark, Config, ScaleFormat, Sparsity};
//! let result = benchmark(&Config {
//!     format: ScaleFormat::Nvfp4,
//!     sparsity: Sparsity::Dense,
//!     ..Default::default()
//! })?;
//! println!("mean {} / best {} TFLOPS", result.tflops.mean, result.tflops.max);
//! # Ok::<(), flops_ceiling::Error>(())
//! ```
mod driver;
mod ptx;
mod values;

use crate::{Error, Result, Statistics, Telemetry, telemetry, unix_ms};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    /// Full-precision scalar FP32 FMA on CUDA cores; dense only.
    Fp32,
    /// Unscaled BF16 Tensor Core MMA, FP32 accumulation.
    Bf16,
    /// Unscaled FP8 E4M3 Tensor Core MMA, FP32 accumulation.
    Fp8,
    /// E2M1 data, UE4M3 scale per 16 elements, scale_vec::4X.
    Nvfp4,
    /// E2M1 data, UE8M0 scale per 32 elements, scale_vec::2X.
    /// Matches the scale format in the community register-only benchmark.
    Mxfp4,
}
/// Compatibility name retained for existing FP4 callers.
pub type ScaleFormat = Format;

impl Format {
    pub(crate) fn block_scaled(self) -> bool {
        matches!(self, Self::Nvfp4 | Self::Mxfp4)
    }
    pub(crate) fn dense_k(self) -> u32 {
        match self {
            Self::Fp32 => 1,
            Self::Bf16 => 16,
            Self::Fp8 => 32,
            Self::Nvfp4 | Self::Mxfp4 => 64,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Sparsity {
    Dense,
    /// Structured 2:4 sparse A. TFLOPS counts dense-equivalent work, including
    /// implicit zeros; executed nonzero arithmetic is half this reported count.
    TwoOfFour,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub device: i32,
    pub format: ScaleFormat,
    pub sparsity: Sparsity,
    /// Uniform finite values exactly representable in the selected format.
    #[serde(default = "one")]
    pub a_value: f32,
    #[serde(default = "one")]
    pub b_value: f32,
    /// Positive UE4M3 / UE8M0 block scales for FP4; must be 1 for other formats.
    #[serde(default = "one")]
    pub scale_a: f32,
    #[serde(default = "one")]
    pub scale_b: f32,
    /// Scalar accumulator j starts at start + j*step. Values must be distinct.
    #[serde(default)]
    pub accumulator_start: f32,
    #[serde(default = "one")]
    pub accumulator_step: f32,
    /// Independent MMA chains per warp, or scalar FMA chains per FP32 thread (1..=32).
    pub accumulators: u32,
    pub threads_per_block: u32,
    /// Grid size is this multiplier times the physical SM count; not residency.
    pub blocks_per_sm: u32,
    /// Loop iterations inside each GPU launch (1..=65536).
    pub iterations: u32,
    pub warmup_launches: u32,
    pub launches_per_trial: u32,
    pub trials: u32,
}
fn one() -> f32 {
    1.0
}
impl Default for Config {
    fn default() -> Self {
        Self {
            device: 0,
            format: ScaleFormat::Nvfp4,
            sparsity: Sparsity::Dense,
            a_value: 1.0,
            b_value: 1.0,
            scale_a: 1.0,
            scale_b: 1.0,
            accumulator_start: 0.0,
            accumulator_step: 1.0,
            accumulators: 16,
            threads_per_block: 128,
            blocks_per_sm: 8,
            iterations: 700,
            warmup_launches: 10,
            launches_per_trial: 1,
            trials: 10,
        }
    }
}
impl Config {
    pub(crate) fn scalar_accumulators(&self) -> u32 {
        self.accumulators * if self.format == Format::Fp32 { 1 } else { 4 }
    }

    pub fn validate(&self) -> Result<()> {
        if self.format == Format::Fp32 && self.sparsity != Sparsity::Dense {
            return Err(Error(
                "FP32 uses scalar CUDA-core FMA and supports dense mode only".into(),
            ));
        }
        if !self.format.block_scaled() && (self.scale_a != 1.0 || self.scale_b != 1.0) {
            return Err(Error(
                "BF16, FP8 and FP32 instructions are unscaled; scale_a and scale_b must be 1"
                    .into(),
            ));
        }
        if self.device < 0
            || !(1..=32).contains(&self.accumulators)
            || self.threads_per_block == 0
            || self.threads_per_block > 1024
            || self.threads_per_block % 32 != 0
            || self.blocks_per_sm == 0
            || !(1..=65536).contains(&self.iterations)
            || self.warmup_launches == 0
            || self.launches_per_trial == 0
            || self.trials == 0
        {
            return Err(Error("device >= 0; accumulators 1..=32; threads a multiple of 32 <= 1024; iterations 1..=65536; all counts positive".into()));
        }
        values::validate(self)
    }

    /// Compute the CPU FP32 reference checksum without CUDA.
    /// MMA rounding is architecture-dependent; benchmark rejects a GPU mismatch.
    pub fn expected_checksum(&self) -> Result<f32> {
        self.validate()?;
        values::checksum(self, self.iterations, 0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub pci_bus_id: String,
    pub compute_capability: String,
    pub multiprocessors: u32,
    pub l2_bytes: i32,
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
    pub driver_api_version: i32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct KernelInfo {
    pub registers_per_thread: i32,
    pub local_bytes_per_thread: i32,
    pub shared_bytes_per_block: i32,
    pub max_threads_per_block: u32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ChecksumValidation {
    pub case: String,
    pub iterations: u32,
    pub checked_threads: usize,
    pub expected_checksum: f32,
    /// None when reading older records that did not retain observed extrema.
    #[serde(default)]
    pub observed_min: Option<f32>,
    #[serde(default)]
    pub observed_max: Option<f32>,
    pub max_absolute_error: f32,
    pub passed: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Trial {
    pub gpu_ms: f64,
    pub gpu_ms_per_launch: f64,
    pub tflops: f64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub schema_version: u32,
    pub config: Config,
    /// A CUDA profiler range was requested. This does not detect attached tools.
    /// Treat timings collected under a profiler as diagnostics, not baseline rates.
    #[serde(default)]
    pub profiler_range: bool,
    pub started_unix_ms: u128,
    pub finished_unix_ms: u128,
    pub device: DeviceInfo,
    pub kernel: KernelInfo,
    pub grid_blocks: u32,
    pub warps_per_launch: u64,
    /// Legacy MMA field; zero for scalar FP32. Use flops_per_warp_instruction.
    pub flops_per_warp_mma: u32,
    #[serde(default)]
    pub flops_per_warp_instruction: u32,
    #[serde(default)]
    pub instruction: String,
    pub flops_per_launch: f64,
    pub flop_counting: String,
    pub timing_scope: String,
    pub telemetry_before: Telemetry,
    pub telemetry_after: Telemetry,
    pub validation: Vec<ChecksumValidation>,
    pub trials: Vec<Trial>,
    pub tflops: Statistics,
    /// Exact input to the CUDA driver JIT; useful for external SASS inspection.
    pub ptx: String,
}

/// Generate the same PTX used by [`benchmark`], without initializing CUDA.
pub fn kernel_ptx(config: &Config) -> Result<String> {
    config.validate()?;
    ptx::generate(config)
}

fn validate_checksum(
    driver: &driver::Driver,
    c: &Config,
    iterations: u32,
    case: u32,
    name: &str,
) -> Result<ChecksumValidation> {
    let expected = values::checksum(c, iterations, case)?;
    let output = driver.download()?;
    let mut max_error = 0_f32;
    for (i, actual) in output.iter().enumerate() {
        if !actual.is_finite() || *actual != expected {
            return Err(Error(format!(
                "checksum {name} failed at thread {i}: {actual}, expected {expected}; \
                 no throughput result accepted. MMA rounding can differ from the strict \
                 CPU FP32 reference when scales or initial values have very different magnitudes"
            )));
        }
        max_error = max_error.max((*actual - expected).abs());
    }
    Ok(ChecksumValidation {
        case: name.into(),
        iterations,
        checked_threads: output.len(),
        expected_checksum: expected,
        observed_min: output.iter().copied().reduce(f32::min),
        observed_max: output.iter().copied().reduce(f32::max),
        max_absolute_error: max_error,
        passed: true,
    })
}

/// Run synchronously on the calling thread, restoring its CUDA context.
///
/// Compilation, allocation, validation and warmups precede timing. Each event
/// interval includes the configured launches, loop control and checksum stores.
/// Serialize calls to avoid GPU contention. Sparse TFLOPS is dense-equivalent.
/// Validates uniform-input checksum probes, not arbitrary GEMM numerical accuracy.
pub fn benchmark(config: &Config) -> Result<BenchmarkResult> {
    run(config, false)
}

/// Mark the measurement phase with cuProfilerStart/Stop for external Nsight tools.
///
/// Launch under ncu with --profile-from-start off, or under nsys with
/// --capture-range=cudaProfilerApi. Excludes setup, probes, warmup and validation.
/// This function does not launch a profiler. Profiling can perturb GPU timings;
/// use [`benchmark`] without attached tools for baseline throughput.
pub fn benchmark_with_profiler_range(config: &Config) -> Result<BenchmarkResult> {
    run(config, true)
}

fn run(config: &Config, profiler_range: bool) -> Result<BenchmarkResult> {
    config.validate()?;
    let started = unix_ms();
    let ptx = kernel_ptx(config)?;
    let (mut driver, device) = driver::Driver::new(config.device)?;
    let blocks = device
        .multiprocessors
        .checked_mul(config.blocks_per_sm)
        .filter(|b| *b <= device.max_grid_x)
        .ok_or_else(|| Error("grid size exceeds device limit".into()))?;
    let threads = blocks
        .checked_mul(config.threads_per_block)
        .ok_or_else(|| Error("total thread count exceeds u32 indexing".into()))?;
    if config.threads_per_block > device.max_threads_per_block {
        return Err(Error("threads per block exceed device limit".into()));
    }
    let kernel = driver.prepare(&ptx, threads as usize)?;
    if config.format == Format::Fp32 {
        driver.upload_operands(config)?;
    }
    if config.threads_per_block > kernel.max_threads_per_block {
        return Err(Error(
            "threads per block exceed this kernel's resource limit".into(),
        ));
    }
    let mut validation = Vec::new();
    for (case, name) in [
        (0, "timed_inputs"),
        (
            1,
            if config.format.block_scaled() {
                "scale_two"
            } else {
                "input_two"
            },
        ),
        (2, "negative_times_two"),
    ] {
        driver.launch(blocks, config.threads_per_block, 3, case as usize)?;
        validation.push(validate_checksum(&driver, config, 3, case, name)?);
    }
    let telemetry_before = telemetry(&device.pci_bus_id);
    for _ in 0..config.warmup_launches {
        driver.launch(blocks, config.threads_per_block, config.iterations, 0)?;
    }
    driver.sync()?;
    let warps = u64::from(blocks) * u64::from(config.threads_per_block / 32);
    let scalar = config.format == Format::Fp32;
    let k = config.format.dense_k()
        * if config.sparsity == Sparsity::Dense {
            1
        } else {
            2
        };
    let flops_per_warp_mma = if scalar { 0 } else { 2 * 16 * 8 * k };
    let flops_per_warp_instruction = if scalar { 2 * 32 } else { flops_per_warp_mma };
    let flops = warps as f64
        * f64::from(config.iterations)
        * f64::from(config.accumulators)
        * f64::from(flops_per_warp_instruction);
    let mut trials = Vec::new();
    let capture = if profiler_range {
        Some(driver.profiler_range()?)
    } else {
        None
    };
    for _ in 0..config.trials {
        driver.begin()?;
        for _ in 0..config.launches_per_trial {
            driver.launch(blocks, config.threads_per_block, config.iterations, 0)?;
        }
        let ms = driver.end_ms()?;
        let per_launch = ms / f64::from(config.launches_per_trial);
        trials.push(Trial {
            gpu_ms: ms,
            gpu_ms_per_launch: per_launch,
            tflops: flops / (per_launch * 1e9),
        });
    }
    if let Some(capture) = capture {
        capture.stop()?;
    }
    validation.push(validate_checksum(
        &driver,
        config,
        config.iterations,
        0,
        "after_timing",
    )?);
    let telemetry_after = telemetry(&device.pci_bus_id);
    let tflops = Statistics::of(&trials.iter().map(|t| t.tflops).collect::<Vec<_>>());
    Ok(BenchmarkResult { schema_version: 3, config: config.clone(), profiler_range, started_unix_ms: started,
        finished_unix_ms: unix_ms(), device, kernel, grid_blocks: blocks, warps_per_launch: warps,
        flops_per_warp_mma, flops_per_warp_instruction, flops_per_launch: flops,
        instruction: if scalar { "fma.rn.f32 (CUDA cores)" } else { "mma.sync (Tensor Cores)" }.into(),
        flop_counting: if scalar {
            "2 FLOPs per thread FMA; 32 threads per warp; repeated full FP32 arithmetic".into()
        } else if config.sparsity == Sparsity::Dense {
            format!("2*16*8*{k} per warp MMA; repeated arithmetic on reused register operands")
        } else {
            format!("2*16*8*{k} dense-equivalent per warp MMA; includes implicit zeros; nonzero arithmetic is half")
        },
        timing_scope: "CUDA events around launches; includes parameter setup, register initialization (FP32 loads A/B once from device memory), arithmetic loop, checksum reduction and one global store per thread; excludes JIT, allocation, host/device transfers, warmup, validation and telemetry; no operand loads in arithmetic loop".into(),
        telemetry_before, telemetry_after, validation, trials, tflops, ptx })
}
