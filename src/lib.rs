//! Measure dense GEMM throughput with owned CUDA resources and validated outputs.
//!
//! ```no_run
//! use flops_ceiling::{BenchmarkConfig, Precision, Shape, benchmark};
//! let config = BenchmarkConfig {
//!     shape: Shape { m: 4096, n: 4096, k: 4096 },
//!     precision: Precision::Nvfp4,
//!     ..Default::default()
//! };
//! let result = benchmark(&config)?;
//! println!("{} TFLOPS", result.tflops.mean);
//! # Ok::<(), flops_ceiling::Error>(())
//! ```
mod cuda;
mod ffi;
mod input;
pub mod registers;
#[cfg(feature = "cutile")]
mod tile;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    process::Command,
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// Full IEEE FP32 inputs, accumulation and output; no TF32 substitution.
    Fp32,
    /// FP32 storage with TF32 Tensor Core multiplication and FP32 accumulation.
    /// Supported by cuBLASLt only.
    Tf32,
    Bf16,
    Fp8,
    Nvfp4,
}

impl Precision {
    pub(crate) fn fp32_storage(self) -> bool {
        matches!(self, Self::Fp32 | Self::Tf32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Timing {
    Stream,
    Graph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    Heuristic,
    Tune,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cublaslt,
    Cutile,
}
impl Default for Backend {
    fn default() -> Self {
        Self::Cublaslt
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}
impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}", self.m, self.n, self.k)
    }
}
impl FromStr for Shape {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let dims: std::result::Result<Vec<usize>, _> = s.split('x').map(str::parse).collect();
        match dims {
            Ok(d) if d.len() == 3 => Ok(Self {
                m: d[0],
                n: d[1],
                k: d[2],
            }),
            _ => Err(Error(
                "shape must be MxNxK, for example 4096x4096x4096".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    #[serde(default)]
    pub backend: Backend,
    pub device: i32,
    pub shape: Shape,
    pub precision: Precision,
    pub timing: Timing,
    pub selection: Selection,
    pub workspace_bytes: usize,
    pub candidates: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub trials: usize,
    pub tune_warmup: usize,
    pub tune_iterations: usize,
    pub tune_trials: usize,
    pub validation_samples: usize,
    pub seed: u64,
}
impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Cublaslt,
            device: 0,
            shape: Shape {
                m: 4096,
                n: 4096,
                k: 4096,
            },
            precision: Precision::Nvfp4,
            timing: Timing::Graph,
            selection: Selection::Tune,
            workspace_bytes: 64 * 1024 * 1024,
            candidates: 32,
            warmup: 10,
            iterations: 100,
            trials: 5,
            tune_warmup: 5,
            tune_iterations: 20,
            tune_trials: 3,
            validation_samples: 256,
            seed: 42,
        }
    }
}
impl BenchmarkConfig {
    pub fn validate(&self) -> Result<()> {
        if self.precision == Precision::Tf32 && self.backend != Backend::Cublaslt {
            return Err(Error(
                "TF32 is supported only by the cuBLASLt backend".into(),
            ));
        }
        if self.backend == Backend::Cutile {
            if !cfg!(feature = "cutile") {
                return Err(Error(
                    "rebuild with --features cutile to enable the native Rust GEMM backend".into(),
                ));
            }
        }
        let s = self.shape;
        if s.m == 0 || s.n == 0 || s.k == 0 || s.m % 16 != 0 || s.n % 16 != 0 || s.k % 32 != 0 {
            return Err(Error(
                "M and N must be positive multiples of 16; K must be a positive multiple of 32"
                    .into(),
            ));
        }
        for (a, b) in [(s.m, s.n), (s.m, s.k), (s.n, s.k)] {
            if a > i32::MAX as usize
                || b > i32::MAX as usize
                || a.checked_mul(b)
                    .and_then(|v| v.checked_mul(4))
                    .is_none_or(|v| v > isize::MAX as usize)
            {
                return Err(Error(
                    "matrix dimensions overflow supported allocation or BLAS limits".into(),
                ));
            }
        }
        if self.device < 0
            || self.candidates == 0
            || self.candidates > 1024
            || self.warmup == 0
            || self.iterations == 0
            || self.trials == 0
            || self.tune_warmup == 0
            || self.tune_iterations == 0
            || self.tune_trials == 0
            || self.validation_samples < 64
        {
            return Err(Error("counts must be positive, candidates <= 1024, device >= 0, validation_samples >= 64".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: f64,
    pub sample_stddev: Option<f64>,
    pub min: f64,
    pub max: f64,
}
impl Statistics {
    pub(crate) fn of(values: &[f64]) -> Self {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Self {
            mean,
            sample_stddev: (values.len() > 1).then(|| {
                (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64)
                    .sqrt()
            }),
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Validation {
    pub samples: usize,
    pub all_outputs_finite: bool,
    pub relative_rmse: f64,
    pub max_scaled_error: f64,
    pub passed: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub index: usize,
    /// cuBLAS algorithm ID, or -1 for the cuTile backend.
    pub algorithm_id: i32,
    /// cuBLAS version-specific algorithm bytes as u64 words. For cuTile, the
    /// first three words are [tile_N, tile_M, tile_K]; remaining words are zero.
    pub opaque_config: [u64; 8],
    pub workspace_bytes: usize,
    /// cuBLASLt numerical implementation flags; absent for cuTile/older records.
    #[serde(default)]
    pub numerical_impl_flags: Option<u64>,
    pub validation: Option<Validation>,
    pub tuning_ms_per_gemm: Vec<f64>,
    pub error: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Telemetry {
    pub unix_ms: u128,
    pub csv: Option<String>,
    pub unavailable_reason: Option<String>,
}
pub(crate) fn telemetry(pci_bus: &str) -> Telemetry {
    let output = Command::new("nvidia-smi").args(["-i",pci_bus,"--query-gpu=name,uuid,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem","--format=csv,noheader"]).output();
    let (csv, reason) = match output {
        Ok(o) if o.status.success() => (
            Some(String::from_utf8_lossy(&o.stdout).trim().to_owned()),
            None,
        ),
        Ok(o) => (
            None,
            Some(String::from_utf8_lossy(&o.stderr).trim().to_owned()),
        ),
        Err(e) => (None, Some(e.to_string())),
    };
    Telemetry {
        unix_ms: unix_ms(),
        csv,
        unavailable_reason: reason,
    }
}
pub fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub pci_bus_id: String,
    pub compute_capability: String,
    pub multiprocessors: i32,
    pub l2_bytes: i32,
    pub cuda_runtime_version: i32,
    pub cuda_driver_api_version: i32,
    pub cublas_version: usize,
    pub loaded_libraries: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Trial {
    pub gpu_ms_per_gemm: f64,
    pub host_ms_per_gemm: f64,
    pub tflops: f64,
    pub telemetry_after: Telemetry,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub schema_version: u32,
    #[serde(default)]
    pub output_precision: String,
    #[serde(default)]
    pub compute_mode: String,
    pub backend: String,
    pub config: BenchmarkConfig,
    /// Measurement capture was requested; this does not detect attached tools.
    #[serde(default)]
    pub profiler_range: bool,
    pub started_unix_ms: u128,
    pub finished_unix_ms: u128,
    pub device: DeviceInfo,
    pub telemetry_before: Telemetry,
    pub timing_scope: String,
    pub input_description: String,
    pub useful_flops_per_gemm: f64,
    pub selected_candidate: usize,
    pub candidates: Vec<Candidate>,
    pub validation_after: Validation,
    pub trials: Vec<Trial>,
    pub tflops: Statistics,
}

/// Run a benchmark on the calling thread. Restores the caller's current device.
///
/// Each call owns its stream, descriptors, allocations and graph. Setup, input
/// encoding, transfers, tuning and validation are excluded from reported timing.
/// Serialize benchmark calls to avoid measuring contention between runs.
pub fn benchmark(config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    config.validate()?;
    cuda::run(config, false)
}

/// Mark measured GEMM trials with CUDA profiler start/stop for external Nsight.
/// Setup, algorithm selection, warmup and validation are outside the capture.
/// Between-trial telemetry stays outside event timing but inside this range.
/// This does not launch a profiler; use ordinary [`benchmark`] for baseline rates.
pub fn benchmark_with_profiler_range(config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    config.validate()?;
    cuda::run(config, true)
}
