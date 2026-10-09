use clap::Parser;
use flops_ceiling::{Backend, BenchmarkConfig, Precision, Selection, Shape, Timing};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::Command,
};

#[derive(Parser)]
#[command(
    version,
    about = "Validated dense GEMM throughput on NVIDIA GPUs (BF16/FP8/NVFP4, full FP32 and TF32)"
)]
struct Args {
    #[arg(long, value_enum, default_value = "cublaslt")]
    backend: Backend,
    #[arg(long, value_enum, default_value = "nvfp4")]
    precision: Precision,
    /// Comma-separated MxNxK shapes; M/N multiples of 16, K multiple of 32.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "64x4096x4096,512x4096x4096,4096x4096x4096"
    )]
    shapes: Vec<Shape>,
    #[arg(long, value_enum, default_value = "graph")]
    timing: Timing,
    #[arg(long, value_enum, default_value = "tune")]
    selection: Selection,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 64)]
    workspace_mib: usize,
    #[arg(long, default_value_t = 32)]
    candidates: usize,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    trials: usize,
    #[arg(long, default_value_t = 5)]
    tune_warmup: usize,
    #[arg(long, default_value_t = 20)]
    tune_iterations: usize,
    #[arg(long, default_value_t = 3)]
    tune_trials: usize,
    /// Use at least 64; set to M*N for an exhaustive CPU reference on small cases.
    #[arg(long, default_value_t = 256)]
    validation_samples: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// A NEW directory; existing directories are refused to preserve raw results.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Mark measured GEMMs for external Nsight capture; profiled timings are diagnostic.
    #[arg(long)]
    profile_range: bool,
}
fn command(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let workspace_bytes = args
        .workspace_mib
        .checked_mul(1024 * 1024)
        .ok_or("workspace size overflow")?;
    let output = args.output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/{}-{}",
            flops_ceiling::unix_ms(),
            std::process::id()
        ))
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    let started = flops_ceiling::unix_ms();
    let mut manifest = json!({
        "schema_version":1, "status":"running", "started_unix_ms":started,
        "started_utc":command("date", &["-u","+%Y-%m-%dT%H:%M:%SZ"]),
        "argv":std::env::args().collect::<Vec<_>>(),
        "crate_version":env!("CARGO_PKG_VERSION"),
        "profiler_range":args.profile_range,
        "git_revision":command("git", &["rev-parse","HEAD"]),
        "git_status":command("git", &["status","--porcelain"]),
        "rustc":command("rustc", &["--version"]),
        "binary_sha256":std::env::current_exe().ok().and_then(|p|command("sha256sum", &[p.to_str()?])),
        "cutile_feature_enabled":cfg!(feature = "cutile"),
        "cutile_revision":if cfg!(feature = "cutile") { Some("cc720f182f38bf46527753caa340e78d6d5fa3cc") } else { None },
        "cutile_tileiras_path":std::env::var("CUTILE_TILEIRAS_PATH").ok(),
        "cutile_bytecode_version":std::env::var("CUTILE_BYTECODE_VERSION").ok(),
        "host":command("uname", &["-a"]),
        "ld_library_path":std::env::var("LD_LIBRARY_PATH").ok(),
        "cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        "nvidia_tf32_override":std::env::var("NVIDIA_TF32_OVERRIDE").ok(),
        "cuda_home":std::env::var("CUDA_HOME").ok(),
        "telemetry_csv_columns":["name","uuid","pci.bus_id","driver_version","temperature.gpu","utilization.gpu","power.draw","clocks.sm","clocks.mem"],
    });
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    if lock.is_file() {
        fs::copy(lock, output.join("Cargo.lock"))?;
    }
    let mut results = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("results.jsonl"))?;
    let mut summary = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("summary.csv"))?;
    writeln!(
        summary,
        "backend,precision,m,n,k,timing,selection,cublas_version,mean_tflops,sample_stddev,min_tflops,max_tflops,algorithm_id,validation_relative_rmse,profiler_range"
    )?;
    let mut errors = Vec::new();
    if args.profile_range {
        eprintln!(
            "Profiler capture ranges enabled. For TFLOPS (no profiler attached), run without Nsight Systems or Nsight Compute. Timings under Nsight are diagnostic."
        );
    }
    for shape in args.shapes {
        let config = BenchmarkConfig {
            backend: args.backend,
            device: args.device,
            shape,
            precision: args.precision,
            timing: args.timing,
            selection: args.selection,
            workspace_bytes,
            candidates: args.candidates,
            warmup: args.warmup,
            iterations: args.iterations,
            trials: args.trials,
            tune_warmup: args.tune_warmup,
            tune_iterations: args.tune_iterations,
            tune_trials: args.tune_trials,
            validation_samples: args.validation_samples,
            seed: args.seed,
        };
        eprintln!(
            "Benchmarking {:?} {shape}, {:?}, {:?}",
            args.precision, args.selection, args.timing
        );
        let result = if args.profile_range {
            flops_ceiling::benchmark_with_profiler_range(&config)
        } else {
            flops_ceiling::benchmark(&config)
        };
        match result {
            Ok(result) => {
                let stats = &result.tflops;
                let sd = stats
                    .sample_stddev
                    .map(|s| format!("{s:.2}"))
                    .unwrap_or_else(|| "n/a (single trial)".into());
                println!(
                    "{shape}: {:.2} ± {sd} TFLOPS; range {:.2}–{:.2}; validation RMSE {:.5}",
                    stats.mean, stats.min, stats.max, result.validation_after.relative_rmse
                );
                serde_json::to_writer(&mut results, &result)?;
                writeln!(results)?;
                results.flush()?;
                writeln!(
                    summary,
                    "{:?},{:?},{},{},{},{:?},{:?},{},{},{},{},{},{},{},{}",
                    args.backend,
                    args.precision,
                    shape.m,
                    shape.n,
                    shape.k,
                    args.timing,
                    args.selection,
                    result.device.cublas_version,
                    stats.mean,
                    stats
                        .sample_stddev
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                    stats.min,
                    stats.max,
                    result.candidates[result.selected_candidate].algorithm_id,
                    result.validation_after.relative_rmse,
                    result.profiler_range
                )?;
                summary.flush()?;
            }
            Err(error) => {
                eprintln!("{shape}: {error}");
                errors.push(json!({"config":config,"error":error.to_string(),"unix_ms":flops_ceiling::unix_ms()}));
                fs::write(
                    output.join("errors.json"),
                    serde_json::to_vec_pretty(&errors)?,
                )?;
            }
        }
    }
    manifest["status"] = json!(if errors.is_empty() {
        "completed"
    } else {
        "failed"
    });
    manifest["exit_code"] = json!(if errors.is_empty() { 0 } else { 1 });
    manifest["finished_unix_ms"] = json!(flops_ceiling::unix_ms());
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("Artifacts: {}", output.display());
    if !errors.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}
