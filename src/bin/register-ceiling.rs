use clap::Parser;
use flops_ceiling::registers::{self, Config, Format, Sparsity};
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
    about = "Register-resident BF16/FP8/FP4 MMA and full FP32 FMA throughput on DGX Spark; sparse rates are dense-equivalent"
)]
struct Args {
    #[arg(long, value_enum, value_delimiter = ',', default_value = "mxfp4,nvfp4")]
    formats: Vec<Format>,
    /// Defaults to all supported modes per format (FP32: dense only).
    #[arg(long, value_enum, value_delimiter = ',')]
    sparsities: Vec<Sparsity>,
    /// Uniform A values; use --a-values=-1,0.5 for signed lists.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    a_values: Vec<f32>,
    /// Uniform B values; values must be exactly representable.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    b_values: Vec<f32>,
    /// Positive A block scales for FP4; must be 1 for BF16, FP8 and FP32.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    scale_a_values: Vec<f32>,
    /// Positive B block scales for FP4; must be 1 for BF16, FP8 and FP32.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    scale_b_values: Vec<f32>,
    /// Scalar accumulator j starts at start + j*step.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    accumulator_start: f32,
    /// Nonzero step; initial FP32 accumulator values must remain distinct.
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    accumulator_step: f32,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 16)]
    accumulators: u32,
    #[arg(long, default_value_t = 128)]
    threads: u32,
    #[arg(long, default_value_t = 8)]
    blocks_per_sm: u32,
    #[arg(long, default_value_t = 700)]
    iterations: u32,
    #[arg(long, default_value_t = 10)]
    warmup: u32,
    #[arg(long, default_value_t = 1)]
    launches_per_trial: u32,
    #[arg(long, default_value_t = 10)]
    trials: u32,
    /// Repeat all format/sparsity/input/scale combinations; odd rounds reverse order.
    #[arg(long, default_value_t = 1)]
    rounds: u32,
    /// New directory; refuses to overwrite an existing run.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Only write generated PTX and a manifest, without initializing CUDA.
    #[arg(long)]
    emit_ptx_only: bool,
    /// Mark measured launches for external Nsight capture; profiled timings are diagnostic.
    #[arg(long, conflicts_with = "emit_ptx_only")]
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
    let a = Args::parse();
    if a.rounds == 0 {
        return Err("rounds must be positive".into());
    }
    let out = a.output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/registers-{}-{}",
            flops_ceiling::unix_ms(),
            std::process::id()
        ))
    });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&out)?;
    let mut manifest = json!({"schema_version":3, "status":"running", "measurement":"register_only_instruction_throughput",
        "started_unix_ms":flops_ceiling::unix_ms(), "started_utc":command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "argv":std::env::args().collect::<Vec<_>>(), "crate_version":env!("CARGO_PKG_VERSION"),
        "git_revision":command("git", &["rev-parse", "HEAD"]), "git_status":command("git", &["status", "--porcelain"]),
        "rustc":command("rustc", &["--version"]), "host":command("uname", &["-a"]),
        "binary_sha256":std::env::current_exe().ok().and_then(|p| command("sha256sum", &[p.to_str()?])),
        "cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        "ld_library_path":std::env::var("LD_LIBRARY_PATH").ok(),
        "ptx_target":"sm_121a", "ptx_version":"9.0", "compiler":"CUDA driver JIT",
        "profiler_range":a.profile_range,
        "telemetry_csv_columns":["name","uuid","pci.bus_id","driver_version","temperature.gpu","utilization.gpu","power.draw","clocks.sm","clocks.mem"]});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    if lock.is_file() {
        fs::copy(lock, out.join("Cargo.lock"))?;
    }
    let mut records = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("results.jsonl"))?;
    let mut csv = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("summary.csv"))?;
    writeln!(
        csv,
        "round,case_id,format,sparsity,a_value,b_value,scale_a,scale_b,accumulator_start,accumulator_step,mean_tflops,sample_stddev,min_tflops,max_tflops,registers_per_thread,local_bytes_per_thread,profiler_range"
    )?;
    let mut cases = Vec::new();
    for format in a.formats {
        let sparsities = if a.sparsities.is_empty() {
            if format == Format::Fp32 {
                vec![Sparsity::Dense]
            } else {
                vec![Sparsity::Dense, Sparsity::TwoOfFour]
            }
        } else {
            a.sparsities.clone()
        };
        for sparsity in &sparsities {
            for &a_value in &a.a_values {
                for &b_value in &a.b_values {
                    for &scale_a in &a.scale_a_values {
                        for &scale_b in &a.scale_b_values {
                            cases.push(Config {
                                device: a.device,
                                format,
                                sparsity: *sparsity,
                                a_value,
                                b_value,
                                scale_a,
                                scale_b,
                                accumulator_start: a.accumulator_start,
                                accumulator_step: a.accumulator_step,
                                accumulators: a.accumulators,
                                threads_per_block: a.threads,
                                blocks_per_sm: a.blocks_per_sm,
                                iterations: a.iterations,
                                warmup_launches: a.warmup,
                                launches_per_trial: a.launches_per_trial,
                                trials: a.trials,
                            });
                        }
                    }
                }
            }
        }
    }
    manifest["cases"] = json!(cases);
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let mut cases: Vec<_> = cases.into_iter().enumerate().collect();
    let mut errors = Vec::new();
    if a.profile_range {
        eprintln!(
            "Profiler capture ranges enabled. For TFLOPS (no profiler attached), run without Nsight Systems or Nsight Compute. Timings under Nsight are diagnostic."
        );
    }
    for round in 0..a.rounds {
        if round > 0 {
            cases.reverse();
        }
        for (case_id, c) in &cases {
            let tag = format!("r{round}-c{case_id}-{:?}-{:?}", c.format, c.sparsity).to_lowercase();
            let result = (|| -> Result<(), Box<dyn std::error::Error>> {
                fs::write(out.join(format!("{tag}.ptx")), registers::kernel_ptx(c)?)?;
                if a.emit_ptx_only {
                    return Ok(());
                }
                eprintln!(
                    "Round {round}, case {case_id}: {:?} {:?}, A={}, B={}, scales={}/{}, accumulator start={}, step={}",
                    c.format,
                    c.sparsity,
                    c.a_value,
                    c.b_value,
                    c.scale_a,
                    c.scale_b,
                    c.accumulator_start,
                    c.accumulator_step
                );
                let result = if a.profile_range {
                    registers::benchmark_with_profiler_range(c)?
                } else {
                    registers::benchmark(c)?
                };
                let s = &result.tflops;
                let sd = s
                    .sample_stddev
                    .map(|x| format!("{x:.2}"))
                    .unwrap_or_else(|| "n/a (preliminary)".into());
                println!(
                    "{tag}: {:.2} ± {sd} TFLOPS; best {:.2}; {} registers/thread, {} local bytes; checksums passed",
                    s.mean,
                    s.max,
                    result.kernel.registers_per_thread,
                    result.kernel.local_bytes_per_thread
                );
                serde_json::to_writer(
                    &mut records,
                    &json!({"round":round,"case_id":case_id,"result":result}),
                )?;
                writeln!(records)?;
                records.flush()?;
                writeln!(
                    csv,
                    "{round},{case_id},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    c.format,
                    c.sparsity,
                    c.a_value,
                    c.b_value,
                    c.scale_a,
                    c.scale_b,
                    c.accumulator_start,
                    c.accumulator_step,
                    s.mean,
                    s.sample_stddev.map(|x| x.to_string()).unwrap_or_default(),
                    s.min,
                    s.max,
                    result.kernel.registers_per_thread,
                    result.kernel.local_bytes_per_thread,
                    result.profiler_range
                )?;
                csv.flush()?;
                Ok(())
            })();
            if let Err(e) = result {
                eprintln!("{tag}: {e}");
                errors.push(json!({"round":round,"case_id":case_id,"config":c,"error":e.to_string(),"unix_ms":flops_ceiling::unix_ms()}));
                fs::write(out.join("errors.json"), serde_json::to_vec_pretty(&errors)?)?;
            }
        }
    }
    manifest["status"] = json!(if !errors.is_empty() {
        "failed"
    } else if a.emit_ptx_only {
        "ptx_only"
    } else {
        "completed"
    });
    manifest["exit_code"] = json!(if errors.is_empty() { 0 } else { 1 });
    manifest["finished_unix_ms"] = json!(flops_ceiling::unix_ms());
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("Artifacts: {}", out.display());
    if !errors.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}
