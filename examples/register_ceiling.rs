use flops_ceiling::registers::{Config, ScaleFormat, Sparsity, benchmark};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        device: 0,
        format: ScaleFormat::Nvfp4,
        sparsity: Sparsity::Dense,
        a_value: 1.5,
        b_value: 2.0,
        scale_a: 1.0,
        scale_b: 0.5,
        accumulator_start: 2.5,
        accumulator_step: 0.25,
        accumulators: 16,
        threads_per_block: 128,
        blocks_per_sm: 8,
        iterations: 700,
        warmup_launches: 100,
        launches_per_trial: 1,
        trials: 10,
    };
    println!("Expected checksum: {}", config.expected_checksum()?);
    let result = benchmark(&config)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
