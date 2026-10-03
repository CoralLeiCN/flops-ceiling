use flops_ceiling::{BenchmarkConfig, Precision, Shape, benchmark};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = benchmark(&BenchmarkConfig {
        shape: Shape {
            m: 4096,
            n: 4096,
            k: 4096,
        },
        precision: Precision::Nvfp4,
        ..Default::default()
    })?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
