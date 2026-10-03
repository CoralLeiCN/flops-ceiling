//! Exact constant encodings and a CPU FP32 recurrence for uniform operands.
use super::{Config, ScaleFormat};
use crate::{Error, Result};

fn packed_e2m1(value: f32, name: &str) -> Result<u32> {
    let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let code = magnitudes.iter().position(|&x| x == value.abs())
        .ok_or_else(|| Error(format!("{name}={value} is not exactly representable in E2M1; use 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, or ±6")))?;
    let nibble = code as u32 | if value.is_sign_negative() { 8 } else { 0 };
    Ok(nibble * 0x11111111)
}

fn packed_scale(value: f32, format: ScaleFormat, name: &str) -> Result<u32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error(format!("{name} must be finite and positive")));
    }
    let code = match format {
        ScaleFormat::Nvfp4 => (1_u32..=126).find(|&code| {
            let exponent = code >> 3;
            let mantissa = code & 7;
            let decoded = if exponent == 0 {
                mantissa as f32 / 512.0
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2_f32.powi(exponent as i32 - 7)
            };
            decoded == value
        }),
        ScaleFormat::Mxfp4 => (0_u32..=254)
            .find(|&code| 2_f32.powi(code as i32 - 127) == value),
        _ => return Ok(0),
    }.ok_or_else(|| Error(format!("{name}={value} is not exactly representable as a positive {format:?} block scale; values are not silently rounded")))?;
    Ok(code * 0x01010101)
}

pub(super) fn initial(c: &Config, j: u32) -> f32 {
    c.accumulator_start + j as f32 * c.accumulator_step
}

pub(super) fn validate(c: &Config) -> Result<()> {
    operands(c, 0)?;
    if !c.accumulator_start.is_finite()
        || !c.accumulator_step.is_finite()
        || c.accumulator_step == 0.0
    {
        return Err(Error(
            "accumulator start must be finite and step must be finite and nonzero".into(),
        ));
    }
    let mut seen = Vec::new();
    for j in 0..c.scalar_accumulators() {
        let x = initial(c, j);
        if !x.is_finite() || seen.contains(&x) {
            return Err(Error("accumulator initialization must produce distinct finite FP32 values; adjust start/step to preserve independent chains".into()));
        }
        seen.push(x);
    }
    // Check the reference before GPU allocation/compilation, including probes.
    for case in 0..3 {
        let iterations = if case == 0 { c.iterations } else { 3 };
        checksum(c, iterations, case)?;
    }
    Ok(())
}

pub(super) fn input_values(c: &Config, case: u32) -> (f32, f32, f32, f32) {
    match case {
        1 if c.format.block_scaled() => (1.0, 1.0, 2.0, 1.0),
        1 => (1.0, 2.0, 1.0, 1.0),
        2 => (-1.0, 2.0, 1.0, 1.0),
        _ => (c.a_value, c.b_value, c.scale_a, c.scale_b),
    }
}

fn packed_value(value: f32, format: ScaleFormat, name: &str) -> Result<u32> {
    if !value.is_finite() {
        return Err(Error(format!("{name} must be finite")));
    }
    match format {
        ScaleFormat::Nvfp4 | ScaleFormat::Mxfp4 => packed_e2m1(value, name),
        ScaleFormat::Fp32 => Ok(value.to_bits()),
        ScaleFormat::Bf16 if value.to_bits() & 0xffff == 0 => Ok((value.to_bits() >> 16) * 0x10001),
        ScaleFormat::Fp8 => (0_u32..=126)
            .find(|&code| crate::input::e4m3(code as u8) == value.abs())
            .map(|code| (code | if value.is_sign_negative() { 128 } else { 0 }) * 0x01010101)
            .ok_or_else(|| {
                Error(format!(
                    "{name}={value} is not exactly representable in FP8 E4M3"
                ))
            }),
        _ => Err(Error(format!(
            "{name}={value} is not exactly representable in BF16; values are not silently rounded"
        ))),
    }
}

pub(super) fn operands(c: &Config, case: u32) -> Result<([u32; 4], f32)> {
    let (a, b, sa, sb) = input_values(c, case);
    let packed = [
        packed_value(a, c.format, "A")?,
        packed_value(b, c.format, "B")?,
        packed_scale(sa, c.format, "scale A")?,
        packed_scale(sb, c.format, "scale B")?,
    ];
    let product = f64::from(a) * f64::from(b) * f64::from(sa) * f64::from(sb);
    let increment64 = product * f64::from(c.format.dense_k());
    let increment = increment64 as f32;
    // Scalar FMA performs one fused rounding per recurrence, handled below.
    if c.format == ScaleFormat::Fp32 {
        return Ok((packed, increment));
    }
    // Reject FP32 overflow/underflow combinations and subnormal products rather
    // than depending on device-specific denormal or reduction behavior.
    if !increment.is_finite()
        || f64::from(increment) != increment64
        || (product != 0.0 && product.abs() < f64::from(f32::MIN_POSITIVE))
    {
        return Err(Error("product must be zero or normal FP32, and the MMA increment must be finite and exactly representable in FP32".into()));
    }
    Ok((packed, increment))
}

pub(super) fn checksum(c: &Config, iterations: u32, case: u32) -> Result<f32> {
    let increment = operands(c, case)?.1;
    let (a, b, _, _) = input_values(c, case);
    let mut d: Vec<f32> = (0..c.scalar_accumulators())
        .map(|j| initial(c, j))
        .collect();
    // Repeated FP32 rounding matters for large/small increments. Do not replace
    // this recurrence with start + iterations*increment or reassociate the sum.
    for _ in 0..iterations {
        for x in &mut d {
            *x = if c.format == ScaleFormat::Fp32 {
                a.mul_add(b, *x)
            } else {
                *x + increment
            };
        }
    }
    let mut sum = 0_f32;
    for x in d {
        sum += x;
    }
    if !sum.is_finite() {
        return Err(Error("FP32 reference checksum overflowed; reduce values, scales, initialization or iterations".into()));
    }
    Ok(sum)
}
