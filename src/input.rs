use crate::{Precision, Shape};

pub(crate) struct Operand {
    pub bytes: Vec<u8>,
    pub scales: Vec<u8>,
    pub decoded: Vec<f32>,
}

fn hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// NVIDIA's 128-row x 4-scale-column layout: [row/128, col/4, row%32, row%128/32, col%4].
pub(crate) fn scale_offset(row: usize, col: usize, scale_cols: usize) -> usize {
    ((row / 128) * scale_cols.div_ceil(4) + col / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + col % 4
}

pub(crate) fn e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        (mantissa as f32) / 512.0
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    };
    if code & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub(crate) fn generate(rows: usize, k: usize, precision: Precision, seed: u64) -> Operand {
    let len = rows * k;
    let mut bytes = vec![
        0;
        match precision {
            Precision::Fp32 | Precision::Tf32 => len * 4,
            Precision::Bf16 => len * 2,
            Precision::Fp8 => len,
            Precision::Nvfp4 => len / 2,
        }
    ];
    let mut scales = if precision == Precision::Nvfp4 {
        vec![0; rows.div_ceil(128) * (k / 16).div_ceil(4) * 512]
    } else {
        1.0_f32.to_ne_bytes().to_vec()
    };
    let mut decoded = vec![0.0; len];
    for row in 0..rows {
        for block in 0..k / 16 {
            let code = [0x2c, 0x34, 0x38, 0x3c]
                [(hash(seed ^ ((row * k / 16 + block) as u64)) & 3) as usize];
            let scale = e4m3(code);
            if precision == Precision::Nvfp4 {
                scales[scale_offset(row, block, k / 16)] = code;
            }
            for lane in 0..16 {
                let i = row * k + block * 16 + lane;
                let bits = hash(seed.wrapping_add(i as u64));
                decoded[i] = match precision {
                    Precision::Fp32 | Precision::Tf32 => {
                        let value =
                            (((bits >> 32) as u32) as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32;
                        bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_ne_bytes());
                        value
                    }
                    Precision::Nvfp4 => {
                        let nibble = (bits >> 32) as u8 & 15;
                        bytes[i / 2] |= nibble << ((i % 2) * 4);
                        let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize];
                        (if nibble & 8 != 0 { -value } else { value }) * scale
                    }
                    Precision::Fp8 => {
                        let code = (((bits >> 32) % 96) as u8) | ((bits >> 24) as u8 & 128);
                        bytes[i] = code;
                        e4m3(code)
                    }
                    Precision::Bf16 => {
                        let value =
                            (((bits >> 32) as u32) as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32;
                        let raw = value.to_bits();
                        let bf16 = ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) as u16;
                        bytes[i * 2..i * 2 + 2].copy_from_slice(&bf16.to_ne_bytes());
                        f32::from_bits((bf16 as u32) << 16)
                    }
                };
            }
        }
    }
    Operand {
        bytes,
        scales,
        decoded,
    }
}

pub(crate) fn reference(
    shape: Shape,
    a: &Operand,
    b: &Operand,
    samples: usize,
    seed: u64,
) -> Vec<(usize, f64)> {
    let count = samples.min(shape.m * shape.n);
    // Include corners and tile boundaries; fill the rest with reproducible distributed samples.
    let mut indices = std::collections::BTreeSet::new();
    for row in [0, 31, 32, 63, 64, 127, 128, shape.m - 1] {
        for col in [0, 31, 32, 63, 64, 127, 128, shape.n - 1] {
            if row < shape.m && col < shape.n && indices.len() < count {
                indices.insert(col * shape.m + row);
            }
        }
    }
    let mut cursor = 0_u64;
    while indices.len() < count {
        indices.insert((hash(seed.wrapping_add(cursor)) as usize) % (shape.m * shape.n));
        cursor += 1;
    }
    indices
        .into_iter()
        .map(|i| {
            let row = i % shape.m;
            let col = i / shape.m;
            let sum = (0..shape.k)
                .map(|k| a.decoded[row * shape.k + k] as f64 * b.decoded[col * shape.k + k] as f64)
                .sum();
            (i, sum)
        })
        .collect()
}
