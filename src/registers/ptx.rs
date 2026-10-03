use super::{Config, ScaleFormat, Sparsity};
use crate::Result;
use std::fmt::Write;

/// Generate the complete SM121 kernel. No operand loads or stores occur inside
/// the MMA loop. Each thread writes one checksum after consuming every result.
pub(super) fn generate(c: &Config) -> Result<String> {
    let mut ptx = String::from(".version 9.0\n.target sm_121a\n.address_size 64\n\n");
    for (case, name) in [
        (0, "register_mma"),
        (1, "probe_scale"),
        (2, "probe_negative"),
    ] {
        ptx.push_str(&entry(c, name, super::values::operands(c, case)?.0));
    }
    Ok(ptx)
}

fn entry(c: &Config, name: &str, values: [u32; 4]) -> String {
    let mut ptx = format!(
        ".visible .entry {name}(\n.param .u64 output, .param .u32 iterations, .param .u64 operands)\n{{\n"
    );
    ptx.push_str(
        ".reg .b32 a, b, sa, sb, metadata;\n\
         .reg .u32 count, i, tid, block, block_size, index;\n\
         .reg .u64 out, offset, address;\n\
         .reg .pred again;\n\
         .reg .f32 sum;\n",
    );
    writeln!(ptx, ".reg .f32 d<{}>;", c.scalar_accumulators()).unwrap();
    ptx.push_str(
        "ld.param.u64 out, [output];\n\
         ld.param.u32 count, [iterations];\n\
         mov.b32 metadata, 0xeeeeeeee;\n",
    );
    // Operand parameters can be rematerialized as constant-memory loads inside
    // the machine loop. Specialize constants to guarantee register/immediate use.
    if c.format == ScaleFormat::Fp32 {
        ptx.push_str(".reg .u64 inputs;\n.reg .f32 fa, fb;\nld.param.u64 inputs, [operands];\nld.global.f32 fa, [inputs];\nld.global.f32 fb, [inputs+4];\n");
    } else {
        for (reg, value) in ["a", "b", "sa", "sb"].into_iter().zip(values) {
            writeln!(ptx, "mov.b32 {reg}, 0x{value:08x};").unwrap();
        }
    }
    // Distinct initial accumulator values prevent identical recurrence chains
    // from being coalesced by the assembler. Validation checks this in FP32.
    for j in 0..c.scalar_accumulators() {
        writeln!(
            ptx,
            "mov.f32 d{j}, 0f{:08x};",
            super::values::initial(c, j).to_bits()
        )
        .unwrap();
    }
    ptx.push_str("mov.u32 i, 0;\nMMA_LOOP:\n");
    let k = c.format.dense_k() * if c.sparsity == Sparsity::Dense { 1 } else { 2 };
    let shape = format!("m16n8k{k}");
    let (op, b, metadata) = match c.sparsity {
        Sparsity::Dense => ("mma", "{b,b}", ""),
        Sparsity::TwoOfFour => ("mma.sp::ordered_metadata", "{b,b,b,b}", "metadata, 0, "),
    };
    let (scale_vec, scale_type) = match c.format {
        ScaleFormat::Nvfp4 => ("4X", "ue4m3"),
        ScaleFormat::Mxfp4 => ("2X", "ue8m0"),
        _ => ("", ""),
    };
    for j in 0..c.accumulators {
        if c.format == ScaleFormat::Fp32 {
            writeln!(ptx, "fma.rn.f32 d{j}, fa, fb, d{j};").unwrap();
            continue;
        }
        let d = (j * 4..j * 4 + 4)
            .map(|i| format!("d{i}"))
            .collect::<Vec<_>>()
            .join(",");
        if c.format.block_scaled() {
            writeln!(ptx,
            "{op}.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::{scale_vec}.{shape}.row.col.f32.e2m1.e2m1.f32.{scale_type} \
             {{{d}}}, {{a,a,a,a}}, {b}, {{{d}}}, {metadata}sa, {{0,0}}, sb, {{0,0}};"
        ).unwrap();
        } else {
            let dtype = if c.format == ScaleFormat::Bf16 {
                "bf16"
            } else {
                "e4m3"
            };
            let suffix = if c.sparsity == Sparsity::Dense {
                ""
            } else {
                ", metadata, 0"
            };
            writeln!(ptx, "{op}.sync.aligned.{shape}.row.col.f32.{dtype}.{dtype}.f32 {{{d}}}, {{a,a,a,a}}, {b}, {{{d}}}{suffix};").unwrap();
        }
    }
    ptx.push_str(
        "add.u32 i, i, 1;\nsetp.lt.u32 again, i, count;\n@again bra MMA_LOOP;\n\
         mov.f32 sum, 0f00000000;\n",
    );
    for j in 0..c.scalar_accumulators() {
        writeln!(ptx, "add.rn.f32 sum, sum, d{j};").unwrap();
    }
    ptx.push_str(
        "mov.u32 tid, %tid.x;\nmov.u32 block, %ctaid.x;\n\
         mov.u32 block_size, %ntid.x;\n\
         mad.lo.u32 index, block, block_size, tid;\n\
         mul.wide.u32 offset, index, 4;\nadd.u64 address, out, offset;\n\
         st.global.f32 [address], sum;\nret;\n}\n",
    );
    ptx
}
