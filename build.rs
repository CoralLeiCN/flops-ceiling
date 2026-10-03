fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUBLAS_LIB_DIR");
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let cublas = std::env::var("CUBLAS_LIB_DIR").unwrap_or_else(|_| format!("{cuda}/lib64"));
    println!("cargo:rustc-link-search=native={cublas}");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // The register-only API loads Rust-generated PTX through the CUDA driver.
    // Stubs support linking without a visible GPU; never add them to RUNPATH.
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    println!("cargo:rustc-link-lib=dylib=cuda");
    // RUNPATH can be overridden by LD_LIBRARY_PATH for isolated library comparisons.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cublas}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
