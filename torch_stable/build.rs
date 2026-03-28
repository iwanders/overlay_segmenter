use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .file("support/alloc_stableivalue.cpp")
        .cpp(true)
        .compile("iw_torch_stable");
    println!("cargo::rerun-if-changed=support/alloc_stableivalue.cpp");
    println!("cargo:rustc-link-lib=iw_torch_stable");

    let lib_path = "../train/.venv/lib/python3.13/site-packages/torch/lib";
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={lib_path}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch_cuda");
}
