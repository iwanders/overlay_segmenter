use std::env;
use std::path::PathBuf;

fn main() {
    let lib_path = "../train/.venv/lib/python3.13/site-packages/torch/lib";
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={lib_path}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
}
