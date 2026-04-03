use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .file("support/alloc_stableivalue.cpp")
        .cpp(true)
        .compile("iw_torch_stable");
    println!("cargo::rerun-if-changed=support/alloc_stableivalue.cpp");

    let lib_path = if std::env::var("CARGO_FEATURE_USE_TORCH_DEVEL").is_ok() {
        "/workspace/ivor/ml/pytorch_dev/pytorch/build/lib"
    } else {
        "../train/.venv/lib/python3.13/site-packages/torch/lib"
    };

    //println!("cargo::rerun-if-changed=build.rs");
    // let lib_path = ;
    //let lib_path = "/workspace/ivor/ml/pytorch_dev/pytorch/build/lib";
    // let lib_path = "/home/ivor/Documents/Code/rust/overlay_segmenter/repo/train/.venv/lib/python3.13/site-packages/torch/lib/";
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={lib_path}");
    println!("cargo:rustc-link-lib=iw_torch_stable");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    // println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_cuda");
    //
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
    // println!("cargo:rustc-link-arg=-L{}", lib_path);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // println!("cargo:rustc-link-search=static:{}", out_path.display());
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_stable");
}
