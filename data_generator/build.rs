// build.rs
// https://github.com/LaurentMazare/tch-rs/issues/923#issuecomment-2669687652
fn main() {
    let lib_path = if std::env::var("CARGO_FEATURE_USE_TORCH_DEVEL").is_ok() {
        "/workspace/ivor/ml/pytorch_dev/pytorch/build/lib"
    } else {
        "../train/.venv/lib/python3.13/site-packages/torch/lib"
    };

    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
    // println!("cargo:rustc-link-arg=-L{}", lib_path);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);
}
