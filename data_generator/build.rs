// build.rs
// https://github.com/LaurentMazare/tch-rs/issues/923#issuecomment-2669687652
fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
            }
            // $VIRTUAL_ENV/lib/python3.13/site-packages/torch/lib
            if let Some(env_path) = std::env::var_os("VIRTUAL_ENV") {
                println!("cargo:rustc-link-search={}/lib/python3.13/site-packages/torch/lib", env_path.display());

            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            //println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
}
