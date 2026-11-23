use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN/_backends:$ORIGIN");
    } else if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path/_backends");
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path");
    }
}
