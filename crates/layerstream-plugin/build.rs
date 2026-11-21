fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var("CARGO_FEATURE_DYNAMIC_BACKENDS").is_ok() {
        if target_os == "linux" {
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
        } else if target_os == "macos" {
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@loader_path");
        }
    }
    if target_os == "linux" && std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,/run/opengl-driver/lib");
    }
}
