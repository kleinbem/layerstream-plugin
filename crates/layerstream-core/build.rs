use std::env;
use std::path::{Path, PathBuf};

fn resolve_llama_cpp_dir() -> PathBuf {
    if let Ok(dir) = env::var("LLAMA_CPP_DIR") {
        let path = PathBuf::from(dir);
        if path.join("CMakeLists.txt").is_file() {
            return path;
        }
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let local_vendor = manifest_dir.join("../../third_party/llama.cpp");
    if local_vendor.join("CMakeLists.txt").is_file() {
        return local_vendor;
    }

    // Convenient local sibling checkout during development.
    let dev_checkout = PathBuf::from("/home/martin/Develop/github.com/ggml-org/llama.cpp");
    if dev_checkout.join("CMakeLists.txt").is_file() {
        return dev_checkout;
    }

    panic!("Could not locate llama.cpp source tree. Set LLAMA_CPP_DIR or enter `nix develop`.");
}

fn find_static_libs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_static_libs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("a") {
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-changed=cpp/stream_engine.h");
    println!("cargo:rerun-if-changed=cpp/stream_engine.cpp");

    let llama_dir = resolve_llama_cpp_dir();
    let use_vulkan = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let use_metal = env::var("CARGO_FEATURE_METAL").is_ok();

    let mut cfg = cmake::Config::new(&llama_dir);
    cfg.profile("Release")
        .build_target("llama")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_VULKAN", if use_vulkan { "ON" } else { "OFF" })
        .define("GGML_CUDA", if use_cuda { "ON" } else { "OFF" })
        .define("GGML_CUDA_GRAPHS", "OFF")
        .define("GGML_METAL", if use_metal { "ON" } else { "OFF" });

    if env::var("CMAKE_GENERATOR").is_err() && which_ninja() {
        cfg.generator("Ninja");
    }

    let dst = cfg.build();

    // Compile our thin zero-patch C++ shim against llama.cpp headers (including internal llama-model.h).
    let mut cc_build = cc::Build::new();
    cc_build
        .cpp(true)
        .std("c++17")
        .file("cpp/stream_engine.cpp")
        .include("cpp")
        .include(llama_dir.join("include"))
        .include(llama_dir.join("src"))
        .include(llama_dir.join("ggml/include"))
        .include(llama_dir.join("ggml/src"))
        .warnings(false);

    if use_cuda {
        cc_build.define("LAYERSTREAM_USE_CUDA", None);
        if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
            cc_build.include(Path::new(&cuda_path).join("include"));
        }
    }
    if use_vulkan {
        cc_build.define("LAYERSTREAM_USE_VULKAN", None);
    }
    if use_metal {
        cc_build.define("LAYERSTREAM_USE_METAL", None);
    }

    cc_build.compile("layerstream_cpp");

    // Link static archives built by llama.cpp in dependency order.
    let mut lib_dirs = std::collections::BTreeSet::new();
    let mut libs = Vec::new();
    for subdir in ["lib", "lib64", "build"] {
        let dir = dst.join(subdir);
        if dir.is_dir() {
            find_static_libs(&dir, &mut libs);
        }
    }
    for lib in &libs {
        if let Some(parent) = lib.parent() {
            lib_dirs.insert(parent.to_path_buf());
        }
    }
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    let ordered_names = [
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-cuda",
        "ggml-vulkan",
        "ggml-metal",
        "ggml-blas",
        "ggml-base",
    ];
    for name in ordered_names {
        let archive = format!("lib{name}.a");
        if libs
            .iter()
            .any(|p| p.file_name().and_then(|f| f.to_str()) == Some(&archive))
        {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }

    // Link platform & backend runtime libraries.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        if use_vulkan {
            println!("cargo:rustc-link-lib=vulkan");
        }
        if use_cuda {
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=cublasLt");
        }
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if use_metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }
}

fn which_ninja() -> bool {
    std::process::Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
