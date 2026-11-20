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

    panic!(
        "Could not locate pinned llama.cpp source tree. \
         Set LLAMA_CPP_DIR, checkout third_party/llama.cpp, or enter `nix develop`."
    );
}

fn find_libs_with_filter(root: &Path, out: &mut Vec<PathBuf>, pred: &dyn Fn(&Path) -> bool) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_libs_with_filter(&path, out, pred);
        } else if pred(&path) {
            out.push(path);
        }
    }
}

fn is_static_lib(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("a" | "lib"))
}

fn is_shared_lib(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.ends_with(".so")
        || name.contains(".so.")
        || name.ends_with(".dylib")
        || name.ends_with(".dll")
}

fn stage_shared_libs(shared_libs: &[PathBuf], stage_dir: &Path) {
    let _ = std::fs::create_dir_all(stage_dir);
    for src in shared_libs {
        let Some(fname) = src.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !(fname.starts_with("libggml")
            || fname.starts_with("libllama")
            || fname.starts_with("ggml")
            || fname.starts_with("llama"))
        {
            continue;
        }
        let dst = stage_dir.join(fname);
        // Copy dereferenced contents if symlink, or regular file, so archives stay self-contained.
        if let Ok(meta) = std::fs::metadata(src)
            && meta.is_file()
        {
            let _ = std::fs::copy(src, &dst);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LAYERSTREAM_BACKEND_STAGE_DIR");
    println!("cargo:rerun-if-changed=cpp/stream_engine.h");
    println!("cargo:rerun-if-changed=cpp/stream_engine.cpp");

    let llama_dir = resolve_llama_cpp_dir();
    let use_vulkan = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let use_metal = env::var("CARGO_FEATURE_METAL").is_ok();
    let dynamic_backends = env::var("CARGO_FEATURE_DYNAMIC_BACKENDS").is_ok();

    let mut cfg = cmake::Config::new(&llama_dir);
    cfg.profile("Release")
        .define(
            "BUILD_SHARED_LIBS",
            if dynamic_backends { "ON" } else { "OFF" },
        )
        .define(
            "GGML_BACKEND_DL",
            if dynamic_backends { "ON" } else { "OFF" },
        )
        .define("GGML_CPU_ALL_VARIANTS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_VULKAN", if use_vulkan { "ON" } else { "OFF" })
        .define("GGML_CUDA", if use_cuda { "ON" } else { "OFF" })
        .define("GGML_CUDA_GRAPHS", "OFF")
        .define("GGML_METAL", if use_metal { "ON" } else { "OFF" })
        .define(
            "GGML_METAL_EMBED_LIBRARY",
            if use_metal { "ON" } else { "OFF" },
        )
        .define("GGML_METAL_NDEBUG", "ON");

    if !dynamic_backends {
        cfg.build_target("llama");
    }

    if use_cuda {
        if let Ok(archs) = env::var("CMAKE_CUDA_ARCHITECTURES") {
            cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
        }
        if let Ok(root) = env::var("CUDAToolkit_ROOT").or_else(|_| env::var("CUDA_PATH")) {
            cfg.define("CUDAToolkit_ROOT", root);
        }
        if let Ok(compiler) = env::var("CMAKE_CUDA_COMPILER") {
            cfg.define("CMAKE_CUDA_COMPILER", compiler);
        }
        if let Ok(host_cc) = env::var("CMAKE_CUDA_HOST_COMPILER") {
            cfg.define("CMAKE_CUDA_HOST_COMPILER", host_cc);
        }
    }

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

    if dynamic_backends {
        cc_build.define("LAYERSTREAM_DYNAMIC_BACKENDS", None);
    }

    if use_cuda && !dynamic_backends {
        cc_build.define("LAYERSTREAM_USE_CUDA", None);
        let has_nixos_driver = Path::new("/run/opengl-driver/lib/libcuda.so.1").exists();
        if Path::new("/run/opengl-driver/lib").is_dir() {
            println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");
            println!("cargo:rustc-link-arg-tests=-Wl,-rpath,/run/opengl-driver/lib");
            println!("cargo:rustc-link-arg-examples=-Wl,-rpath,/run/opengl-driver/lib");
        }
        if let Ok(cuda_path) = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDAToolkit_ROOT"))
            .or_else(|_| env::var("CUDA_HOME"))
        {
            cc_build.include(Path::new(&cuda_path).join("include"));
            for sub in ["lib", "lib64"] {
                let d = Path::new(&cuda_path).join(sub);
                if d.is_dir() {
                    println!("cargo:rustc-link-search=native={}", d.display());
                }
            }
            // Only fall back to CUDA link-time stubs when no real host driver library
            // is present (e.g., pure sandbox / cross-build). Putting Nix store stub dirs
            // in -L causes ld-wrapper to record them in DT_RUNPATH, overriding the host
            // libcuda.so.1 at runtime when spawned outside `nix develop .#cuda`!
            if !has_nixos_driver {
                for sub in ["lib/stubs", "lib64/stubs"] {
                    let d = Path::new(&cuda_path).join(sub);
                    if d.is_dir() {
                        println!("cargo:rustc-link-search=native={}", d.display());
                    }
                }
            }
        }
    }
    if use_vulkan {
        cc_build.define("LAYERSTREAM_USE_VULKAN", None);
    }
    if use_metal {
        cc_build.define("LAYERSTREAM_USE_METAL", None);
    }

    cc_build.compile("layerstream_cpp");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if dynamic_backends {
        let mut shared_libs = Vec::new();
        for subdir in ["lib", "lib64", "bin", "build"] {
            let dir = dst.join(subdir);
            if dir.is_dir() {
                find_libs_with_filter(&dir, &mut shared_libs, &is_shared_lib);
            }
        }
        let mut link_dirs = std::collections::BTreeSet::new();
        for lib in &shared_libs {
            if let Some(parent) = lib.parent() {
                link_dirs.insert(parent.to_path_buf());
            }
        }
        if target_os == "windows" {
            let mut import_libs = Vec::new();
            for subdir in ["lib", "lib64", "build"] {
                let dir = dst.join(subdir);
                if dir.is_dir() {
                    find_libs_with_filter(&dir, &mut import_libs, &is_static_lib);
                }
            }
            for lib in &import_libs {
                if let Some(parent) = lib.parent() {
                    link_dirs.insert(parent.to_path_buf());
                }
            }
        }
        for dir in &link_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }

        // Stage sibling dynamic backend plugins (`libggml-*.so`, `libllama.so`, etc.)
        // right next to Cargo's output binaries so `$ORIGIN` / `@loader_path` finds them.
        if let Ok(out_dir) = env::var("OUT_DIR") {
            let out_path = PathBuf::from(out_dir);
            if let Some(profile_dir) = out_path.ancestors().nth(3) {
                stage_shared_libs(&shared_libs, profile_dir);
                stage_shared_libs(&shared_libs, &profile_dir.join("deps"));
                stage_shared_libs(&shared_libs, &profile_dir.join("examples"));
            }
        }
        if let Ok(custom_stage) = env::var("LAYERSTREAM_BACKEND_STAGE_DIR")
            && !custom_stage.trim().is_empty()
        {
            stage_shared_libs(&shared_libs, Path::new(&custom_stage));
        }

        for name in ["llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={name}");
        }

        if target_os == "linux" {
            println!("cargo:rustc-link-arg-tests=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg-examples=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        } else if target_os == "macos" {
            println!("cargo:rustc-link-arg-tests=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg-examples=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-lib=c++");
        }
        return;
    }

    // Link static archives built by llama.cpp in dependency order.
    let mut lib_dirs = std::collections::BTreeSet::new();
    let mut libs = Vec::new();
    for subdir in ["lib", "lib64", "build"] {
        let dir = dst.join(subdir);
        if dir.is_dir() {
            find_libs_with_filter(&dir, &mut libs, &is_static_lib);
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
        let unix_archive = format!("lib{name}.a");
        let win_archive = format!("{name}.lib");
        if libs.iter().any(|p| {
            matches!(
                p.file_name().and_then(|f| f.to_str()),
                Some(f) if f == unix_archive || f == win_archive
            )
        }) {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }

    // Link platform & backend runtime libraries.
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
    } else if target_os == "windows" {
        if use_vulkan {
            println!("cargo:rustc-link-lib=vulkan-1");
        }
        if use_cuda {
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=cublasLt");
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
