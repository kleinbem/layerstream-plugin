#include "cuda_shim.hpp"
#include "engine_internal.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>


#ifdef _WIN32
#include <malloc.h>
#ifdef LAYERSTREAM_DYNAMIC_BACKENDS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#else
#ifdef LAYERSTREAM_DYNAMIC_BACKENDS
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif
#endif



namespace {
static std::once_flag g_llama_init_once;

#ifdef LAYERSTREAM_DYNAMIC_BACKENDS
static std::string parent_dir_of(const std::string & path) {
#ifdef _WIN32
    const size_t sep = path.find_last_of("\\/");
#else
    const size_t sep = path.find_last_of('/');
#endif
    if (sep == std::string::npos) {
        return {};
    }
    return path.substr(0, sep);
}

static std::string current_executable_dir() {
    std::string path;
#ifdef _WIN32
    char buf[4096];
    DWORD len = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (len > 0 && len < sizeof(buf)) {
        path.assign(buf, len);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size > 0) {
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            path.assign(buf.data());
        }
    }
#else
    char buf[4096];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        path.assign(buf, static_cast<size_t>(len));
    }
#endif
    return parent_dir_of(path);
}

static std::string current_library_dir() {
    std::string path;
#ifdef _WIN32
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&current_library_dir),
            &hm
        )) {
        char buf[4096];
        DWORD len = GetModuleFileNameA(hm, buf, sizeof(buf));
        if (len > 0 && len < sizeof(buf)) {
            path.assign(buf, len);
        }
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&current_library_dir), &info) != 0 && info.dli_fname) {
        path.assign(info.dli_fname);
    }
#endif
    return parent_dir_of(path);
}

static bool dir_exists(const std::string & path) {
    if (path.empty()) {
        return false;
    }
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

static bool file_exists(const std::string & path) {
    if (path.empty()) {
        return false;
    }
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
#else
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

static bool has_registered_gpu_device() {
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto kind = ggml_backend_dev_type(dev);
        if (kind == GGML_BACKEND_DEVICE_TYPE_GPU || kind == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

static bool has_registered_cpu_device() {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) != nullptr;
}

static bool load_best_cpu_backend_in_dir(const std::string & dir) {
    namespace fs = std::filesystem;
#ifdef _WIN32
    constexpr std::string_view kVariantPrefix = "ggml-cpu-";
    constexpr std::string_view kPlainName = "ggml-cpu.dll";
    constexpr std::string_view kExt = ".dll";
#elif defined(__APPLE__)
    constexpr std::string_view kVariantPrefix = "libggml-cpu-";
    constexpr std::string_view kPlainName = "libggml-cpu.dylib";
    constexpr std::string_view kExt = ".dylib";
#else
    constexpr std::string_view kVariantPrefix = "libggml-cpu-";
    constexpr std::string_view kPlainName = "libggml-cpu.so";
    constexpr std::string_view kExt = ".so";
#endif
    std::error_code ec;
    int best_score = 0;
    std::string best_path;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) {
            continue;
        }
        const std::string fname = it->path().filename().string();
        if (!fname.starts_with(kVariantPrefix) || !fname.ends_with(kExt)) {
            continue;
        }
        const std::string full = it->path().string();
#ifdef _WIN32
        HMODULE handle = LoadLibraryA(full.c_str());
        if (!handle) {
            continue;
        }
        auto score_fn = reinterpret_cast<int (*)()>(GetProcAddress(handle, "ggml_backend_score"));
        if (score_fn) {
            const int s = score_fn();
            if (s > best_score) {
                best_score = s;
                best_path = full;
            }
        }
        FreeLibrary(handle);
#else
        void * handle = dlopen(full.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            continue;
        }
        auto score_fn = reinterpret_cast<int (*)()>(dlsym(handle, "ggml_backend_score"));
        if (score_fn) {
            const int s = score_fn();
            if (s > best_score) {
                best_score = s;
                best_path = full;
            }
        }
        dlclose(handle);
#endif
    }
    if (best_score > 0 && !best_path.empty()) {
        return ggml_backend_load(best_path.c_str()) != nullptr;
    }
    const fs::path plain = fs::path(dir) / kPlainName;
    if (fs::is_regular_file(plain, ec)) {
        return ggml_backend_load(plain.string().c_str()) != nullptr;
    }
    return false;
}

static void append_unique_dir(std::vector<std::string> & dirs, const std::string & candidate) {
    if (candidate.empty() || !dir_exists(candidate)) {
        return;
    }
    for (const auto & existing : dirs) {
        if (existing == candidate) {
            return;
        }
    }
    dirs.push_back(candidate);
}

static void append_env_path_list(std::vector<std::string> & dirs, const char * env_name) {
    const char * raw = std::getenv(env_name);
    if (!raw || raw[0] == '\0') {
        return;
    }
#ifdef _WIN32
    constexpr char kSep = ';';
#else
    constexpr char kSep = ':';
#endif
    std::string s(raw);
    size_t start = 0;
    while (start <= s.size()) {
        const size_t end = s.find(kSep, start);
        const std::string token = (end == std::string::npos) ? s.substr(start) : s.substr(start, end - start);
        if (!token.empty()) {
            append_unique_dir(dirs, token);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}
#endif

static std::atomic<bool> g_log_verbose{true};

static void ls_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) {
        return;
    }
    if (!g_log_verbose.load(std::memory_order_relaxed)) {
        if (level != GGML_LOG_LEVEL_ERROR && level != GGML_LOG_LEVEL_WARN) {
            return;
        }
    }
    std::fputs(text, stderr);
}

}  // namespace

namespace ls_internal {

void ensure_llama_initialized() {
    std::call_once(g_llama_init_once, []() {
        // Workaround: CUDA graphs hang sm_120 channels under sustained load and churn on sparse MoE
        // pointer updates (tracking: kleinbem/layerstream-plugin#5, upstream: ggml-org/llama.cpp#27330).
#ifdef _WIN32
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
#ifdef LAYERSTREAM_DYNAMIC_BACKENDS
        std::vector<std::string> search_dirs;
        append_env_path_list(search_dirs, "LAYERSTREAM_BACKEND_DIRS");
        if (const char * custom_dir = std::getenv("LAYERSTREAM_BACKEND_DIR");
            custom_dir && custom_dir[0] != '\0') {
            append_unique_dir(search_dirs, custom_dir);
        }
        if (const char * mesh_dir = std::getenv("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR");
            mesh_dir && mesh_dir[0] != '\0') {
            append_unique_dir(search_dirs, mesh_dir);
        }
        const std::string lib_dir = current_library_dir();
        if (!lib_dir.empty()) {
#ifdef _WIN32
            append_unique_dir(search_dirs, lib_dir + "\\_backends");
#else
            append_unique_dir(search_dirs, lib_dir + "/_backends");
#endif
            append_unique_dir(search_dirs, lib_dir);
        }
        const std::string exe_dir = current_executable_dir();
        if (!exe_dir.empty()) {
            append_unique_dir(search_dirs, exe_dir);
        }

        // 1. Probe highest-priority CUDA / Metal plugins across all candidate directories first
        //    so CUDA always wins over Vulkan regardless of directory order, without initializing
        //    a redundant Vulkan context on the same GPU.
        for (const auto & dir : search_dirs) {
#ifdef _WIN32
            const char * hi_plugins[] = {"\\ggml-cuda.dll"};
#elif defined(__APPLE__)
            const char * hi_plugins[] = {"/libggml-metal.dylib", "/libggml-metal.so"};
#else
            const char * hi_plugins[] = {"/libggml-cuda.so"};
#endif
            for (const char * suffix : hi_plugins) {
                const std::string plugin = dir + suffix;
                if (file_exists(plugin)) {
                    ggml_backend_load(plugin.c_str());
                }
            }
            if (has_registered_gpu_device()) {
                break;
            }
        }
        // 2. Fall back to Vulkan / remaining GPU plugins only if no CUDA/Metal device registered.
        if (!has_registered_gpu_device()) {
            for (const auto & dir : search_dirs) {
                ggml_backend_load_all_from_path(dir.c_str());
                if (has_registered_gpu_device()) {
                    break;
                }
            }
        }
        // 3. Ensure the highest-scoring SIMD CPU backend variant is registered for embeddings / host fallbacks.
        if (!has_registered_cpu_device()) {
            for (const auto & dir : search_dirs) {
                load_best_cpu_backend_in_dir(dir);
                if (has_registered_cpu_device()) {
                    break;
                }
            }
        }
        if (ggml_backend_dev_count() == 0) {
            ggml_backend_load_all();
        }
#endif
        llama_backend_init();
    });
}

void set_error(char * buf, size_t cap, const std::string & msg) {
    if (!buf || cap == 0) {
        return;
    }
    const size_t n = std::min(cap - 1, msg.size());
    std::memcpy(buf, msg.data(), n);
    buf[n] = '\0';
}

int gpu_device_priority(ggml_backend_dev_t dev) {
    if (!dev) {
        return 0;
    }
    const char * name = ggml_backend_dev_name(dev);
    if (!name) {
        return 1;
    }
    if (std::strstr(name, "CUDA") != nullptr ||
        std::strstr(name, "MTL") != nullptr ||
        std::strstr(name, "Metal") != nullptr ||
        std::strstr(name, "ROCm") != nullptr ||
        std::strstr(name, "HIP") != nullptr) {
        return 3;
    }
    if (std::strstr(name, "Vulkan") != nullptr) {
        return 2;
    }
    return 1;
}

ggml_backend_dev_t select_primary_gpu_dev() {
    const size_t n_dev = ggml_backend_dev_count();
    ggml_backend_dev_t chosen = nullptr;
    int best_prio = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto kind = ggml_backend_dev_type(dev);
        if (kind == GGML_BACKEND_DEVICE_TYPE_GPU || kind == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            const int prio = gpu_device_priority(dev);
            if (prio > best_prio) {
                chosen = dev;
                best_prio = prio;
            }
        }
    }
    return chosen;
}

bool select_target_buft(
    bool use_gpu,
    ggml_backend_dev_t & out_gpu_dev,
    ggml_backend_buffer_type_t & out_buft,
    bool & out_using_gpu,
    bool & out_is_uma_zero_copy,
    bool & out_is_unified_memory,
    bool & out_is_cuda_backend,
    std::string & out_backend_name,
    std::string & out_err
) {
    out_gpu_dev = nullptr;
    out_using_gpu = false;
    out_is_uma_zero_copy = true;
    out_is_unified_memory = true;
    out_is_cuda_backend = false;
    out_backend_name = "CPU";
    out_err.clear();

    if (use_gpu) {
        ggml_backend_dev_t chosen = select_primary_gpu_dev();
        if (chosen == nullptr) {
            out_err =
                "no usable GPU device found (CUDA/Vulkan/Metal); verify GPU drivers "
                "(on NixOS CUDA ensure /run/opengl-driver/lib is reachable) or build with the matching GPU feature";
            return false;
        }

        out_gpu_dev = chosen;
        out_using_gpu = true;
        out_buft = ggml_backend_dev_buffer_type(chosen);
        out_is_uma_zero_copy = ggml_backend_buft_is_host(out_buft);
        out_is_unified_memory = (ggml_backend_dev_type(chosen) == GGML_BACKEND_DEVICE_TYPE_IGPU);
        const char * dev_name = ggml_backend_dev_name(chosen);
        out_backend_name = (dev_name != nullptr && dev_name[0] != '\0') ? dev_name : "GPU";
        out_is_cuda_backend = (dev_name != nullptr && std::strstr(dev_name, "CUDA") != nullptr);

#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (out_is_cuda_backend && ls_cuda::ensure_loaded()) {
            int integrated = 0;
            if (ls_cuda::device_get_attribute(&integrated, ls_cuda::kDevAttrIntegrated, 0) == ls_cuda::kSuccess &&
                integrated != 0) {
                out_is_unified_memory = true;
                ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(chosen);
                if (host_buft != nullptr && ggml_backend_dev_supports_buft(chosen, host_buft)) {
                    out_buft = host_buft;
                    out_is_uma_zero_copy = true;
                }
            }
        }
#endif
        return out_buft != nullptr;
    }

    out_buft = ggml_backend_cpu_buffer_type();
    out_using_gpu = false;
    out_is_uma_zero_copy = true;
    out_is_unified_memory = true;
    out_is_cuda_backend = false;
    out_backend_name = "CPU";
    return out_buft != nullptr;
}

}  // namespace ls_internal

void ls_set_log_verbose(bool verbose) {
    g_log_verbose.store(verbose, std::memory_order_relaxed);
    llama_log_set(ls_log_callback, nullptr);
    ggml_log_set(ls_log_callback, nullptr);
}

bool ls_probe_gpu_device(
    bool use_gpu,
    uint64_t * out_free_bytes,
    uint64_t * out_total_bytes,
    char * out_backend_name,
    size_t backend_name_cap
) {
    if (out_free_bytes) {
        *out_free_bytes = 0;
    }
    if (out_total_bytes) {
        *out_total_bytes = 0;
    }
    if (out_backend_name && backend_name_cap > 0) {
        out_backend_name[0] = '\0';
    }
    if (!use_gpu) {
        return false;
    }
    ls_internal::ensure_llama_initialized();
    ggml_backend_dev_t dev = ls_internal::select_primary_gpu_dev();
    if (!dev) {
        return false;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    if (total_b == 0) {
        return false;
    }
    if (out_free_bytes) {
        *out_free_bytes = static_cast<uint64_t>(free_b);
    }
    if (out_total_bytes) {
        *out_total_bytes = static_cast<uint64_t>(total_b);
    }
    if (out_backend_name && backend_name_cap > 0) {
        const char * name = ggml_backend_dev_name(dev);
        const char * eff = (name && name[0] != '\0') ? name : "GPU";
        const size_t n = std::min(backend_name_cap - 1, std::strlen(eff));
        std::memcpy(out_backend_name, eff, n);
        out_backend_name[n] = '\0';
    }
    return true;
}

bool ls_probe_gpu_memory(bool use_gpu, uint64_t * out_free_bytes, uint64_t * out_total_bytes) {
    return ls_probe_gpu_device(use_gpu, out_free_bytes, out_total_bytes, nullptr, 0);
}
