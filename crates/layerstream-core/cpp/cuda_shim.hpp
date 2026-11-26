#pragma once

#include <cstddef>
#include <type_traits>

#if defined(LAYERSTREAM_USE_CUDA)
#define LAYERSTREAM_HAS_CUDA_RT 1
#include <cuda_runtime.h>
#elif defined(LAYERSTREAM_USE_CUDA_DYNAMIC) && !defined(_WIN32)
#define LAYERSTREAM_HAS_CUDA_RT 1
#include <dlfcn.h>
#endif

#ifdef LAYERSTREAM_HAS_CUDA_RT
namespace ls_cuda {
#if defined(LAYERSTREAM_USE_CUDA)
using Stream = cudaStream_t;
using Event = cudaEvent_t;
static constexpr int kSuccess = static_cast<int>(cudaSuccess);
static constexpr int kMemcpyHostToDevice = static_cast<int>(cudaMemcpyHostToDevice);
static constexpr unsigned int kStreamNonBlocking = cudaStreamNonBlocking;
static constexpr unsigned int kEventDisableTiming = cudaEventDisableTiming;
static constexpr unsigned int kHostAllocDefault = cudaHostAllocDefault;
static constexpr int kDevAttrIntegrated = static_cast<int>(cudaDevAttrIntegrated);

inline bool ensure_loaded() { return true; }
inline int device_get_attribute(int * value, int attr, int device) {
    return static_cast<int>(cudaDeviceGetAttribute(value, static_cast<cudaDeviceAttr>(attr), device));
}
inline int stream_create_with_flags(Stream * s, unsigned int flags) {
    return static_cast<int>(cudaStreamCreateWithFlags(s, flags));
}
inline int stream_destroy(Stream s) { return static_cast<int>(cudaStreamDestroy(s)); }
inline int stream_synchronize(Stream s) { return static_cast<int>(cudaStreamSynchronize(s)); }
inline int event_create_with_flags(Event * e, unsigned int flags) {
    return static_cast<int>(cudaEventCreateWithFlags(e, flags));
}
inline int event_destroy(Event e) { return static_cast<int>(cudaEventDestroy(e)); }
inline int event_record(Event e, Stream s) { return static_cast<int>(cudaEventRecord(e, s)); }
inline int event_synchronize(Event e) { return static_cast<int>(cudaEventSynchronize(e)); }
inline int host_alloc(void ** ptr, size_t bytes, unsigned int flags) {
    return static_cast<int>(cudaHostAlloc(ptr, bytes, flags));
}
inline int free_host(void * ptr) { return static_cast<int>(cudaFreeHost(ptr)); }
inline int memcpy_async(void * dst, const void * src, size_t count, int kind, Stream s) {
    return static_cast<int>(cudaMemcpyAsync(dst, src, count, static_cast<cudaMemcpyKind>(kind), s));
}
inline int memset_async(void * dst, int value, size_t count, Stream s) {
    return static_cast<int>(cudaMemsetAsync(dst, value, count, s));
}
#else
using Stream = void *;
using Event = void *;
static constexpr int kSuccess = 0;
static constexpr int kMemcpyHostToDevice = 1;
static constexpr unsigned int kStreamNonBlocking = 0x01;
static constexpr unsigned int kEventDisableTiming = 0x02;
static constexpr unsigned int kHostAllocDefault = 0x00;
static constexpr int kDevAttrIntegrated = 18;

struct Api {
    int (*device_get_attribute)(int *, int, int) = nullptr;
    int (*stream_create_with_flags)(Stream *, unsigned int) = nullptr;
    int (*stream_destroy)(Stream) = nullptr;
    int (*stream_synchronize)(Stream) = nullptr;
    int (*event_create_with_flags)(Event *, unsigned int) = nullptr;
    int (*event_destroy)(Event) = nullptr;
    int (*event_record)(Event, Stream) = nullptr;
    int (*event_synchronize)(Event) = nullptr;
    int (*host_alloc)(void **, size_t, unsigned int) = nullptr;
    int (*free_host)(void *) = nullptr;
    int (*memcpy_async)(void *, const void *, size_t, int, Stream) = nullptr;
    int (*memset_async)(void *, int, size_t, Stream) = nullptr;
    bool ready = false;
};

static const Api & get_api() {
    static Api api = []() {
        Api out;
        static constexpr const char * kCandidates[] = {
            "libcudart.so.13",
            "libcudart.so",
        };
        void * handle = nullptr;
        for (const char * name : kCandidates) {
            handle = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
            if (handle) {
                break;
            }
        }
        if (!handle) {
            for (const char * name : kCandidates) {
                handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
                if (handle) {
                    break;
                }
            }
        }
        if (!handle) {
            return out;
        }
        auto load_sym = [&](auto & fn, const char * sym) -> bool {
            void * p = dlsym(handle, sym);
            if (!p) {
                return false;
            }
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
            return true;
        };
        out.ready = load_sym(out.device_get_attribute, "cudaDeviceGetAttribute") &&
                    load_sym(out.stream_create_with_flags, "cudaStreamCreateWithFlags") &&
                    load_sym(out.stream_destroy, "cudaStreamDestroy") &&
                    load_sym(out.stream_synchronize, "cudaStreamSynchronize") &&
                    load_sym(out.event_create_with_flags, "cudaEventCreateWithFlags") &&
                    load_sym(out.event_destroy, "cudaEventDestroy") &&
                    load_sym(out.event_record, "cudaEventRecord") &&
                    load_sym(out.event_synchronize, "cudaEventSynchronize") &&
                    load_sym(out.host_alloc, "cudaHostAlloc") &&
                    load_sym(out.free_host, "cudaFreeHost") &&
                    load_sym(out.memcpy_async, "cudaMemcpyAsync") &&
                    load_sym(out.memset_async, "cudaMemsetAsync");
        return out;
    }();
    return api;
}

inline bool ensure_loaded() { return get_api().ready; }
inline int device_get_attribute(int * value, int attr, int device) {
    const auto & a = get_api();
    return a.ready ? a.device_get_attribute(value, attr, device) : -1;
}
inline int stream_create_with_flags(Stream * s, unsigned int flags) {
    const auto & a = get_api();
    return a.ready ? a.stream_create_with_flags(s, flags) : -1;
}
inline int stream_destroy(Stream s) {
    const auto & a = get_api();
    return a.ready ? a.stream_destroy(s) : -1;
}
inline int stream_synchronize(Stream s) {
    const auto & a = get_api();
    return a.ready ? a.stream_synchronize(s) : -1;
}
inline int event_create_with_flags(Event * e, unsigned int flags) {
    const auto & a = get_api();
    return a.ready ? a.event_create_with_flags(e, flags) : -1;
}
inline int event_destroy(Event e) {
    const auto & a = get_api();
    return a.ready ? a.event_destroy(e) : -1;
}
inline int event_record(Event e, Stream s) {
    const auto & a = get_api();
    return a.ready ? a.event_record(e, s) : -1;
}
inline int event_synchronize(Event e) {
    const auto & a = get_api();
    return a.ready ? a.event_synchronize(e) : -1;
}
inline int host_alloc(void ** ptr, size_t bytes, unsigned int flags) {
    const auto & a = get_api();
    return a.ready ? a.host_alloc(ptr, bytes, flags) : -1;
}
inline int free_host(void * ptr) {
    const auto & a = get_api();
    return a.ready ? a.free_host(ptr) : -1;
}
inline int memcpy_async(void * dst, const void * src, size_t count, int kind, Stream s) {
    const auto & a = get_api();
    return a.ready ? a.memcpy_async(dst, src, count, kind, s) : -1;
}
inline int memset_async(void * dst, int value, size_t count, Stream s) {
    const auto & a = get_api();
    return a.ready ? a.memset_async(dst, value, count, s) : -1;
}
#endif
} // namespace ls_cuda
#endif
