#ifndef LAYERSTREAM_H
#define LAYERSTREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a loaded LayerStream inference session. */
typedef struct LayerStreamEngine LayerStreamEngine;

/** Thread-safe cooperative cancellation token. */
typedef struct LayerStreamCancelToken LayerStreamCancelToken;

/**
 * Streaming token callback.
 *
 * @param chunk_utf8 Non-null UTF-8 byte slice for the emitted token piece.
 * @param chunk_len  Byte length of @p chunk_utf8 (not guaranteed NUL-terminated).
 * @param user_data  Opaque pointer forwarded from the caller.
 * @return true to continue generation, false to cancel early.
 */
typedef bool (*layerstream_token_cb)(
    const char *chunk_utf8,
    size_t chunk_len,
    void *user_data
);

/** Return the static NUL-terminated library version string (do NOT free). */
const char *layerstream_version(void);

/**
 * Return the thread-local last error message, or NULL if the last call succeeded.
 * The returned pointer is valid until the next layerstream call on the same thread.
 * Do NOT free the returned pointer.
 */
const char *layerstream_last_error(void);

/** Clear the thread-local last error message. */
void layerstream_clear_error(void);

/** Free a heap-allocated string returned by any layerstream_* function. Safe on NULL. */
void layerstream_string_free(char *ptr);

/** Probe available GPU memory/device and return a JSON string (free with layerstream_string_free). */
char *layerstream_probe_gpu_json(bool use_gpu);

/** Inspect a local GGUF file header and return a JSON string (free with layerstream_string_free). */
char *layerstream_inspect_gguf_json(const char *path);

/** Compile a JSON Schema string into a GBNF grammar string (free with layerstream_string_free). */
char *layerstream_json_schema_to_gbnf(const char *schema_json);

/** Return a GBNF grammar constraining output to any JSON object (free with layerstream_string_free). */
char *layerstream_json_object_gbnf(void);

/** Parse <think> reasoning and <tool_call> tags from raw assistant text into JSON. */
char *layerstream_parse_assistant_output_json(const char *raw_text);

/** Create a new cooperative cancellation token (free with layerstream_cancel_token_free). */
LayerStreamCancelToken *layerstream_cancel_token_new(void);

/** Signal cancellation on a token. Safe to call from any thread; no-op on NULL. */
void layerstream_cancel_token_cancel(const LayerStreamCancelToken *tok);

/** Free a cancellation token. Safe on NULL. */
void layerstream_cancel_token_free(LayerStreamCancelToken *tok);

/**
 * Open a GGUF model (local path or HuggingFace spec).
 *
 * @param model_spec     Non-null UTF-8 path or HuggingFace spec.
 * @param open_opts_json Optional UTF-8 JSON object with open options (may be NULL or "{}").
 * @return Engine handle on success, or NULL on error (see layerstream_last_error()).
 */
LayerStreamEngine *layerstream_engine_open(
    const char *model_spec,
    const char *open_opts_json
);

/** Free a loaded engine handle. Safe on NULL. */
void layerstream_engine_free(LayerStreamEngine *engine);

/** Return engine stats snapshot as JSON (free with layerstream_string_free). */
char *layerstream_engine_stats_json(const LayerStreamEngine *engine);

/**
 * Run non-streaming text generation.
 *
 * @return JSON object {"text": "...", "summary": {...}}, or NULL on error.
 */
char *layerstream_engine_generate(
    LayerStreamEngine *engine,
    const char *prompt,
    const char *gen_opts_json,
    const LayerStreamCancelToken *cancel_tok
);

/**
 * Run streaming text generation, invoking @p cb for each UTF-8 token piece.
 *
 * @return JSON summary object on completion, or NULL on error.
 */
char *layerstream_engine_generate_stream(
    LayerStreamEngine *engine,
    const char *prompt,
    const char *gen_opts_json,
    const LayerStreamCancelToken *cancel_tok,
    layerstream_token_cb cb,
    void *user_data
);

/**
 * Run non-streaming chat completion.
 *
 * @param messages_json JSON array of {"role": "...", "content": "..."} message objects.
 * @return JSON object {"text": "...", "parsed": {...}|null, "summary": {...}}, or NULL on error.
 */
char *layerstream_engine_chat(
    LayerStreamEngine *engine,
    const char *messages_json,
    const char *gen_opts_json,
    const LayerStreamCancelToken *cancel_tok
);

/**
 * Run streaming chat completion, invoking @p cb for each UTF-8 token piece.
 *
 * @return JSON summary object on completion, or NULL on error.
 */
char *layerstream_engine_chat_stream(
    LayerStreamEngine *engine,
    const char *messages_json,
    const char *gen_opts_json,
    const LayerStreamCancelToken *cancel_tok,
    layerstream_token_cb cb,
    void *user_data
);

/** Clear the KV cache. Returns 0 on success, -1 on error. */
int layerstream_engine_kv_clear(LayerStreamEngine *engine);

/** Trim KV cache from pos_min onward. Returns 1 if trimmed, 0 if no-op, -1 on error. */
int layerstream_engine_kv_trim(LayerStreamEngine *engine, int32_t pos_min);

/** Save KV cache state to @p path. Returns bytes written (>= 0) on success, -1 on error. */
int64_t layerstream_engine_save_state(LayerStreamEngine *engine, const char *path);

/** Load KV cache state from @p path. Returns bytes read (>= 0) on success, -1 on error. */
int64_t layerstream_engine_load_state(LayerStreamEngine *engine, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* LAYERSTREAM_H */
