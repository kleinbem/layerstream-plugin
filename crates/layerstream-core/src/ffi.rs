use std::ffi::{c_char, c_int, c_void};

#[repr(C)]
pub struct LsEngine {
    _private: [u8; 0],
}

pub type LsLoadTensorsCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    layer_idx: i32,
    tensor_names: *const *const c_char,
    tensor_offsets: *const u64,
    tensor_sizes: *const u64,
    tensor_count: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> c_int;

pub type LsPrefetchStartCb = LsLoadTensorsCb;
pub type LsPrefetchWaitCb = unsafe extern "C" fn(user_data: *mut c_void, layer_idx: i32) -> c_int;

#[repr(C)]
pub struct LsEngineConfig {
    pub model_path: *const c_char,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: i32,
    pub vram_budget_bytes: u64,
    pub kv_cache_type: i32,
    pub use_gpu: bool,
    pub full_resident_reference: bool,
    pub io_user_data: *mut c_void,
    pub load_tensors_cb: Option<LsLoadTensorsCb>,
    pub prefetch_start_cb: Option<LsPrefetchStartCb>,
    pub prefetch_wait_cb: Option<LsPrefetchWaitCb>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LsEngineStats {
    pub n_layers: u32,
    pub resident_layers: u32,
    pub streamed_layers: u32,
    pub resident_bytes: u64,
    pub slot_bytes: u64,
    pub total_layer_loads: u64,
    pub is_uma_zero_copy: bool,
    pub using_gpu: bool,
    pub can_speculate: bool,
}

#[repr(C)]
pub struct LsChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LsSampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub seed: u32,
}

impl Default for LsSampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.0,
            seed: 42,
        }
    }
}

unsafe extern "C" {
    pub fn ls_engine_create(
        cfg: *const LsEngineConfig,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> *mut LsEngine;

    pub fn ls_engine_free(eng: *mut LsEngine);

    pub fn ls_engine_get_stats(eng: *const LsEngine, out: *mut LsEngineStats);

    pub fn ls_engine_apply_chat_template(
        eng: *const LsEngine,
        messages: *const LsChatMessage,
        n_messages: usize,
        add_assistant_prompt: bool,
        out_buf: *mut c_char,
        out_cap: i32,
    ) -> i32;

    pub fn ls_engine_tokenize(
        eng: *const LsEngine,
        text: *const c_char,
        text_len: usize,
        out_tokens: *mut i32,
        max_tokens: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn ls_engine_token_to_piece(
        eng: *const LsEngine,
        token: i32,
        out_buf: *mut c_char,
        buf_cap: i32,
    ) -> i32;

    pub fn ls_engine_token_is_eog(eng: *const LsEngine, token: i32) -> bool;

    pub fn ls_engine_kv_clear(eng: *mut LsEngine);

    pub fn ls_engine_kv_trim(eng: *mut LsEngine, pos_min: i32) -> bool;

    pub fn ls_engine_eval_greedy(
        eng: *mut LsEngine,
        tokens: *const i32,
        n_tokens: i32,
        pos_start: i32,
        out_next_token: *mut i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;

    pub fn ls_engine_eval_sample(
        eng: *mut LsEngine,
        tokens: *const i32,
        n_tokens: i32,
        pos_start: i32,
        sample_params: *const LsSampleParams,
        out_next_token: *mut i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;

    pub fn ls_engine_eval_verify_greedy(
        eng: *mut LsEngine,
        tokens: *const i32,
        n_tokens: i32,
        pos_start: i32,
        out_argmax_tokens: *mut i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}
