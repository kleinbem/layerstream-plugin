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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LsSpanSlice {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub len: u64,
    pub tensor_idx: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LsTensorSliceReq {
    pub tensor_name: *const c_char,
    pub src_offset_in_tensor: u64,
    pub dst_offset: u64,
    pub len: u64,
    pub tensor_idx: u32,
}

pub type LsLoadSlicesCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    reqs: *const LsTensorSliceReq,
    req_count: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> c_int;

pub type LsLoadLayerDirectCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    layer_idx: i32,
    tensor_names: *const *const c_char,
    tensor_offsets: *const u64,
    tensor_sizes: *const u64,
    tensor_count: usize,
    pin_buf: *mut u8,
    pin_cap: usize,
    out_slices: *mut LsSpanSlice,
    max_slices: usize,
) -> c_int;

#[repr(C)]
pub struct LsEngineConfig {
    pub model_path: *const c_char,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: i32,
    pub vram_budget_bytes: u64,
    pub vram_budget_auto: bool,
    pub kv_cache_type: i32,
    pub use_gpu: bool,
    pub full_resident_reference: bool,
    pub cpu_embeddings: bool,
    pub cpu_head: bool,
    pub stream_head: bool,
    pub moe_active_streaming: bool,
    pub io_user_data: *mut c_void,
    pub load_tensors_cb: Option<LsLoadTensorsCb>,
    pub load_layer_direct_cb: Option<LsLoadLayerDirectCb>,
    pub load_slices_cb: Option<LsLoadSlicesCb>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LsEngineStats {
    pub n_layers: u32,
    pub resident_layers: u32,
    pub streamed_layers: u32,
    pub resident_bytes: u64,
    pub cpu_resident_bytes: u64,
    pub slot_bytes: u64,
    pub expert_slot_bytes: u64,
    pub total_layer_loads: u64,
    pub moe_layers: u32,
    pub moe_experts_loaded: u64,
    pub moe_experts_total: u64,
    pub head_streamed: bool,
    pub moe_active_streaming: bool,
    pub is_uma_zero_copy: bool,
    pub is_unified_memory: bool,
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
    pub seed: u32,
}

impl Default for LsSampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
        }
    }
}

pub type LsCancelCheckCb = unsafe extern "C" fn(user_data: *const c_void) -> bool;

unsafe extern "C" {
    pub fn ls_engine_create(
        cfg: *const LsEngineConfig,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> *mut LsEngine;

    pub fn ls_engine_free(eng: *mut LsEngine);

    pub fn ls_engine_get_stats(eng: *const LsEngine, out: *mut LsEngineStats);

    pub fn ls_engine_vocab_size(eng: *const LsEngine) -> i32;

    pub fn ls_engine_n_ctx(eng: *const LsEngine) -> u32;

    pub fn ls_engine_set_cancel(eng: *mut LsEngine, cancel: bool);

    pub fn ls_engine_set_cancel_cb(
        eng: *mut LsEngine,
        cb: Option<LsCancelCheckCb>,
        user_data: *const c_void,
    );

    pub fn ls_probe_gpu_memory(
        use_gpu: bool,
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
    ) -> bool;

    pub fn ls_sample_from_logits(
        logits: *const f32,
        n_vocab: i32,
        sample_params: *const LsSampleParams,
    ) -> i32;

    pub fn ls_verify_sample_from_logits(
        logits_per_pos: *const *const f32,
        n_vocab: i32,
        draft_tokens: *const i32,
        k: i32,
        sample_params: *const LsSampleParams,
        out_accepted_count: *mut i32,
        out_next_token: *mut i32,
    );

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

    pub fn ls_engine_eval_verify_sample(
        eng: *mut LsEngine,
        tokens: *const i32,
        n_tokens: i32,
        pos_start: i32,
        sample_params: *const LsSampleParams,
        out_accepted_count: *mut i32,
        out_next_token: *mut i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}
