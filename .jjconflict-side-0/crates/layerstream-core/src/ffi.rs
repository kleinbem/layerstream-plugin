use std::ffi::{c_char, c_int, c_void};

#[repr(C)]
pub struct LsEngine {
    _private: [u8; 0],
}

pub use crate::sampler::{
    LsGrammar, ls_engine_eval_greedy, ls_engine_eval_sample, ls_engine_eval_verify_greedy,
    ls_engine_eval_verify_sample, ls_engine_eval_verify_with_grammar, ls_engine_eval_with_grammar,
    ls_grammar_accept, ls_grammar_create, ls_grammar_free, ls_sample_from_logits,
    ls_sample_from_logits_with_history, ls_verify_sample_from_logits,
    ls_verify_sample_from_logits_with_history,
};

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

pub type LsSpanSlice = gguf_direct_io::DirectSpanSlice;

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
    pub rs_rollback_max: u32,
    pub mtp_layers: u32,
    pub backend_name: [c_char; 32],
}

impl LsEngineStats {
    pub fn backend_name_str(&self) -> String {
        // SAFETY: `self.backend_name` is a fixed-size `[c_char; 32]` array; reinterpreting
        // its initialized bytes as `&[u8]` of the same length is valid for any `c_char` bit pattern.
        let raw = unsafe {
            std::slice::from_raw_parts(
                self.backend_name.as_ptr().cast::<u8>(),
                self.backend_name.len(),
            )
        };
        let nul = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        if nul == 0 {
            if self.using_gpu {
                "GPU".to_string()
            } else {
                "CPU".to_string()
            }
        } else {
            String::from_utf8_lossy(&raw[..nul]).into_owned()
        }
    }
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
    pub min_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub penalty_last_n: i32,
    pub seed: u32,
}

impl Default for LsSampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 64,
            seed: 42,
        }
    }
}

impl LsSampleParams {
    pub fn has_penalties(&self) -> bool {
        if self.penalty_last_n == 0 {
            return false;
        }
        let has_rep = self.repeat_penalty.is_finite()
            && self.repeat_penalty > 0.0
            && (self.repeat_penalty - 1.0).abs() > 1e-6;
        let has_freq = self.frequency_penalty.is_finite() && self.frequency_penalty.abs() > 1e-6;
        let has_pres = self.presence_penalty.is_finite() && self.presence_penalty.abs() > 1e-6;
        has_rep || has_freq || has_pres
    }

    /// Validates all floating-point sampling parameters consistently across server, Python, and Node.
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be a finite non-negative number");
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0.0, 1.0]");
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err("min_p must be in [0.0, 1.0]");
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err("repeat_penalty must be a finite positive number");
        }
        if !self.frequency_penalty.is_finite() {
            return Err("frequency_penalty must be finite");
        }
        if !self.presence_penalty.is_finite() {
            return Err("presence_penalty must be finite");
        }
        if self.penalty_last_n < -1 {
            return Err("penalty_last_n must be >= -1 (-1 = full context, 0 = disabled)");
        }
        Ok(())
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

    pub fn ls_probe_gpu_device(
        use_gpu: bool,
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
        out_backend_name: *mut c_char,
        backend_name_cap: usize,
    ) -> bool;

    pub fn ls_set_log_verbose(verbose: bool);

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

    pub fn ls_engine_mtp_draft(
        eng: *mut LsEngine,
        last_token: i32,
        pos0: i32,
        max_draft: i32,
        out_tokens: *mut i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> i32;

    pub fn ls_engine_mtp_accept(eng: *mut LsEngine, n_accepted: i32);

    pub fn ls_engine_save_state(
        eng: *mut LsEngine,
        path: *const c_char,
        tokens: *const i32,
        n_tokens: usize,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;

    pub fn ls_engine_load_state(
        eng: *mut LsEngine,
        path: *const c_char,
        out_tokens: *mut i32,
        cap_tokens: usize,
        out_n_tokens: *mut usize,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_params_validate_rejects_invalid_boundaries() {
        assert!(LsSampleParams::default().validate().is_ok());
        assert!(!LsSampleParams::default().has_penalties());

        let bad_cases = [
            LsSampleParams {
                temperature: -0.1,
                ..Default::default()
            },
            LsSampleParams {
                temperature: f32::NAN,
                ..Default::default()
            },
            LsSampleParams {
                top_p: 0.0,
                ..Default::default()
            },
            LsSampleParams {
                top_p: 1.01,
                ..Default::default()
            },
            LsSampleParams {
                min_p: -0.01,
                ..Default::default()
            },
            LsSampleParams {
                min_p: 1.5,
                ..Default::default()
            },
            LsSampleParams {
                repeat_penalty: 0.0,
                ..Default::default()
            },
            LsSampleParams {
                frequency_penalty: f32::INFINITY,
                ..Default::default()
            },
            LsSampleParams {
                presence_penalty: f32::NEG_INFINITY,
                ..Default::default()
            },
            LsSampleParams {
                penalty_last_n: -2,
                ..Default::default()
            },
        ];
        for p in bad_cases {
            assert!(
                p.validate().is_err(),
                "expected validation failure for {p:?}"
            );
        }

        let with_pen = LsSampleParams {
            repeat_penalty: 1.15,
            ..Default::default()
        };
        assert!(with_pen.validate().is_ok());
        assert!(with_pen.has_penalties());

        let disabled_pen = LsSampleParams {
            repeat_penalty: 1.15,
            penalty_last_n: 0,
            ..Default::default()
        };
        assert!(!disabled_pen.has_penalties());
    }

    #[test]
    fn backend_name_str_handles_empty_nul_and_full_buffers() {
        let mut stats = LsEngineStats::default();
        assert_eq!(stats.backend_name_str(), "CPU");
        stats.using_gpu = true;
        assert_eq!(stats.backend_name_str(), "GPU");

        for (i, b) in b"CUDA0\0".iter().enumerate() {
            stats.backend_name[i] = *b as c_char;
        }
        assert_eq!(stats.backend_name_str(), "CUDA0");

        stats.backend_name = [b'X' as c_char; 32];
        assert_eq!(stats.backend_name_str(), "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
    }

    #[test]
    fn gpu_probe_disabled_and_log_verbosity_toggle_work() {
        assert!(crate::StreamEngine::probe_gpu_device(false).is_none());
        assert!(crate::StreamEngine::probe_gpu_memory(false).is_none());
        crate::StreamEngine::set_log_verbose(false);
        crate::StreamEngine::set_log_verbose(true);
    }
}
