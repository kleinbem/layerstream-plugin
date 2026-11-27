use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffi::{
    LsChatMessage, LsEngine, LsEngineConfig, LsEngineStats, LsGrammar, LsSampleParams,
    ls_engine_apply_chat_template, ls_engine_create, ls_engine_eval_greedy, ls_engine_eval_sample,
    ls_engine_eval_verify_greedy, ls_engine_eval_verify_sample, ls_engine_eval_verify_with_grammar,
    ls_engine_eval_with_grammar, ls_engine_free, ls_engine_get_stats, ls_engine_kv_clear,
    ls_engine_kv_trim, ls_engine_load_state, ls_engine_mtp_accept, ls_engine_mtp_draft,
    ls_engine_n_ctx, ls_engine_save_state, ls_engine_set_cancel_cb, ls_engine_token_is_eog,
    ls_engine_token_to_piece, ls_engine_tokenize, ls_engine_vocab_size, ls_grammar_accept,
    ls_grammar_create, ls_grammar_free, ls_probe_gpu_device, ls_probe_gpu_memory,
    ls_set_log_verbose,
};
use crate::gguf_index::{GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA};
use crate::io_reader::{IoStats, LayerIoPool};
use crate::speculator::{DraftModelProposer, NgramProposer, SpeculationStats};

const RC_CANCELLED: c_int = -100;

#[derive(Debug, Clone)]
pub struct PromptTooLongError {
    pub prompt_tokens: usize,
    pub n_ctx: usize,
}

impl fmt::Display for PromptTooLongError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prompt has {} tokens, which exceeds context window n_ctx={} (need at least 1 token for generation)",
            self.prompt_tokens, self.n_ctx
        )
    }
}

impl std::error::Error for PromptTooLongError {}

#[derive(Debug, Clone, Copy)]
pub struct CancelledError;

impl fmt::Display for CancelledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "generation cancelled")
    }
}

impl std::error::Error for CancelledError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    #[default]
    #[serde(alias = "F16")]
    F16 = 0,
    #[serde(alias = "Q8_0")]
    Q8_0 = 1,
    #[serde(alias = "Q4_0")]
    Q4_0 = 2,
}

impl std::str::FromStr for KvCacheType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            other => bail!("invalid kv_cache_type '{other}' (expected f16, q8_0, or q4_0)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: Option<u32>,
    pub vram_budget_bytes: u64,
    /// Probe free GPU memory automatically and pick a safe hybrid residency budget.
    pub vram_budget_auto: bool,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
    pub full_resident_reference: bool,
    /// Pin `token_embd.weight` in CPU RAM instead of GPU VRAM.
    pub cpu_embeddings: bool,
    /// Pin `output.weight` (LM head) in CPU RAM instead of GPU VRAM.
    pub cpu_head: bool,
    /// Stream `output.weight` through ping-pong GPU slots when <= slot_bytes (else CPU fallback).
    pub stream_head: bool,
    /// Enable per-expert active streaming for sparse MoE layers (default true).
    pub moe_active_streaming: bool,
    /// Optional explicit I/O worker thread count (defaults to profile/env or 1).
    pub io_threads: Option<usize>,
    /// Optional explicit TinyLFU hot-expert RAM cache ceiling in bytes (None = RAM-tier auto).
    pub expert_cache_bytes: Option<usize>,
}

impl EngineConfig {
    pub fn streaming(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            n_ctx: 4096,
            n_batch: 512,
            // Default to 2 CPU threads when GPU offloading is enabled so host cores stay free
            // for NVMe/dm-crypt BIO completion and HTTP serving.
            n_threads: 2,
            resident_layers: None,
            vram_budget_bytes: 0,
            vram_budget_auto: false,
            kv_cache_type: KvCacheType::F16,
            use_gpu: true,
            prefer_direct_io: true,
            full_resident_reference: false,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
            io_threads: None,
            expert_cache_bytes: None,
        }
    }

    pub fn reference(model_path: impl AsRef<Path>) -> Self {
        Self {
            full_resident_reference: true,
            ..Self::streaming(model_path)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub sample: LsSampleParams,
    pub stop: Vec<String>,
    pub enable_ngram_speculation: bool,
    pub max_draft_tokens: usize,
    /// Optional GBNF grammar (with `root` start rule) constraining generated output.
    pub grammar: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sample: LsSampleParams::default(),
            stop: Vec::new(),
            enable_ngram_speculation: true,
            max_draft_tokens: 5,
            grammar: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateSummary {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub speculation: SpeculationStats,
}

pub struct StreamEngine {
    raw: NonNull<LsEngine>,
    /// Raw heap-allocated `LayerIoPool` owned by `StreamEngine` via `Box::into_raw` so
    /// moving `StreamEngine` never invalidates the C++ callback pointer under Stacked Borrows.
    /// Reconstructed and dropped in `Drop` strictly AFTER `ls_engine_free` joins the I/O worker.
    io_pool: Option<*mut LayerIoPool>,
    n_batch: usize,
    cached_kv_tokens: Vec<i32>,
    expert_profile_path: Option<PathBuf>,
}

fn default_expert_profile_path(model_path: &Path) -> Option<PathBuf> {
    if std::env::var("LAYERSTREAM_DISABLE_EXPERT_PROFILE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return None;
    }
    let meta = std::fs::metadata(model_path).ok()?;
    let stem = model_path.file_name()?.to_string_lossy();
    let cache_base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(
        cache_base
            .join("layerstream")
            .join("expert_profiles")
            .join(format!("{stem}-{}.lsep", meta.len())),
    )
}

// SAFETY: `StreamEngine` owns its C++ `LsEngine` handle and `LayerIoPool` exclusively; all mutable
// operations require `&mut self` (or external synchronization), so moving it across threads is sound.
unsafe impl Send for StreamEngine {}

impl StreamEngine {
    pub fn open(cfg: &EngineConfig) -> Result<Self> {
        #[cfg(unix)]
        let path_bytes = std::os::unix::ffi::OsStrExt::as_bytes(cfg.model_path.as_os_str());
        #[cfg(not(unix))]
        let path_lossy = cfg.model_path.to_string_lossy();
        #[cfg(not(unix))]
        let path_bytes = path_lossy.as_bytes();
        let c_path = CString::new(path_bytes).context("model_path contains interior NUL")?;

        let io_pool_raw: Option<*mut LayerIoPool> = if cfg.full_resident_reference {
            None
        } else {
            let index = GgufIndex::open(&cfg.model_path)?;
            let pool = LayerIoPool::new_with_options(
                index,
                cfg.prefer_direct_io,
                cfg.io_threads,
                cfg.expert_cache_bytes,
            )?;
            Some(Box::into_raw(Box::new(pool)))
        };

        let io_ptr = io_pool_raw
            .map(|p| p.cast::<c_void>())
            .unwrap_or(std::ptr::null_mut());

        let raw_cfg = LsEngineConfig {
            model_path: c_path.as_ptr(),
            n_ctx: cfg.n_ctx,
            n_batch: cfg.n_batch,
            n_threads: cfg.n_threads,
            resident_layers: cfg.resident_layers.map_or(-1, |v| v as i32),
            vram_budget_bytes: cfg.vram_budget_bytes,
            vram_budget_auto: cfg.vram_budget_auto,
            kv_cache_type: cfg.kv_cache_type as i32,
            use_gpu: cfg.use_gpu,
            full_resident_reference: cfg.full_resident_reference,
            cpu_embeddings: cfg.cpu_embeddings,
            cpu_head: cfg.cpu_head,
            stream_head: cfg.stream_head,
            moe_active_streaming: cfg.moe_active_streaming,
            io_user_data: io_ptr,
            load_tensors_cb: if io_pool_raw.is_some() {
                Some(rust_load_tensors_cb)
            } else {
                None
            },
            load_layer_direct_cb: if io_pool_raw.is_some() {
                Some(rust_load_layer_direct_cb)
            } else {
                None
            },
            load_slices_cb: if io_pool_raw.is_some() {
                Some(rust_load_slices_cb)
            } else {
                None
            },
        };

        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `raw_cfg` fields and C strings live for the duration of `ls_engine_create`, and `err_buf` is valid for `err_buf.len()` bytes.
        let ptr = unsafe { ls_engine_create(&raw_cfg, err_buf.as_mut_ptr(), err_buf.len()) };
        let Some(raw) = NonNull::new(ptr) else {
            if let Some(p) = io_pool_raw {
                // SAFETY: `p` was created via `Box::into_raw` in `StreamEngine::open` and is reclaimed exactly once after C++ no longer references it.
                drop(unsafe { Box::from_raw(p) });
            }
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("failed to create layerstream engine: {msg}");
        };

        let engine = Self {
            raw,
            io_pool: io_pool_raw,
            n_batch: cfg.n_batch.max(1) as usize,
            cached_kv_tokens: Vec::new(),
            expert_profile_path: None,
        };

        let mut engine = engine;
        let st = engine.stats();
        if st.moe_active_streaming && st.moe_layers > 0 {
            engine.expert_profile_path = default_expert_profile_path(&cfg.model_path);
            if let Some(ref prof) = engine.expert_profile_path
                && prof.is_file()
                && let Ok(warmed) = engine.warm_expert_profile(prof)
                && warmed > 0
            {
                tracing::info!(
                    warmed_slices = warmed,
                    path = %prof.display(),
                    "pre-warmed MoE hot-expert cache from saved profile"
                );
            }
        }

        Ok(engine)
    }

    pub fn stats(&self) -> LsEngineStats {
        let mut out = LsEngineStats::default();
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        unsafe { ls_engine_get_stats(self.raw.as_ptr(), &mut out) };
        out
    }

    pub fn io_stats(&self) -> IoStats {
        self.io_pool
            // SAFETY: `io_pool_ptr` is non-null, initialized from `Box::into_raw`, and remains valid until `StreamEngine::drop`.
            .map(|p| unsafe { &*p }.stats())
            .unwrap_or_default()
    }

    pub fn clear_expert_cache(&self) {
        if let Some(p) = self.io_pool {
            // SAFETY: `io_pool_ptr` is non-null, initialized from `Box::into_raw`, and remains valid until `StreamEngine::drop`.
            unsafe { &*p }.clear_expert_cache();
        }
    }

    /// Saves the currently resident MoE hot-expert slices to `path` for zero-cold-start pre-warming.
    pub fn save_expert_profile(&self, path: &Path) -> std::io::Result<usize> {
        let Some(p) = self.io_pool else {
            return Ok(0);
        };
        // SAFETY: `io_pool` is non-null, initialized from `Box::into_raw`, and remains valid until `StreamEngine::drop`.
        unsafe { &*p }.save_expert_profile(path)
    }

    /// Sequentially pre-warms the MoE expert cache from a saved hot-expert profile at `path`.
    pub fn warm_expert_profile(&self, path: &Path) -> std::io::Result<usize> {
        let Some(p) = self.io_pool else {
            return Ok(0);
        };
        // SAFETY: `io_pool` is non-null, initialized from `Box::into_raw`, and remains valid until `StreamEngine::drop`.
        unsafe { &*p }.warm_expert_profile(path)
    }

    pub fn vocab_size(&self) -> i32 {
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        unsafe { ls_engine_vocab_size(self.raw.as_ptr()) }
    }

    pub fn n_ctx(&self) -> u32 {
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        unsafe { ls_engine_n_ctx(self.raw.as_ptr()) }
    }

    /// Probes available and total memory `(free_bytes, total_bytes)` on the primary GPU backend.
    pub fn probe_gpu_memory(use_gpu: bool) -> Option<(u64, u64)> {
        let mut free_b = 0u64;
        let mut total_b = 0u64;
        // SAFETY: Output pointers and buffers are valid, properly aligned stack variables for the duration of the FFI probe.
        let ok = unsafe { ls_probe_gpu_memory(use_gpu, &mut free_b, &mut total_b) };
        if ok && total_b > 0 {
            Some((free_b, total_b))
        } else {
            None
        }
    }

    /// Probes `(free_bytes, total_bytes, backend_name)` on the primary GPU backend.
    pub fn probe_gpu_device(use_gpu: bool) -> Option<(u64, u64, String)> {
        let mut free_b = 0u64;
        let mut total_b = 0u64;
        let mut name_buf = [0 as c_char; 32];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let ok = unsafe {
            ls_probe_gpu_device(
                use_gpu,
                &mut free_b,
                &mut total_b,
                name_buf.as_mut_ptr(),
                name_buf.len(),
            )
        };
        if ok && total_b > 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            Some((
                free_b,
                total_b,
                if name.is_empty() {
                    "GPU".to_string()
                } else {
                    name
                },
            ))
        } else {
            None
        }
    }

    /// Configures `llama.cpp` / `ggml` stderr verbosity (`false` silences INFO/DEBUG logs).
    pub fn set_log_verbose(verbose: bool) {
        // SAFETY: `ls_set_log_verbose` only updates an atomic/global verbosity flag in C++.
        unsafe { ls_set_log_verbose(verbose) };
    }

    /// Verifies that a draft model shares the same base tokenizer vocabulary and token IDs
    /// as this target model. Allows up to `MAX_DRAFT_VOCAB_PAD_DELTA` (256) trailing padded
    /// entries (e.g., Qwen2.5-14B/32B `152,064` vs `0.5B` `151,936`).
    pub fn check_draft_compatible(&self, draft: &StreamEngine) -> std::result::Result<(), String> {
        let target_v = self.vocab_size();
        let draft_v = draft.vocab_size();
        if target_v <= 0 || draft_v <= 0 {
            return Err(format!(
                "invalid vocabulary size (target={target_v}, draft={draft_v})"
            ));
        }
        let delta = (target_v as usize).abs_diff(draft_v as usize);
        if delta > MAX_DRAFT_VOCAB_PAD_DELTA {
            return Err(format!(
                "vocabulary size mismatch (target={target_v}, draft={draft_v}, delta={delta} > {MAX_DRAFT_VOCAB_PAD_DELTA})"
            ));
        }
        const PROBE: &str = "Hello, world! 12345\nfn main() { println!(\"ok\"); }";
        let target_ids = self
            .tokenize(PROBE, false, false)
            .map_err(|e| format!("target probe tokenize failed: {e}"))?;
        let draft_ids = draft
            .tokenize(PROBE, false, false)
            .map_err(|e| format!("draft probe tokenize failed: {e}"))?;
        if target_ids != draft_ids {
            return Err(format!(
                "tokenizer IDs diverged on canonical probe (target={:?}, draft={:?})",
                &target_ids[..target_ids.len().min(6)],
                &draft_ids[..draft_ids.len().min(6)],
            ));
        }
        let shared = target_v.min(draft_v);
        for id in [0, 1, shared / 4, shared / 2, (shared * 3) / 4, shared - 1] {
            if id < 0 || id >= shared {
                continue;
            }
            let tp = self.token_to_piece(id).unwrap_or_default();
            let dp = draft.token_to_piece(id).unwrap_or_default();
            if tp != dp {
                return Err(format!(
                    "boundary token {id} mismatch (target={tp:?}, draft={dp:?})"
                ));
            }
        }
        Ok(())
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_assistant_prompt: bool,
    ) -> Result<String> {
        let c_strings: Vec<(CString, CString)> = messages
            .iter()
            .map(|m| {
                let role = CString::new(m.role.as_bytes()).unwrap_or_else(|_| c"user".to_owned());
                let content = CString::new(m.content.replace('\0', ""))
                    .expect("NUL bytes already stripped from chat content");
                (role, content)
            })
            .collect();
        let raw_msgs: Vec<LsChatMessage> = c_strings
            .iter()
            .map(|(r, c)| LsChatMessage {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();

        let mut buf = vec![0u8; 4096];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let needed = unsafe {
            ls_engine_apply_chat_template(
                self.raw.as_ptr(),
                raw_msgs.as_ptr(),
                raw_msgs.len(),
                add_assistant_prompt,
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len() as i32,
            )
        };
        if needed < 0 {
            bail!("ls_engine_apply_chat_template failed (code={needed})");
        }
        let needed_usize = needed as usize;
        if needed_usize >= buf.len() {
            buf.resize(needed_usize + 1, 0);
            // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
            let n2 = unsafe {
                ls_engine_apply_chat_template(
                    self.raw.as_ptr(),
                    raw_msgs.as_ptr(),
                    raw_msgs.len(),
                    add_assistant_prompt,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                bail!("ls_engine_apply_chat_template resize failed (code={n2})");
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(needed_usize);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let bytes = text.as_bytes();
        let cap = (bytes.len() + 16).min(i32::MAX as usize) as i32;
        let mut out = vec![0i32; cap as usize];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let n = unsafe {
            ls_engine_tokenize(
                self.raw.as_ptr(),
                bytes.as_ptr().cast::<c_char>(),
                bytes.len(),
                out.as_mut_ptr(),
                cap,
                add_special,
                parse_special,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            out.resize(needed, 0);
            // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
            let n2 = unsafe {
                ls_engine_tokenize(
                    self.raw.as_ptr(),
                    bytes.as_ptr().cast::<c_char>(),
                    bytes.len(),
                    out.as_mut_ptr(),
                    needed as i32,
                    add_special,
                    parse_special,
                )
            };
            if n2 < 0 {
                bail!("llama_tokenize failed with code {n2}");
            }
            out.truncate(n2 as usize);
            return Ok(out);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    fn token_to_piece_bytes(&self, token: i32) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 64];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let n = unsafe {
            ls_engine_token_to_piece(
                self.raw.as_ptr(),
                token,
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len() as i32,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            buf.resize(needed, 0);
            // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
            let n2 = unsafe {
                ls_engine_token_to_piece(
                    self.raw.as_ptr(),
                    token,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                bail!("llama_token_to_piece failed for token {token}");
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        Ok(buf)
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        let buf = self.token_to_piece_bytes(token)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut out = String::with_capacity(tokens.len() * 4);
        let mut utf8 = Utf8PieceAssembler::default();
        for &tok in tokens {
            let bytes = self.token_to_piece_bytes(tok)?;
            out.push_str(&utf8.push(&bytes));
        }
        out.push_str(&utf8.flush());
        Ok(out)
    }

    pub fn token_is_eog(&self, token: i32) -> bool {
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        unsafe { ls_engine_token_is_eog(self.raw.as_ptr(), token) }
    }

    pub fn kv_clear(&mut self) {
        self.cached_kv_tokens.clear();
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        unsafe { ls_engine_kv_clear(self.raw.as_ptr()) };
    }

    pub fn kv_trim(&mut self, pos_min: i32) -> bool {
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let ok = unsafe { ls_engine_kv_trim(self.raw.as_ptr(), pos_min) };
        if ok {
            if pos_min <= 0 {
                self.cached_kv_tokens.clear();
            } else {
                self.cached_kv_tokens.truncate(pos_min as usize);
            }
        }
        ok
    }

    /// Returns the number of committed tokens currently tracked in the engine's KV prefix cache.
    pub fn cached_kv_len(&self) -> usize {
        self.cached_kv_tokens.len()
    }

    /// Drafts up to `max_draft` tokens using the target model's built-in MTP/NextN head(s).
    /// Returns an empty `Vec` when the loaded model has no MTP layers.
    pub fn mtp_draft(&mut self, last_token: i32, pos0: i32, max_draft: usize) -> Result<Vec<i32>> {
        if max_draft == 0 || self.stats().mtp_layers == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0i32; max_draft];
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle; `out` and `err_buf` are valid writable buffers.
        let n = unsafe {
            ls_engine_mtp_draft(
                self.raw.as_ptr(),
                last_token,
                pos0,
                max_draft as i32,
                out.as_mut_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if n < 0 {
            // SAFETY: C++ always NUL-terminates `err_buf` on error.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
            bail!("ls_engine_mtp_draft failed (rc={n}): {msg}");
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Advances the built-in MTP hidden-state cursor after `n_accepted` draft tokens were accepted.
    pub fn mtp_accept(&mut self, n_accepted: usize) {
        if self.stats().mtp_layers == 0 {
            return;
        }
        // SAFETY: `self.raw` is a live `LsEngine` handle.
        unsafe { ls_engine_mtp_accept(self.raw.as_ptr(), n_accepted as i32) };
    }
}

mod generate;
mod state;

pub use generate::StopFilter;

impl Drop for StreamEngine {
    fn drop(&mut self) {
        // Free C++ engine FIRST so `worker_thread` is joined before touching or freeing `LayerIoPool`.
        // SAFETY: `self.raw` is a valid engine handle owned exclusively by `self` and freed once on drop before reclaiming `io_pool_ptr`.
        unsafe { ls_engine_free(self.raw.as_ptr()) };
        if let Some(p) = self.io_pool.take() {
            // SAFETY: `p` was created via `Box::into_raw` in `StreamEngine::open` and is reclaimed exactly once after C++ no longer references it.
            let pool = unsafe { Box::from_raw(p) };
            if let Some(ref prof) = self.expert_profile_path {
                let st = pool.stats();
                if st.cache_hits + st.cache_misses > 0 {
                    let _ = pool.save_expert_profile(prof);
                }
            }
            drop(pool);
        }
    }
}

/// Routing-aware MTP break-even guard.
///
/// On dense models (`moe_layers == 0`), during cold-start (`hits + misses < 32`), or
/// when the MoE hot-expert cache hit rate is high (`>= 90%`), MTP runs at full
/// `base_cap`. When a sparse MoE is in the mid-range hit band (`45%..90%`), drafting
/// many extra tokens pulls in too many cold experts from NVMe per sweep, so draft
/// depth is capped to `1`. Below `45%`, it is capped to `2`.
pub fn effective_mtp_draft_cap(
    base_cap: usize,
    moe_layers: u32,
    cache_hits: u64,
    cache_misses: u64,
) -> usize {
    if base_cap == 0 || moe_layers == 0 {
        return base_cap;
    }
    let total = cache_hits.saturating_add(cache_misses);
    if total < 32 {
        return base_cap;
    }
    let hit_rate = cache_hits as f64 / total as f64;
    if hit_rate >= 0.90 {
        base_cap
    } else if hit_rate >= 0.45 {
        base_cap.min(1)
    } else {
        base_cap.min(2)
    }
}

mod callbacks;
use callbacks::{
    Utf8PieceAssembler, rust_load_layer_direct_cb, rust_load_slices_cb, rust_load_tensors_cb,
};

#[cfg(test)]
mod mtp_guard_tests {
    use super::*;

    #[test]
    fn effective_mtp_draft_cap_respects_dense_cold_and_moe_hit_bands() {
        assert_eq!(effective_mtp_draft_cap(0, 16, 80, 20), 0);
        assert_eq!(effective_mtp_draft_cap(4, 0, 50, 50), 4); // dense model
        assert_eq!(effective_mtp_draft_cap(4, 16, 10, 10), 4); // cold start (< 32 samples)
        assert_eq!(effective_mtp_draft_cap(4, 16, 95, 5), 4); // >= 90% hot cache -> full speed
        assert_eq!(effective_mtp_draft_cap(4, 16, 80, 20), 1); // 80% mid-band -> cap to 1
        assert_eq!(effective_mtp_draft_cap(4, 16, 20, 80), 2); // 20% low-band -> cap to 2

        let cfg = EngineConfig::streaming("demo.gguf");
        assert!(cfg.io_threads.is_none());
        assert!(cfg.expert_cache_bytes.is_none());
        let ref_cfg = EngineConfig::reference("demo.gguf");
        assert!(ref_cfg.full_resident_reference);
    }
}
