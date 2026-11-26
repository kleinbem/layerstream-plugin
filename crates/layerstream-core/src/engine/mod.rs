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
    ls_engine_kv_trim, ls_engine_load_state, ls_engine_n_ctx, ls_engine_save_state,
    ls_engine_set_cancel_cb, ls_engine_token_is_eog, ls_engine_token_to_piece, ls_engine_tokenize,
    ls_engine_vocab_size, ls_grammar_accept, ls_grammar_create, ls_grammar_free,
    ls_probe_gpu_device, ls_probe_gpu_memory, ls_set_log_verbose,
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
            let pool = LayerIoPool::new(index, cfg.prefer_direct_io)?;
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

        Ok(Self {
            raw,
            io_pool: io_pool_raw,
            n_batch: cfg.n_batch.max(1) as usize,
            cached_kv_tokens: Vec::new(),
        })
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

    /// Saves the engine's current KV cache memory and committed token sequence to `path` on disk
    /// (allowing instant ~20 ms prefix restore on future runs without re-streaming prompt layers).
    /// Returns the number of KV tokens saved.
    pub fn save_state(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        let path_ref = path.as_ref();
        let parent_dir = path_ref
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent_dir)
            .with_context(|| format!("create parent directory '{}'", parent_dir.display()))?;

        // Write to a sibling temporary file first and atomically rename on completion so a
        // mid-save I/O error or worker exit never truncates an existing KV snapshot.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let base_name = path_ref
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("state.kv");
        let tmp_path = parent_dir.join(format!(".{base_name}.tmp.{}.{nonce}", std::process::id()));

        #[cfg(unix)]
        let path_bytes = std::os::unix::ffi::OsStrExt::as_bytes(tmp_path.as_os_str());
        #[cfg(not(unix))]
        let path_lossy = tmp_path.to_string_lossy();
        #[cfg(not(unix))]
        let path_bytes = path_lossy.as_bytes();
        let c_path = CString::new(path_bytes).context("state path contains interior NUL")?;

        // Trim any uncommitted speculative tail above `cached_kv_tokens` before saving.
        if self.stats().can_speculate {
            let _ =
                // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
                unsafe { ls_engine_kv_trim(self.raw.as_ptr(), self.cached_kv_tokens.len() as i32) };
        }

        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_save_state(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                if self.cached_kv_tokens.is_empty() {
                    std::ptr::null()
                } else {
                    self.cached_kv_tokens.as_ptr()
                },
                self.cached_kv_tokens.len(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            let _ = std::fs::remove_file(&tmp_path);
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("save_state failed: {msg}");
        }
        if let Err(rename_err) = std::fs::rename(&tmp_path, path_ref) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(rename_err).with_context(|| {
                format!(
                    "replace KV state snapshot '{}' from '{}'",
                    path_ref.display(),
                    tmp_path.display()
                )
            });
        }
        Ok(self.cached_kv_tokens.len())
    }

    /// Restores a previously saved KV cache snapshot and token sequence from `path`.
    /// Subsequent `.generate()` / `.chat()` calls automatically reuse the restored KV prefix.
    /// Returns the number of KV tokens restored.
    pub fn load_state(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        let path_ref = path.as_ref();
        if !path_ref.is_file() {
            bail!("KV state file not found: '{}'", path_ref.display());
        }
        let st = self.stats();
        if !st.can_speculate || st.rs_rollback_max > 0 {
            bail!(
                "KV state restore requires a KV-trimmable transformer architecture \
                 (unsupported for hybrid recurrent/SSM models)"
            );
        }
        #[cfg(unix)]
        let path_bytes = std::os::unix::ffi::OsStrExt::as_bytes(path_ref.as_os_str());
        #[cfg(not(unix))]
        let path_lossy = path_ref.to_string_lossy();
        #[cfg(not(unix))]
        let path_bytes = path_lossy.as_bytes();
        let c_path = CString::new(path_bytes).context("state path contains interior NUL")?;

        let cap = (self.n_ctx().max(1) as usize).max(16);
        let mut tokens_out = vec![0i32; cap];
        let mut n_loaded: usize = 0;
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_load_state(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                tokens_out.as_mut_ptr(),
                cap,
                &mut n_loaded,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            self.cached_kv_tokens.clear();
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("load_state failed: {msg}");
        }
        tokens_out.truncate(n_loaded.min(cap));
        self.cached_kv_tokens = tokens_out;
        Ok(self.cached_kv_tokens.len())
    }

    pub fn eval_greedy(&mut self, tokens: &[i32], pos_start: i32) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_greedy(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_greedy failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    pub fn eval_sample(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
    ) -> Result<i32> {
        self.eval_sample_with_history(tokens, pos_start, sample, &[])
    }

    pub fn eval_sample_with_history(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
    ) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_sample failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    pub fn eval_verify_greedy(&mut self, tokens: &[i32], pos_start: i32) -> Result<Vec<i32>> {
        let mut out = vec![0i32; tokens.len()];
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_verify_greedy(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                out.as_mut_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_greedy failed (rc={rc}): {msg}");
        }
        Ok(out)
    }

    /// Lossless speculative rejection sampling verification for `temperature > 0` (or penalized greedy).
    /// Returns `(accepted_draft_count, next_token)`.
    pub fn eval_verify_sample(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
    ) -> Result<(usize, i32)> {
        self.eval_verify_sample_with_history(tokens, pos_start, sample, &[])
    }

    pub fn eval_verify_sample_with_history(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
    ) -> Result<(usize, i32)> {
        let mut accepted = 0i32;
        let mut next_tok = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_verify_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                &mut accepted,
                &mut next_tok,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_sample failed (rc={rc}): {msg}");
        }
        Ok((accepted.max(0) as usize, next_tok))
    }

    fn eval_step_with_grammar(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
        grammar: Option<&GrammarHandle>,
    ) -> Result<i32> {
        let Some(g) = grammar else {
            return if sample.temperature <= 0.0 && !sample.has_penalties() {
                self.eval_greedy(tokens, pos_start)
            } else {
                self.eval_sample_with_history(tokens, pos_start, sample, penalty_history)
            };
        };
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_with_grammar(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                g.raw.as_ptr(),
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_with_grammar failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    fn eval_verify_with_grammar(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
        grammar: &GrammarHandle,
    ) -> Result<(usize, i32)> {
        let mut accepted = 0i32;
        let mut next_tok = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_verify_with_grammar(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                grammar.raw.as_ptr(),
                &mut accepted,
                &mut next_tok,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_with_grammar failed (rc={rc}): {msg}");
        }
        Ok((accepted.max(0) as usize, next_tok))
    }

    /// Runs chunked layer-major prefill followed by greedy decode (without speculation).
    pub fn generate_greedy(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>> {
        let mut opts = GenerateOptions {
            max_tokens,
            enable_ngram_speculation: false,
            ..Default::default()
        };
        opts.sample.temperature = 0.0;
        let mut out = Vec::with_capacity(max_tokens);
        self.generate_stream(prompt, &opts, None, |tok, _piece| {
            out.push(tok);
            true
        })?;
        Ok(out)
    }

    /// Runs chunked layer-major prefill + streaming generation with optional n-gram
    /// and/or resident draft-model speculative verification.
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        draft_proposer: Option<&mut DraftModelProposer>,
        on_token: F,
    ) -> Result<GenerateSummary>
    where
        F: FnMut(i32, &str) -> bool,
    {
        self.generate_stream_with_cancel(prompt, opts, draft_proposer, None, on_token)
    }

    /// Same as `generate_stream`, plus an optional cooperative cancellation flag that
    /// aborts prefill and mid-layer `llama_decode` passes immediately on client disconnect
    /// or process shutdown.
    pub fn generate_stream_with_cancel<F>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        mut draft_proposer: Option<&mut DraftModelProposer>,
        cancel: Option<&AtomicBool>,
        mut on_token: F,
    ) -> Result<GenerateSummary>
    where
        F: FnMut(i32, &str) -> bool,
    {
        let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        if is_cancelled() {
            return Ok(GenerateSummary {
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "cancelled".to_string(),
                speculation: SpeculationStats::default(),
            });
        }

        let prompt_tokens = self.tokenize(prompt, true, true)?;
        if prompt_tokens.is_empty() {
            bail!("prompt tokenized to 0 tokens");
        }
        let n_ctx = self.n_ctx() as usize;
        if n_ctx > 0 && prompt_tokens.len() >= n_ctx {
            return Err(PromptTooLongError {
                prompt_tokens: prompt_tokens.len(),
                n_ctx,
            }
            .into());
        }

        let mut grammar = match opts.grammar.as_deref() {
            None => None,
            Some(raw) => {
                let g_str = raw.trim();
                if g_str.is_empty() {
                    bail!("grammar string must not be empty");
                }
                Some(GrammarHandle::create(self.raw, g_str)?)
            }
        };

        // Reuse longest matching KV prefix on multi-turn chat continuations
        // (avoids re-prefilling unchanged system prompts & prior turns off NVMe!).
        let st = self.stats();
        let can_reuse_kv = st.can_speculate && st.rs_rollback_max == 0;
        let mut common_prefix = 0usize;
        if can_reuse_kv && !self.cached_kv_tokens.is_empty() {
            let max_shared = self
                .cached_kv_tokens
                .len()
                .min(prompt_tokens.len().saturating_sub(1));
            while common_prefix < max_shared
                && self.cached_kv_tokens[common_prefix] == prompt_tokens[common_prefix]
            {
                common_prefix += 1;
            }
        }
        if common_prefix > 0 && self.kv_trim(common_prefix as i32) {
            tracing::debug!(common_prefix, "reusing cached KV prefix across turn");
        } else {
            self.kv_clear();
            common_prefix = 0;
        }
        // Truncate cached_kv_tokens BEFORE prefill so any mid-prefill error or cancellation
        // never leaves cached_kv_tokens ahead of actual KV cache state.
        self.cached_kv_tokens.truncate(common_prefix);

        if let Some(dp) = draft_proposer.as_deref_mut() {
            dp.trim_to(common_prefix);
        }

        let _cancel_guard = CancelCbGuard::install(self.raw, cancel);

        let mut history = prompt_tokens.clone();
        let mut pos = common_prefix as i32;
        let mut next_tok = 0i32;
        let mut spec_stats = SpeculationStats::default();

        let prefill_slice = &prompt_tokens[common_prefix..];
        let total_prefill_chunks = prefill_slice.chunks(self.n_batch).len();
        for (chunk_idx, chunk) in prefill_slice.chunks(self.n_batch).enumerate() {
            if is_cancelled() {
                return Ok(GenerateSummary {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    finish_reason: "cancelled".to_string(),
                    speculation: spec_stats,
                });
            }
            let is_last_prefill = chunk_idx + 1 == total_prefill_chunks;
            let eval_res = if is_last_prefill {
                self.eval_step_with_grammar(chunk, pos, &opts.sample, &history, grammar.as_ref())
            } else {
                // Intermediate prefill chunks discard their sampled output token, so run pure greedy.
                self.eval_greedy(chunk, pos)
            };
            match eval_res {
                Ok(tok) => {
                    next_tok = tok;
                    self.cached_kv_tokens.extend_from_slice(chunk);
                    spec_stats.sweeps += 1;
                    pos += chunk.len() as i32;
                }
                Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                    // Partial/aborted chunk may have touched KV cache; clear so next turn stays clean.
                    self.kv_clear();
                    return Ok(GenerateSummary {
                        prompt_tokens: prompt_tokens.len(),
                        completion_tokens: 0,
                        finish_reason: "cancelled".to_string(),
                        speculation: spec_stats,
                    });
                }
                Err(e) => {
                    self.kv_clear();
                    return Err(e);
                }
            }
        }

        let target_vocab = self.vocab_size();
        let max_gen_tokens = if n_ctx > 0 {
            opts.max_tokens
                .min(n_ctx.saturating_sub(prompt_tokens.len()))
        } else {
            opts.max_tokens
        };
        let rs_rollback_cap = if st.rs_rollback_max > 0 {
            st.rs_rollback_max as usize
        } else {
            usize::MAX
        };
        let can_speculate =
            st.can_speculate && (opts.enable_ngram_speculation || draft_proposer.is_some());

        let min_adaptive_k = if draft_proposer.is_some() {
            opts.max_draft_tokens.clamp(6, 10)
        } else {
            3
        }
        .min(rs_rollback_cap);
        let mut adaptive_draft_k = opts
            .max_draft_tokens
            .max(min_adaptive_k)
            .min(rs_rollback_cap);
        let max_adaptive_k = if draft_proposer.is_some() {
            (opts.max_draft_tokens * 3).clamp(12, 20)
        } else {
            (opts.max_draft_tokens * 2).clamp(4, 12)
        };

        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut utf8_assembler = Utf8PieceAssembler::default();
        let mut stop_filter = StopFilter::new(&opts.stop);
        let mut hit_stop_any = false;
        let mut last_emitted_tok = next_tok;

        'gen_loop: while completion_tokens < max_gen_tokens {
            if is_cancelled() {
                finish_reason = "cancelled".to_string();
                break;
            }
            if self.token_is_eog(next_tok) {
                finish_reason = "stop".to_string();
                break;
            }
            if let Some(ref mut g) = grammar {
                g.accept(next_tok)?;
            }

            // Emit `next_tok` BEFORE calling `draft_proposer.propose(...)` so draft latency
            // never pads TTFT, buffering incomplete multi-byte UTF-8 sequences across tokens
            // and routing through `StopFilter` so stop text is never leaked.
            let raw_bytes = self.token_to_piece_bytes(next_tok)?;
            let piece = utf8_assembler.push(&raw_bytes);
            completion_tokens += 1;
            history.push(next_tok);
            last_emitted_tok = next_tok;
            let (safe_emit, hit_stop_seq) = stop_filter.push(&piece);
            if (!safe_emit.is_empty()
                || (opts.stop.is_empty() && piece.is_empty() && raw_bytes.is_empty()))
                && !on_token(next_tok, &safe_emit)
            {
                finish_reason = "cancelled".to_string();
                break 'gen_loop;
            }
            if hit_stop_seq {
                hit_stop_any = true;
                finish_reason = "stop".to_string();
                break 'gen_loop;
            }
            if completion_tokens >= max_gen_tokens {
                finish_reason = "length".to_string();
                break 'gen_loop;
            }

            // Remaining budget for this pass: `pos` will hold `next_tok`, and `draft` tokens
            // occupy `pos + 1 .. pos + 1 + draft.len()`. Ensure `pos + 1 + draft.len() <= n_ctx`.
            let remaining_by_max = max_gen_tokens - completion_tokens;
            let remaining_by_ctx = if n_ctx > 0 {
                n_ctx.saturating_sub((pos as usize) + 1)
            } else {
                remaining_by_max
            };
            let max_step_draft = remaining_by_max.min(remaining_by_ctx).min(rs_rollback_cap);
            if n_ctx > 0 && (pos as usize) >= n_ctx {
                finish_reason = "length".to_string();
                break 'gen_loop;
            }

            // Speculative multi-token path (exact greedy or unbiased rejection sampling in 1 sweep).
            if can_speculate && max_step_draft > 0 {
                // Cascade: when a neural draft model is active, only let n-gram override it
                // on strong verbatim matches (min_n=4, e.g. code edits/copying) so noisy
                // 2-token prose hits never preempt the smarter draft model.
                let ngram = if draft_proposer.is_some() {
                    NgramProposer {
                        min_n: 4,
                        max_n: 6,
                        max_draft: adaptive_draft_k,
                    }
                } else {
                    NgramProposer {
                        max_draft: adaptive_draft_k,
                        ..Default::default()
                    }
                };
                let mut draft = if opts.enable_ngram_speculation {
                    ngram.propose(&history, max_step_draft)
                } else {
                    Vec::new()
                };
                if draft.len() < 2
                    && let Some(dp) = draft_proposer.as_deref_mut()
                {
                    draft = dp.propose_with_cancel(
                        &history,
                        max_step_draft.min(adaptive_draft_k),
                        cancel,
                    )?;
                    if is_cancelled() {
                        finish_reason = "cancelled".to_string();
                        break;
                    }
                }
                // Guard against padded-vocab draft models proposing an out-of-range token ID.
                if let Some(bad_idx) = draft.iter().position(|&t| t < 0 || t >= target_vocab) {
                    draft.truncate(bad_idx);
                }

                if !draft.is_empty() {
                    let mut batch = Vec::with_capacity(1 + draft.len());
                    batch.push(next_tok);
                    batch.extend_from_slice(&draft);

                    let mut step_sample = opts.sample;
                    if step_sample.temperature > 0.0 {
                        step_sample.seed = step_sample
                            .seed
                            .wrapping_add((completion_tokens as u32).wrapping_mul(2654435761));
                    }

                    let verify_res = if let Some(ref g) = grammar {
                        self.eval_verify_with_grammar(&batch, pos, &step_sample, &history, g)
                    } else if step_sample.temperature <= 0.0 && !step_sample.has_penalties() {
                        self.eval_verify_greedy(&batch, pos).map(|verified| {
                            let mut acc = 0usize;
                            let mut next_after = verified[draft.len()];
                            for (i, &d_tok) in draft.iter().enumerate() {
                                if verified[i] == d_tok {
                                    acc += 1;
                                } else {
                                    next_after = verified[i];
                                    break;
                                }
                            }
                            (acc, next_after)
                        })
                    } else {
                        self.eval_verify_sample_with_history(&batch, pos, &step_sample, &history)
                    };

                    let (accepted_n, corrected_or_bonus) = match verify_res {
                        Ok(v) => v,
                        Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                            self.kv_clear();
                            finish_reason = "cancelled".to_string();
                            break 'gen_loop;
                        }
                        Err(e) => {
                            self.kv_clear();
                            return Err(e);
                        }
                    };
                    spec_stats.sweeps += 1;
                    spec_stats.drafted_tokens += draft.len();

                    let mut accepted = 0usize;
                    let mut early_stop = false;
                    for &d_tok in draft.iter().take(accepted_n) {
                        spec_stats.accepted_tokens += 1;
                        if self.token_is_eog(d_tok) {
                            finish_reason = "stop".to_string();
                            early_stop = true;
                            break;
                        }
                        accepted += 1;
                        if let Some(ref mut g) = grammar {
                            g.accept(d_tok)?;
                        }
                        let d_bytes = self.token_to_piece_bytes(d_tok)?;
                        let d_piece = utf8_assembler.push(&d_bytes);
                        completion_tokens += 1;
                        history.push(d_tok);
                        last_emitted_tok = d_tok;
                        let (safe_emit, hit_stop_seq) = stop_filter.push(&d_piece);
                        if (!safe_emit.is_empty()
                            || (opts.stop.is_empty() && d_piece.is_empty() && d_bytes.is_empty()))
                            && !on_token(d_tok, &safe_emit)
                        {
                            finish_reason = "cancelled".to_string();
                            early_stop = true;
                            break;
                        }
                        if hit_stop_seq {
                            hit_stop_any = true;
                            finish_reason = "stop".to_string();
                            early_stop = true;
                            break;
                        }
                        if completion_tokens >= max_gen_tokens {
                            finish_reason = "length".to_string();
                            early_stop = true;
                            break;
                        }
                    }

                    // Always trim any unaccepted or post-stop tail from KV cache and verify `kv_trim`.
                    let valid_pos = pos + 1 + accepted as i32;
                    if accepted < draft.len() {
                        if !self.kv_trim(valid_pos) {
                            self.kv_clear();
                            bail!(
                                "kv_trim({valid_pos}) failed after speculative verification; \
                                 model memory does not support partial rollback"
                            );
                        }
                        if self.cached_kv_tokens.len() <= history.len() {
                            self.cached_kv_tokens
                                .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                        } else {
                            self.cached_kv_tokens = history.clone();
                        }
                        pos = valid_pos;
                        next_tok = corrected_or_bonus;
                    } else {
                        if self.cached_kv_tokens.len() <= history.len() {
                            self.cached_kv_tokens
                                .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                        } else {
                            self.cached_kv_tokens = history.clone();
                        }
                        pos += 1 + draft.len() as i32;
                        next_tok = corrected_or_bonus;
                    }

                    if early_stop {
                        break 'gen_loop;
                    }

                    // Adapt draft depth upward aggressively on full acceptance streaks
                    // (verifying extra tokens is nearly free compared to an NVMe sweep),
                    // and gently downward on misses without collapsing below `min_adaptive_k`.
                    if accepted == draft.len() {
                        let step = if draft_proposer.is_some() { 4 } else { 2 };
                        adaptive_draft_k = (adaptive_draft_k + step).min(max_adaptive_k);
                    } else if accepted == 0 {
                        adaptive_draft_k = (adaptive_draft_k.saturating_sub(1)).max(min_adaptive_k);
                    }
                    continue 'gen_loop;
                }
            }

            // Standard 1-token sweep fallback.
            let mut step_sample = opts.sample;
            step_sample.seed = step_sample.seed.wrapping_add(completion_tokens as u32);
            let eval_res = self.eval_step_with_grammar(
                &[next_tok],
                pos,
                &step_sample,
                &history,
                grammar.as_ref(),
            );
            match eval_res {
                Ok(tok) => {
                    next_tok = tok;
                    if self.cached_kv_tokens.len() <= history.len() {
                        self.cached_kv_tokens
                            .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                    } else {
                        self.cached_kv_tokens = history.clone();
                    }
                    spec_stats.sweeps += 1;
                    pos += 1;
                }
                Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                    self.kv_clear();
                    finish_reason = "cancelled".to_string();
                    break 'gen_loop;
                }
                Err(e) => {
                    self.kv_clear();
                    return Err(e);
                }
            }
        }

        // Flush any trailing incomplete UTF-8 bytes and held-back stop-filter suffix bytes
        // when generation ended without cancellation or an already-matched stop sequence.
        if finish_reason != "cancelled" && !hit_stop_any {
            let utf8_tail = utf8_assembler.flush();
            if !utf8_tail.is_empty() {
                let (safe_emit, hit_stop_seq) = stop_filter.push(&utf8_tail);
                if !safe_emit.is_empty() && !on_token(last_emitted_tok, &safe_emit) {
                    finish_reason = "cancelled".to_string();
                } else if hit_stop_seq {
                    hit_stop_any = true;
                    finish_reason = "stop".to_string();
                }
            }
        }
        if finish_reason != "cancelled" && !hit_stop_any {
            let tail = stop_filter.flush();
            if !tail.is_empty() && !on_token(last_emitted_tok, &tail) {
                finish_reason = "cancelled".to_string();
            }
        }

        Ok(GenerateSummary {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            finish_reason,
            speculation: spec_stats,
        })
    }
}

struct GrammarHandle {
    raw: NonNull<LsGrammar>,
}

impl GrammarHandle {
    fn create(eng: NonNull<LsEngine>, grammar_str: &str) -> Result<Self> {
        let c_grammar =
            CString::new(grammar_str).context("GBNF grammar contains interior NUL byte")?;
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `eng` is a live `LsEngine`, `c_grammar` and `c"root"` are NUL-terminated, and `err_buf` is valid for its length.
        let ptr = unsafe {
            ls_grammar_create(
                eng.as_ptr(),
                c_grammar.as_ptr(),
                c"root".as_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        let Some(raw) = NonNull::new(ptr) else {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("invalid grammar: {msg}");
        };
        Ok(Self { raw })
    }

    fn accept(&mut self, token: i32) -> Result<()> {
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_grammar_accept(
                self.raw.as_ptr(),
                token,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("grammar state error on token {token}: {msg}");
        }
        Ok(())
    }
}

impl Drop for GrammarHandle {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is a valid grammar handle owned exclusively by `self` and freed once on drop.
        unsafe { ls_grammar_free(self.raw.as_ptr()) };
    }
}

pub(crate) struct CancelCbGuard<'a> {
    raw: NonNull<LsEngine>,
    installed: bool,
    _flag: std::marker::PhantomData<Option<&'a AtomicBool>>,
}

impl StreamEngine {
    pub(crate) fn install_cancel_cb<'a>(
        &self,
        cancel: Option<&'a AtomicBool>,
    ) -> CancelCbGuard<'a> {
        CancelCbGuard::install(self.raw, cancel)
    }
}

impl<'a> CancelCbGuard<'a> {
    fn install(raw: NonNull<LsEngine>, cancel: Option<&'a AtomicBool>) -> Self {
        if let Some(flag) = cancel {
            // SAFETY: `self.raw` is a live `LsEngine`, and `CancelCbGuard` clears the callback before the `AtomicBool` borrow ends.
            unsafe {
                ls_engine_set_cancel_cb(
                    raw.as_ptr(),
                    Some(rust_cancel_check_cb),
                    (flag as *const AtomicBool).cast::<c_void>(),
                );
            }
            Self {
                raw,
                installed: true,
                _flag: std::marker::PhantomData,
            }
        } else {
            Self {
                raw,
                installed: false,
                _flag: std::marker::PhantomData,
            }
        }
    }
}

impl Drop for CancelCbGuard<'_> {
    fn drop(&mut self) {
        if self.installed {
            // SAFETY: `self.raw` is a live `LsEngine`, and `CancelCbGuard` clears the callback before the `AtomicBool` borrow ends.
            unsafe { ls_engine_set_cancel_cb(self.raw.as_ptr(), None, std::ptr::null()) };
        }
    }
}

unsafe extern "C" fn rust_cancel_check_cb(user_data: *const c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: `user_data` was registered by `CancelCbGuard` pointing to a live `AtomicBool` for the duration of the C++ call.
    let flag = unsafe { &*(user_data as *const AtomicBool) };
    flag.load(Ordering::Relaxed)
}

/// Sliding UTF-8-safe holdback buffer for OpenAI `stop` sequences.
/// Guarantees stop sequences are never emitted to streaming clients (even when split across
/// multiple tokens) and catches mid-piece matches (e.g. `"\n\nUser:"` with `stop: ["\n\n"]`).
pub struct StopFilter {
    stops: Vec<String>,
    max_stop_bytes: usize,
    pending: String,
}

impl StopFilter {
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_stop_bytes = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_stop_bytes,
            pending: String::new(),
        }
    }

    /// Pushes a new piece and returns `(safe_prefix_to_emit, matched_stop)`.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (piece.to_string(), false);
        }
        self.pending.push_str(piece);

        let mut earliest_hit: Option<usize> = None;
        for stop in &self.stops {
            if let Some(idx) = self.pending.find(stop.as_str()) {
                earliest_hit = Some(earliest_hit.map_or(idx, |prev| prev.min(idx)));
            }
        }
        if let Some(hit_idx) = earliest_hit {
            let emit = self.pending[..hit_idx].to_string();
            self.pending.clear();
            return (emit, true);
        }

        let keep = self.max_stop_bytes.saturating_sub(1);
        if self.pending.len() <= keep {
            return (String::new(), false);
        }
        let mut split = self.pending.len() - keep;
        while split > 0 && !self.pending.is_char_boundary(split) {
            split -= 1;
        }
        if split == 0 {
            return (String::new(), false);
        }
        let emit = self.pending[..split].to_string();
        self.pending.drain(..split);
        (emit, false)
    }

    /// Flushes any remaining buffered bytes at natural end-of-generation (non-stop).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

impl Drop for StreamEngine {
    fn drop(&mut self) {
        // Free C++ engine FIRST so `worker_thread` is joined before freeing `LayerIoPool`.
        // SAFETY: `self.raw` is a valid engine handle owned exclusively by `self` and freed once on drop before reclaiming `io_pool_ptr`.
        unsafe { ls_engine_free(self.raw.as_ptr()) };
        if let Some(p) = self.io_pool.take() {
            // SAFETY: `p` was created via `Box::into_raw` in `StreamEngine::open` and is reclaimed exactly once after C++ no longer references it.
            drop(unsafe { Box::from_raw(p) });
        }
    }
}

mod callbacks;
use callbacks::{
    Utf8PieceAssembler, rust_load_layer_direct_cb, rust_load_slices_cb, rust_load_tensors_cb,
};
