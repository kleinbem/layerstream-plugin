use super::*;

impl StreamEngine {
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
}
