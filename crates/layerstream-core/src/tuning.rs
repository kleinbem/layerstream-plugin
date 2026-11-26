//! Quality-preserving per-machine scheduling autotuner (`layerstream-plugin tune`).
//!
//! Sweeps storage and I/O scheduling knobs (`prefer_direct_io`, `io_threads`) on a
//! real local file or temporary probe file, records sustained read throughput (MiB/s),
//! combines the winning I/O plan with the detected [`HardwareProfile`], and persists
//! the tuned profile by deterministic hardware fingerprint under
//! `$XDG_CONFIG_HOME/layerstream/tuning/<fingerprint>.json`.
//!
//! Never touches quantization, sampling, expert routing, or model weights.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::io_reader::probe_storage_mib_per_s;
use crate::profile::{HardwareProfile, StorageStack};

pub const TUNING_SCHEMA_VERSION: u32 = 1;
const DEFAULT_PROBE_MIB: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TuneCandidateResult {
    pub prefer_direct_io: bool,
    pub io_threads: usize,
    pub direct_io_active: bool,
    pub measured_mib_per_s: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TunedProfile {
    pub schema_version: u32,
    pub fingerprint: String,
    pub hardware_label: String,
    pub best_n_batch: u32,
    pub best_io_threads: usize,
    pub best_prefer_direct_io: bool,
    pub measured_mib_per_s: f64,
    pub candidates: Vec<TuneCandidateResult>,
}

/// Returns the default directory for persisted per-machine tuning profiles
/// (`$XDG_CONFIG_HOME/layerstream/tuning` or `$HOME/.config/layerstream/tuning`).
pub fn default_tuning_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("layerstream").join("tuning");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home)
            .join(".config")
            .join("layerstream")
            .join("tuning");
    }
    std::env::temp_dir().join("layerstream").join("tuning")
}

/// Computes a deterministic, filesystem-safe hardware fingerprint for caching
/// tuned profiles per machine configuration.
pub fn hardware_fingerprint(profile: &HardwareProfile) -> String {
    let ram_gib = profile.total_ram_bytes / (1024 * 1024 * 1024);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let raw = format!(
        "{}-{}-{}g-{}c",
        std::env::consts::OS,
        profile.label(),
        ram_gib,
        cores
    );
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Selects the best candidate from a sweep.
///
/// Tie-breaking favors `prefer_direct_io = true` (prevents OS page-cache bloat on
/// 8–16 GB edge devices) when within 5% of buffered throughput, and favors fewer
/// `io_threads` when throughput is within 3% (avoids `dm-crypt` / NVMe queue contention).
pub fn select_best_candidate(
    candidates: &[TuneCandidateResult],
    storage: StorageStack,
) -> Option<&TuneCandidateResult> {
    let mut best: Option<&TuneCandidateResult> = None;
    for c in candidates {
        let Some(cur) = best else {
            best = Some(c);
            continue;
        };
        let score = |x: &TuneCandidateResult| -> f64 {
            let mut s = x.measured_mib_per_s;
            if x.prefer_direct_io && x.direct_io_active {
                s *= 1.05;
            }
            if storage == StorageStack::LinuxDmCrypt && x.io_threads == 1 {
                s *= 1.03;
            }
            s
        };
        let s_new = score(c);
        let s_cur = score(cur);
        if s_new > s_cur * 1.001
            || ((s_new - s_cur).abs() <= s_cur * 0.001 && c.io_threads < cur.io_threads)
        {
            best = Some(c);
        }
    }
    best
}

impl TunedProfile {
    /// Persists the tuned profile JSON to `<dir>/<fingerprint>.json`.
    pub fn save(&self, dir: &Path) -> Result<PathBuf> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create tuning directory {}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.fingerprint));
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json)
            .with_context(|| format!("write tuned profile {}", path.display()))?;
        Ok(path)
    }

    /// Loads a persisted tuned profile for `fingerprint` if it exists and matches
    /// [`TUNING_SCHEMA_VERSION`]. Returns `Ok(None)` if absent or stale schema.
    pub fn load(dir: &Path, fingerprint: &str) -> Result<Option<Self>> {
        let path = dir.join(format!("{fingerprint}.json"));
        if !path.is_file() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read tuned profile {}", path.display()))?;
        let parsed: Self = serde_json::from_str(&raw)
            .with_context(|| format!("parse tuned profile {}", path.display()))?;
        if parsed.schema_version != TUNING_SCHEMA_VERSION {
            return Ok(None);
        }
        Ok(Some(parsed))
    }

    /// Formats a human-readable table summarizing the sweep and winning configuration.
    pub fn format_report(&self, saved_path: Option<&Path>) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Hardware profile: {} (fingerprint: {})\n",
            self.hardware_label, self.fingerprint
        ));
        out.push_str(&format!(
            "{:<3} {:<10} {:>10} {:<10} {:>11} {:>9}\n",
            "", "DIRECT_IO", "IO_THREADS", "O_DIRECT", "MIB/S", "ELAPSED"
        ));
        for c in &self.candidates {
            let is_win = c.prefer_direct_io == self.best_prefer_direct_io
                && c.io_threads == self.best_io_threads;
            let mark = if is_win { "★" } else { " " };
            out.push_str(&format!(
                "{:<3} {:<10} {:>10} {:<10} {:>11.1} {:>7} ms\n",
                mark,
                if c.prefer_direct_io { "yes" } else { "no" },
                c.io_threads,
                if c.direct_io_active {
                    "active"
                } else {
                    "buffered"
                },
                c.measured_mib_per_s,
                c.elapsed_ms
            ));
        }
        out.push_str(&format!(
            "\nSelected: n_batch={}, io_threads={}, prefer_direct_io={} ({:.1} MiB/s)\n",
            self.best_n_batch,
            self.best_io_threads,
            self.best_prefer_direct_io,
            self.measured_mib_per_s
        ));
        if let Some(p) = saved_path {
            out.push_str(&format!("Saved tuned profile: {}\n", p.display()));
        }
        out
    }
}

/// Runs a measured, quality-preserving storage/I/O scheduling sweep on `probe_file`.
pub fn run_storage_sweep(probe_file: &Path, profile: &HardwareProfile) -> Result<TunedProfile> {
    if !probe_file.is_file() {
        bail!("probe file does not exist: {}", probe_file.display());
    }
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 4))
        .unwrap_or(1);
    let mut thread_counts = vec![1usize];
    if max_threads >= 2 {
        thread_counts.push(2);
    }
    if max_threads >= 4 && profile.storage != StorageStack::LinuxDmCrypt {
        thread_counts.push(4);
    }

    let mut candidates = Vec::new();
    for prefer_direct in [true, false] {
        for &threads in &thread_counts {
            let t0 = Instant::now();
            let (mib_s, direct_ok) = if threads == 1 {
                probe_storage_mib_per_s(probe_file, prefer_direct)?
            } else {
                probe_parallel_mib_per_s(probe_file, prefer_direct, threads)?
            };
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            candidates.push(TuneCandidateResult {
                prefer_direct_io: prefer_direct,
                io_threads: threads,
                direct_io_active: direct_ok,
                measured_mib_per_s: mib_s,
                elapsed_ms,
            });
        }
    }

    let winner = select_best_candidate(&candidates, profile.storage)
        .context("storage sweep produced no candidates")?;

    Ok(TunedProfile {
        schema_version: TUNING_SCHEMA_VERSION,
        fingerprint: hardware_fingerprint(profile),
        hardware_label: profile.label(),
        best_n_batch: profile.recommended_n_batch,
        best_io_threads: winner.io_threads,
        best_prefer_direct_io: winner.prefer_direct_io,
        measured_mib_per_s: winner.measured_mib_per_s,
        candidates,
    })
}

/// Runs a sweep using a temporary probe file inside `dir` (cleaned up automatically).
pub fn run_dir_storage_sweep(dir: &Path, profile: &HardwareProfile) -> Result<TunedProfile> {
    fs::create_dir_all(dir).with_context(|| format!("create probe directory {}", dir.display()))?;
    let probe_path = dir.join(format!(
        ".layerstream-tune-probe-{}.tmp",
        std::process::id()
    ));
    {
        let mut f = File::create(&probe_path)
            .with_context(|| format!("create tune probe file {}", probe_path.display()))?;
        let chunk = vec![0xA5u8; 1024 * 1024];
        for _ in 0..DEFAULT_PROBE_MIB {
            f.write_all(&chunk)?;
        }
        f.sync_all()?;
    }
    let res = run_storage_sweep(&probe_path, profile);
    let _ = fs::remove_file(&probe_path);
    res
}

fn probe_parallel_mib_per_s(
    path: &Path,
    prefer_direct: bool,
    threads: usize,
) -> Result<(f64, bool)> {
    let t0 = Instant::now();
    let results: Result<Vec<(f64, bool)>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            handles.push(s.spawn(move || probe_storage_mib_per_s(path, prefer_direct)));
        }
        let mut out = Vec::with_capacity(threads);
        for h in handles {
            let r = h
                .join()
                .map_err(|_| anyhow::anyhow!("storage probe worker thread panicked"))??;
            out.push(r);
        }
        Ok(out)
    });
    let per_thread = results?;
    let wall_s = t0.elapsed().as_secs_f64().max(1e-6);
    let file_mib = (fs::metadata(path)?.len().min(64 * 1024 * 1024) as f64) / (1024.0 * 1024.0);
    let aggregate_mib_s = (file_mib * threads as f64) / wall_s;
    let all_direct = per_thread.iter().all(|&(_, d)| d);
    Ok((aggregate_mib_s, all_direct))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{MemoryTier, MemoryTopology, ProfileOverride};

    fn sample_profile(storage: StorageStack) -> HardwareProfile {
        HardwareProfile {
            memory_tier: MemoryTier::Edge,
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            topology: MemoryTopology::Unified,
            storage,
            backend: "cuda".to_string(),
            recommended_n_batch: 256,
            recommended_io_threads: 1,
            recommended_cpu_embeddings: false,
            recommended_stream_head: true,
        }
    }

    #[test]
    fn fingerprint_and_candidate_selection_apply_edge_tiebreakers() {
        let p = sample_profile(StorageStack::LinuxDmCrypt);
        let fp = hardware_fingerprint(&p);
        assert!(fp.contains("edge-unified-cuda-dmcrypt-8g-"));

        let candidates = vec![
            TuneCandidateResult {
                prefer_direct_io: true,
                io_threads: 1,
                direct_io_active: true,
                measured_mib_per_s: 2400.0,
                elapsed_ms: 14,
            },
            TuneCandidateResult {
                prefer_direct_io: false,
                io_threads: 2,
                direct_io_active: false,
                measured_mib_per_s: 2450.0,
                elapsed_ms: 13,
            },
        ];
        // Direct I/O + 1 thread on dm-crypt wins tiebreak over marginally faster buffered 2-thread
        let best = select_best_candidate(&candidates, StorageStack::LinuxDmCrypt)
            .expect("winner selected");
        assert!(best.prefer_direct_io);
        assert_eq!(best.io_threads, 1);

        // Clearly faster NVMe candidate wins on raw NVMe
        let nvme_candidates = vec![
            TuneCandidateResult {
                prefer_direct_io: true,
                io_threads: 1,
                direct_io_active: true,
                measured_mib_per_s: 1500.0,
                elapsed_ms: 20,
            },
            TuneCandidateResult {
                prefer_direct_io: true,
                io_threads: 4,
                direct_io_active: true,
                measured_mib_per_s: 3200.0,
                elapsed_ms: 10,
            },
        ];
        let best_nvme =
            select_best_candidate(&nvme_candidates, StorageStack::LinuxNvme).expect("nvme winner");
        assert_eq!(best_nvme.io_threads, 4);
        assert!(select_best_candidate(&[], StorageStack::LinuxNvme).is_none());
    }

    #[test]
    fn tuned_profile_save_load_report_and_dir_sweep_work() {
        let dir = std::env::temp_dir().join(format!("ls-tune-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");

        // Missing file errors cleanly
        let prof = HardwareProfile::detect(&dir, false, false, ProfileOverride::Auto);
        assert!(run_storage_sweep(&dir.join("missing.gguf"), &prof).is_err());

        // Real probe file sweep (4 MiB for fast unit test)
        let probe = dir.join("small.gguf");
        fs::write(&probe, vec![0x5Au8; 4 * 1024 * 1024]).expect("write 4 MiB probe");
        let tuned = run_storage_sweep(&probe, &prof).expect("run sweep");
        assert_eq!(tuned.schema_version, TUNING_SCHEMA_VERSION);
        assert!(!tuned.candidates.is_empty());
        assert!(tuned.measured_mib_per_s > 0.0);

        let saved = tuned.save(&dir).expect("save tuned profile");
        let loaded = TunedProfile::load(&dir, &tuned.fingerprint)
            .expect("load tuned profile")
            .expect("present");
        assert_eq!(loaded.fingerprint, tuned.fingerprint);
        assert_eq!(loaded.best_io_threads, tuned.best_io_threads);

        let report = tuned.format_report(Some(&saved));
        assert!(report.contains("Hardware profile:"));
        assert!(report.contains("Selected: n_batch="));
        assert!(report.contains("Saved tuned profile:"));

        // Absent fingerprint returns Ok(None)
        assert!(
            TunedProfile::load(&dir, "nonexistent-fp")
                .expect("absent ok")
                .is_none()
        );

        // Mismatched schema version returns Ok(None)
        let mut stale = tuned.clone();
        stale.schema_version = 999;
        stale.fingerprint = "stale-schema".to_string();
        stale.save(&dir).expect("save stale");
        assert!(
            TunedProfile::load(&dir, "stale-schema")
                .expect("stale ok")
                .is_none()
        );

        assert!(!default_tuning_dir().as_os_str().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
