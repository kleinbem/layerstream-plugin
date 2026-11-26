//! Auto-detected hardware & OS capability profiles.
//!
//! Instead of hardcoding device hostnames, `HardwareProfile` probes four
//! orthogonal traits at runtime:
//! 1. **Memory tier** (`Edge <= 12 GiB`, `Standard 12..48 GiB`, `High >= 48 GiB`)
//! 2. **Memory topology** (`Unified` vs `Discrete`)
//! 3. **OS & storage stack** (`LinuxDmCrypt`, `LinuxNvme`, `MacOsNoCache`, `WindowsNoBuffering`)
//! 4. **GPU backend** (`cuda`, `vulkan`, `metal`, `cpu`)
//!
//! When `probe_gpu` is `false` (used during parent coordinator startup so zero
//! CUDA driver context is pinned while idle), only RAM and storage are probed.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// <= 12 GiB total RAM (e.g., 8 GB Jetson Orin Nano, 8 GB laptop).
    Edge,
    /// 12..48 GiB total RAM (e.g., 16–32 GB laptop / desktop).
    Standard,
    /// >= 48 GiB total RAM (e.g., 64 GB AMD APU, Mac Studio, workstation).
    High,
}

impl MemoryTier {
    pub fn from_total_bytes(total_ram_bytes: u64) -> Self {
        if total_ram_bytes == 0 || total_ram_bytes <= 12 * GIB {
            Self::Edge
        } else if total_ram_bytes < 48 * GIB {
            Self::Standard
        } else {
            Self::High
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Edge => "edge",
            Self::Standard => "standard",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTopology {
    Unified,
    Discrete,
}

impl MemoryTopology {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unified => "unified",
            Self::Discrete => "discrete",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageStack {
    LinuxDmCrypt,
    LinuxNvme,
    MacOsNoCache,
    WindowsNoBuffering,
    Generic,
}

impl StorageStack {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinuxDmCrypt => "dmcrypt",
            Self::LinuxNvme => "nvme",
            Self::MacOsNoCache => "macos_nocache",
            Self::WindowsNoBuffering => "win_nobuf",
            Self::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileOverride {
    #[default]
    Auto,
    Edge,
    Balanced,
    Performance,
}

impl fmt::Display for ProfileOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Edge => f.write_str("edge"),
            Self::Balanced => f.write_str("balanced"),
            Self::Performance => f.write_str("performance"),
        }
    }
}

impl FromStr for ProfileOverride {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "edge" | "low-vram" | "low_vram" => Ok(Self::Edge),
            "balanced" | "standard" => Ok(Self::Balanced),
            "performance" | "perf" | "high" => Ok(Self::Performance),
            other => Err(format!(
                "invalid profile '{other}' (expected auto|edge|balanced|performance)"
            )),
        }
    }
}

/// Resolved capability profile and recommended runtime tuning knobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub memory_tier: MemoryTier,
    pub topology: MemoryTopology,
    pub storage: StorageStack,
    pub backend: String,
    pub total_ram_bytes: u64,
    pub recommended_n_batch: u32,
    pub recommended_io_threads: usize,
    pub recommended_cpu_embeddings: bool,
    pub recommended_stream_head: bool,
}

impl HardwareProfile {
    /// Deterministic constructor from already-probed inputs (ideal for testing).
    pub fn from_inputs(
        total_ram_bytes: u64,
        topology: MemoryTopology,
        storage: StorageStack,
        backend: impl Into<String>,
        override_mode: ProfileOverride,
    ) -> Self {
        let detected_tier = MemoryTier::from_total_bytes(total_ram_bytes);
        let memory_tier = match override_mode {
            ProfileOverride::Auto => detected_tier,
            ProfileOverride::Edge => MemoryTier::Edge,
            ProfileOverride::Balanced => MemoryTier::Standard,
            ProfileOverride::Performance => MemoryTier::High,
        };

        let recommended_n_batch = match memory_tier {
            MemoryTier::Edge => 256,
            MemoryTier::Standard | MemoryTier::High => 512,
        };

        let recommended_io_threads = match (storage, memory_tier) {
            (StorageStack::LinuxDmCrypt, _) => 1,
            (StorageStack::LinuxNvme, MemoryTier::High) => 4,
            (
                StorageStack::LinuxNvme
                | StorageStack::MacOsNoCache
                | StorageStack::WindowsNoBuffering,
                _,
            ) => 2,
            (StorageStack::Generic, _) => 1,
        };

        let recommended_cpu_embeddings =
            topology == MemoryTopology::Discrete && memory_tier != MemoryTier::High;
        let recommended_stream_head = override_mode == ProfileOverride::Edge;

        Self {
            memory_tier,
            topology,
            storage,
            backend: backend.into(),
            total_ram_bytes,
            recommended_n_batch,
            recommended_io_threads,
            recommended_cpu_embeddings,
            recommended_stream_head,
        }
    }

    /// Detects the host's hardware & OS profile without initializing GPU drivers
    /// unless `probe_gpu` is `true`.
    pub fn detect(
        storage_path: &Path,
        use_gpu: bool,
        probe_gpu: bool,
        override_mode: ProfileOverride,
    ) -> Self {
        let total_ram_bytes = detect_total_ram_bytes();
        let storage = detect_storage_stack(storage_path);
        let (topology, backend) = if !use_gpu {
            (MemoryTopology::Unified, "cpu".to_string())
        } else if probe_gpu {
            match crate::StreamEngine::probe_gpu_memory(true) {
                Some(_) => (default_os_topology(), default_gpu_backend().to_string()),
                None => (MemoryTopology::Unified, "cpu".to_string()),
            }
        } else {
            (default_os_topology(), default_gpu_backend().to_string())
        };
        Self::from_inputs(total_ram_bytes, topology, storage, backend, override_mode)
    }

    /// Compact human-readable identifier, e.g. `edge-unified-cuda-dmcrypt`.
    pub fn label(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.memory_tier.as_str(),
            self.topology.as_str(),
            self.backend,
            self.storage.as_str(),
        )
    }
}

fn default_os_topology() -> MemoryTopology {
    #[cfg(any(target_os = "macos", target_arch = "aarch64"))]
    {
        MemoryTopology::Unified
    }
    #[cfg(not(any(target_os = "macos", target_arch = "aarch64")))]
    {
        MemoryTopology::Discrete
    }
}

fn default_gpu_backend() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "metal"
    }
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        "cuda"
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        "vulkan"
    }
}

/// Parses Linux `/proc/meminfo` for `MemTotal:`.
pub fn parse_proc_memtotal_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:")
            && let Some(kb_str) = rest.split_whitespace().next()
            && let Ok(kb) = kb_str.parse::<u64>()
        {
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn detect_total_ram_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo")
            && let Some(bytes) = parse_proc_memtotal_bytes(&text)
        {
            return bytes;
        }
    }
    0
}

/// Detects whether `target_path` resides on a Linux device-mapper (`dm-crypt`)
/// mount using `/proc/self/mountinfo` contents.
pub fn is_mountinfo_dmcrypt(mountinfo: &str, target_path: &Path) -> bool {
    let target = target_path.to_string_lossy();
    let mut best_len = 0usize;
    let mut best_is_dm = false;
    for line in mountinfo.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let left_fields: Vec<&str> = left.split_whitespace().collect();
        let right_fields: Vec<&str> = right.split_whitespace().collect();
        if left_fields.len() < 5 || right_fields.len() < 2 {
            continue;
        }
        let mount_point = left_fields[4];
        let mount_source = right_fields[1];
        let matches = mount_point == "/"
            || target.as_ref() == mount_point
            || target.starts_with(&format!("{mount_point}/"));
        if matches && mount_point.len() >= best_len {
            best_len = mount_point.len();
            best_is_dm =
                mount_source.starts_with("/dev/mapper/") || mount_source.starts_with("/dev/dm-");
        }
    }
    best_is_dm
}

fn detect_storage_stack(path: &Path) -> StorageStack {
    #[cfg(target_os = "linux")]
    {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Ok(mountinfo) = std::fs::read_to_string("/proc/self/mountinfo")
            && is_mountinfo_dmcrypt(&mountinfo, &canonical)
        {
            return StorageStack::LinuxDmCrypt;
        }
        StorageStack::LinuxNvme
    }
    #[cfg(target_os = "macos")]
    {
        let _ = path;
        StorageStack::MacOsNoCache
    }
    #[cfg(target_os = "windows")]
    {
        let _ = path;
        StorageStack::WindowsNoBuffering
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = path;
        StorageStack::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_edge_standard_high_and_overrides_cleanly() {
        let orin = HardwareProfile::from_inputs(
            8 * GIB,
            MemoryTopology::Unified,
            StorageStack::LinuxDmCrypt,
            "cuda",
            ProfileOverride::Auto,
        );
        assert_eq!(orin.memory_tier, MemoryTier::Edge);
        assert_eq!(orin.recommended_n_batch, 256);
        assert_eq!(orin.recommended_io_threads, 1);
        assert!(!orin.recommended_cpu_embeddings);
        assert_eq!(orin.label(), "edge-unified-cuda-dmcrypt");

        let workstation = HardwareProfile::from_inputs(
            64 * GIB,
            MemoryTopology::Discrete,
            StorageStack::LinuxNvme,
            "cuda",
            ProfileOverride::Auto,
        );
        assert_eq!(workstation.memory_tier, MemoryTier::High);
        assert_eq!(workstation.recommended_n_batch, 512);
        assert_eq!(workstation.recommended_io_threads, 4);
        assert!(!workstation.recommended_cpu_embeddings);
        assert_eq!(workstation.label(), "high-discrete-cuda-nvme");

        let discrete_16g = HardwareProfile::from_inputs(
            16 * GIB,
            MemoryTopology::Discrete,
            StorageStack::WindowsNoBuffering,
            "vulkan",
            ProfileOverride::Balanced,
        );
        assert_eq!(discrete_16g.memory_tier, MemoryTier::Standard);
        assert!(discrete_16g.recommended_cpu_embeddings);
        assert_eq!(discrete_16g.recommended_io_threads, 2);
        assert_eq!(discrete_16g.label(), "standard-discrete-vulkan-win_nobuf");

        let forced_edge = HardwareProfile::from_inputs(
            64 * GIB,
            MemoryTopology::Unified,
            StorageStack::MacOsNoCache,
            "metal",
            ProfileOverride::Edge,
        );
        assert_eq!(forced_edge.memory_tier, MemoryTier::Edge);
        assert!(forced_edge.recommended_stream_head);
        assert_eq!(forced_edge.label(), "edge-unified-metal-macos_nocache");

        let generic_perf = HardwareProfile::from_inputs(
            8 * GIB,
            MemoryTopology::Unified,
            StorageStack::Generic,
            "cpu",
            ProfileOverride::Performance,
        );
        assert_eq!(generic_perf.memory_tier, MemoryTier::High);
        assert_eq!(generic_perf.recommended_io_threads, 1);
        assert_eq!(generic_perf.label(), "high-unified-cpu-generic");
    }

    #[test]
    fn parses_profile_override_meminfo_and_mountinfo() {
        assert_eq!(
            "auto".parse::<ProfileOverride>().unwrap(),
            ProfileOverride::Auto
        );
        assert_eq!(
            "edge".parse::<ProfileOverride>().unwrap(),
            ProfileOverride::Edge
        );
        assert_eq!(
            "balanced".parse::<ProfileOverride>().unwrap(),
            ProfileOverride::Balanced
        );
        assert_eq!(
            "performance".parse::<ProfileOverride>().unwrap(),
            ProfileOverride::Performance
        );
        assert_eq!(ProfileOverride::Auto.to_string(), "auto");
        assert_eq!(ProfileOverride::Edge.to_string(), "edge");
        assert_eq!(ProfileOverride::Balanced.to_string(), "balanced");
        assert_eq!(ProfileOverride::Performance.to_string(), "performance");
        assert!("bogus".parse::<ProfileOverride>().is_err());

        let mem = "MemTotal:       7659520 kB\nMemFree:        100000 kB\n";
        assert_eq!(parse_proc_memtotal_bytes(mem), Some(7659520 * 1024));
        assert_eq!(parse_proc_memtotal_bytes("NothingHere: 1"), None);

        let mountinfo = "\
24 1 0:21 / / rw,relatime - tmpfs none rw
50 24 254:0 / /mnt/models rw,relatime - ext4 /dev/mapper/vg_orin-models rw
51 24 259:1 / /mnt/fast rw,relatime - ext4 /dev/nvme0n1p1 rw
";
        assert!(is_mountinfo_dmcrypt(
            mountinfo,
            Path::new("/mnt/models/bench/model.gguf")
        ));
        assert!(!is_mountinfo_dmcrypt(
            mountinfo,
            Path::new("/mnt/fast/model.gguf")
        ));

        let detected = HardwareProfile::detect(Path::new("/"), false, false, ProfileOverride::Auto);
        assert_eq!(detected.backend, "cpu");
    }
}
