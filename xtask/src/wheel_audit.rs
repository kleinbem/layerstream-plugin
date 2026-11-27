//! Strict ELF & PyPI compliance auditor for Linux `layerstream` and `layerstream-cuda13` wheels.
//!
//! Workaround: `auditwheel repair` hashes/renames runtime-`dlopen`ed `libggml-*.so` backend plugins
//! (upstream: pypa/auditwheel#594), so Linux wheels build with `--auditwheel skip` and enforce
//! `manylinux_2_34` symbol/RUNPATH bounds here.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const WARN_WHEEL_BYTES: u64 = 90 * 1024 * 1024;
pub const MAX_WHEEL_BYTES: u64 = 98 * 1024 * 1024;
pub const MAX_GLIBC: (u32, u32) = (2, 34);
pub const MAX_GLIBCXX: (u32, u32, u32) = (3, 4, 29);
pub const MAX_CXXABI: (u32, u32, u32) = (1, 3, 13);

const FORBIDDEN_RPATH_SUBSTRINGS: &[&str] = &[
    "/opt/",
    "/nix/",
    "/home/",
    "/tmp/",
    "/root/",
    "/github/",
    "/project/",
    "stubs",
];

fn parse_ver(raw: &str) -> Vec<u32> {
    raw.split('.')
        .filter_map(|p| p.parse::<u32>().ok())
        .collect()
}

fn extract_bracketed_after(line: &str, marker: &str) -> Option<String> {
    let idx = line.find(marker)?;
    let rest = &line[idx + marker.len()..];
    let open = rest.find('[')?;
    let after = &rest[open + 1..];
    let close = after.find(']')?;
    Some(after[..close].to_string())
}

fn parse_needed(dyn_out: &str) -> Vec<String> {
    dyn_out
        .lines()
        .filter(|l| l.contains("(NEEDED)"))
        .filter_map(|l| extract_bracketed_after(l, "Shared library:"))
        .collect()
}

fn parse_rpaths(dyn_out: &str) -> Vec<String> {
    dyn_out
        .lines()
        .filter(|l| l.contains("(RPATH)") || l.contains("(RUNPATH)"))
        .filter_map(|l| {
            extract_bracketed_after(l, "Library runpath:")
                .or_else(|| extract_bracketed_after(l, "Library rpath:"))
        })
        .collect()
}

fn extract_prefixed_versions(ver_out: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = ver_out.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if let Some(pos) = ver_out[i..].find(prefix) {
            let start = i + pos + prefix.len();
            let mut end = start;
            while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
                end += 1;
            }
            let cand = ver_out[start..end].trim_end_matches('.');
            if cand.contains('.') && cand.chars().all(|c| c.is_ascii_digit() || c == '.') {
                out.push(cand.to_string());
            }
            i = end.max(start + 1);
        } else {
            break;
        }
    }
    out
}

pub fn check_elf_metadata(
    member: &str,
    is_base_wheel: bool,
    dyn_out: &str,
    ver_out: &str,
) -> Result<(), String> {
    let needed = parse_needed(dyn_out);
    let rpaths = parse_rpaths(dyn_out);
    println!("[audit]   {member}: NEEDED={needed:?} RUNPATH={rpaths:?}");

    for rp in &rpaths {
        if rp.split(':').any(str::is_empty) {
            return Err(format!(
                "ERROR: empty CWD segment in RUNPATH of {member}: {rp:?}"
            ));
        }
        for bad in FORBIDDEN_RPATH_SUBSTRINGS {
            if rp.contains(bad) {
                return Err(format!(
                    "ERROR: forbidden build path '{bad}' in RUNPATH of {member}: {rp}"
                ));
            }
        }
    }

    let member_name = Path::new(member)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(member);
    if is_base_wheel && member_name.contains("_layerstream") {
        for lib in &needed {
            if ["libvulkan", "libcudart", "libcuda.", "libcublas"]
                .iter()
                .any(|k| lib.contains(k))
            {
                return Err(format!(
                    "ERROR: base extension {member} directly links GPU driver/runtime {lib}"
                ));
            }
        }
    }

    for raw in extract_prefixed_versions(ver_out, "GLIBC_") {
        let v = parse_ver(&raw);
        if v.len() >= 2 && (v[0], v[1]) > MAX_GLIBC {
            return Err(format!(
                "ERROR: {member} requires GLIBC_{raw} > {MAX_GLIBC:?}"
            ));
        }
    }
    for raw in extract_prefixed_versions(ver_out, "GLIBCXX_") {
        let v = parse_ver(&raw);
        if v.len() >= 3 && (v[0], v[1], v[2]) > MAX_GLIBCXX {
            return Err(format!(
                "ERROR: {member} requires GLIBCXX_{raw} > {MAX_GLIBCXX:?}"
            ));
        }
    }
    for raw in extract_prefixed_versions(ver_out, "CXXABI_") {
        let v = parse_ver(&raw);
        if v.len() >= 3 && (v[0], v[1], v[2]) > MAX_CXXABI {
            return Err(format!(
                "ERROR: {member} requires CXXABI_{raw} > {MAX_CXXABI:?}"
            ));
        }
    }

    Ok(())
}

fn list_zip_members(wheel_path: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("unzip")
        .args(["-Z1"])
        .arg(wheel_path)
        .output()
        .map_err(|e| format!("failed to run unzip -Z1: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "unzip -Z1 {} failed: {}",
            wheel_path.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect())
}

fn is_shared_object_member(member: &str) -> bool {
    let name = Path::new(member)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(member);
    member.ends_with(".so") || name.contains(".so.")
}

fn audit_wheel_with_limits(
    wheel_path: &Path,
    warn_bytes: u64,
    max_bytes: u64,
) -> Result<(), String> {
    if !wheel_path.is_file() {
        return Err(format!("ERROR: wheel not found: {}", wheel_path.display()));
    }
    let name = wheel_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("wheel.whl");
    let size = fs::metadata(wheel_path)
        .map_err(|e| format!("stat {}: {e}", wheel_path.display()))?
        .len();
    let size_mib = size as f64 / (1024.0 * 1024.0);
    let headroom_mib = 100.0 - size_mib;
    println!("[audit] {name}: {size_mib:.2} MiB ({headroom_mib:.2} MiB under PyPI 100 MiB cap)");
    if size > max_bytes {
        return Err(format!(
            "ERROR: {name} is {size_mib:.2} MiB, exceeding PyPI 100 MiB safety limit"
        ));
    }
    if size > warn_bytes {
        println!(
            "::warning::{name} is {size_mib:.2} MiB ({headroom_mib:.2} MiB headroom below PyPI's 100 MiB cap)"
        );
    }
    if name.contains("-linux_") {
        return Err(format!(
            "ERROR: un-tagged linux_* wheel will be rejected by PyPI: {name}"
        ));
    }

    let is_base_wheel = name.starts_with("layerstream-");
    let members = list_zip_members(wheel_path)?;
    let so_members: Vec<String> = members
        .into_iter()
        .filter(|m| is_shared_object_member(m))
        .collect();
    if so_members.is_empty() {
        return Err(format!("ERROR: no shared objects found in {name}"));
    }

    let tmp_dir = std::env::temp_dir().join(format!(
        "layerstream-wheel-audit-{}-{}",
        std::process::id(),
        size
    ));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("create temp dir: {e}"))?;

    let result = (|| -> Result<(), String> {
        let mut cmd = Command::new("unzip");
        cmd.args(["-q", "-o"]).arg(wheel_path);
        for m in &so_members {
            cmd.arg(m);
        }
        cmd.arg("-d").arg(&tmp_dir);
        let st = cmd
            .status()
            .map_err(|e| format!("failed running unzip extract: {e}"))?;
        if !st.success() {
            return Err(format!("unzip extract failed for {name}"));
        }

        for member in &so_members {
            let extracted = tmp_dir.join(member);
            let dyn_out = run_readelf("-d", &extracted)?;
            let ver_out = run_readelf("-V", &extracted)?;
            check_elf_metadata(member, is_base_wheel, &dyn_out, &ver_out)?;
        }
        Ok(())
    })();

    let _ = fs::remove_dir_all(&tmp_dir);
    result?;
    println!("[audit] PASS: {name}");
    Ok(())
}

fn run_readelf(flag: &str, path: &Path) -> Result<String, String> {
    let out = Command::new("readelf")
        .arg(flag)
        .arg(path)
        .output()
        .map_err(|e| format!("failed running readelf {flag} {}: {e}", path.display()))?;
    if !out.status.success() {
        return Err(format!(
            "readelf {flag} {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn cmd_audit_wheel_elf(paths: &[String]) -> Result<(), String> {
    if paths.is_empty() {
        return Err("usage: cargo xtask audit-wheel-elf <wheel.whl> [...]".to_string());
    }
    for p in paths {
        audit_wheel_with_limits(&PathBuf::from(p), WARN_WHEEL_BYTES, MAX_WHEEL_BYTES)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_elf_metadata_accepts_valid_manylinux_2_34_and_rejects_violations() {
        let ok_dyn = " 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]\n\
                      0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN]\n";
        let ok_ver = "GLIBC_2.34\nGLIBCXX_3.4.29\nCXXABI_1.3.13\n";
        assert!(
            check_elf_metadata("layerstream/_layerstream.abi3.so", true, ok_dyn, ok_ver).is_ok()
        );

        // Empty CWD segment in RUNPATH
        let empty_rp = " 0x000000000000001d (RUNPATH) Library runpath: [:$ORIGIN]\n";
        assert!(
            check_elf_metadata("layerstream/_backends/libggml.so", true, empty_rp, ok_ver)
                .unwrap_err()
                .contains("empty CWD segment")
        );

        // Forbidden build path in RUNPATH
        let nix_rp = " 0x000000000000001d (RUNPATH) Library runpath: [/nix/store/abc/lib]\n";
        assert!(
            check_elf_metadata("layerstream/_backends/libggml.so", true, nix_rp, ok_ver)
                .unwrap_err()
                .contains("forbidden build path '/nix/'")
        );

        // Base extension directly linking GPU driver
        let gpu_dyn = " 0x0000000000000001 (NEEDED) Shared library: [libvulkan.so.1]\n";
        assert!(
            check_elf_metadata("layerstream/_layerstream.abi3.so", true, gpu_dyn, ok_ver)
                .unwrap_err()
                .contains("directly links GPU driver/runtime")
        );

        // GLIBC / GLIBCXX / CXXABI ceiling violations
        assert!(
            check_elf_metadata("lib.so", false, ok_dyn, "GLIBC_2.38\n")
                .unwrap_err()
                .contains("GLIBC_2.38")
        );
        assert!(
            check_elf_metadata("lib.so", false, ok_dyn, "GLIBCXX_3.4.30\n")
                .unwrap_err()
                .contains("GLIBCXX_3.4.30")
        );
        assert!(
            check_elf_metadata("lib.so", false, ok_dyn, "CXXABI_1.3.14\n")
                .unwrap_err()
                .contains("CXXABI_1.3.14")
        );
    }

    #[test]
    fn audit_wheel_rejects_missing_oversized_and_untagged_linux_wheels() {
        assert!(
            audit_wheel_with_limits(
                Path::new("/nonexistent/demo.whl"),
                WARN_WHEEL_BYTES,
                MAX_WHEEL_BYTES
            )
            .unwrap_err()
            .contains("wheel not found")
        );

        let dir = std::env::temp_dir().join(format!("ls-wheel-audit-unit-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);

        let oversized = dir.join("layerstream_cuda13-0.5.0-py3-none-manylinux_2_34_x86_64.whl");
        fs::write(&oversized, vec![0u8; 32]).unwrap();
        assert!(
            audit_wheel_with_limits(&oversized, 5, 10)
                .unwrap_err()
                .contains("exceeding PyPI 100 MiB safety limit")
        );

        let untagged = dir.join("layerstream-0.5.0-cp314-abi3-linux_x86_64.whl");
        fs::write(&untagged, vec![0u8; 32]).unwrap();
        assert!(
            audit_wheel_with_limits(&untagged, WARN_WHEEL_BYTES, MAX_WHEEL_BYTES)
                .unwrap_err()
                .contains("un-tagged linux_* wheel")
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
