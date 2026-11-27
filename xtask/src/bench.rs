//! Pure-Rust bare-metal benchmark round runner (`bench-round`) and regression
//! checker (`check-regressions`) for LayerStream hosts (e.g. Jetson Orin Nano).
//!
//! Uses only the Rust standard library (zero external dependencies) so it compiles
//! in < 1s and runs on bare-metal NixOS/Linux hosts without requiring Python.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Debug, Default, Clone)]
pub(crate) struct BenchRecord {
    pub model: String,
    pub mode: String,
    pub workload: String,
    pub decode_tok_per_s: Option<f64>,
    pub ttft_s: Option<f64>,
    pub turn2_ttft_s: Option<f64>,
    pub ttft_speedup_x: Option<f64>,
    pub peak_rss_mib: Option<f64>,
    pub swap_delta_mib: Option<f64>,
    pub joules_per_tok: Option<f64>,
    pub using_gpu: Option<bool>,
    pub backend_name: Option<String>,
}

pub(crate) fn canonical_mode(mode: &str) -> String {
    if let Some(rest) = mode.strip_prefix('r') {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && rest.as_bytes().get(digits) == Some(&b'_') {
            return rest[digits + 1..].to_string();
        }
    }
    mode.to_string()
}

/// Parses Linux `/proc/meminfo` for `(mem_avail_mib, swap_used_mib)`.
pub(crate) fn parse_meminfo_mib(meminfo: &str) -> (Option<f64>, Option<f64>) {
    let mut avail_kb: Option<u64> = None;
    let mut swap_total_kb: Option<u64> = None;
    let mut swap_free_kb: Option<u64> = None;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
        } else if let Some(rest) = line.strip_prefix("SwapTotal:") {
            swap_total_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
        } else if let Some(rest) = line.strip_prefix("SwapFree:") {
            swap_free_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
        }
    }
    let avail = avail_kb.map(|kb| kb as f64 / 1024.0);
    let swap_used = match (swap_total_kb, swap_free_kb) {
        (Some(t), Some(f)) => Some(t.saturating_sub(f) as f64 / 1024.0),
        _ => None,
    };
    (avail, swap_used)
}

/// Parses Linux `/sys/block/zram*/mm_stat` for `(orig_data_bytes, mem_used_total_bytes)`.
pub(crate) fn parse_zram_mm_stat(raw: &str) -> Option<(u64, u64)> {
    let mut parts = raw.split_whitespace();
    let orig: u64 = parts.next()?.parse().ok()?;
    let _compr: u64 = parts.next()?.parse().ok()?;
    let mem_used: u64 = parts.next()?.parse().ok()?;
    Some((orig, mem_used))
}

fn read_zram_stats_mib() -> (f64, f64) {
    let mut orig_bytes = 0u64;
    let mut phys_bytes = 0u64;
    let Ok(entries) = fs::read_dir("/sys/block") else {
        return (0.0, 0.0);
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("zram") {
            continue;
        }
        if let Ok(raw) = fs::read_to_string(entry.path().join("mm_stat"))
            && let Some((orig, phys)) = parse_zram_mm_stat(&raw)
        {
            orig_bytes = orig_bytes.saturating_add(orig);
            phys_bytes = phys_bytes.saturating_add(phys);
        }
    }
    let mib = 1024.0 * 1024.0;
    (orig_bytes as f64 / mib, phys_bytes as f64 / mib)
}

/// Minimal zero-dependency parser for flat JSON benchmark record arrays (`[{...}, ...]`).
/// Rejects truncated, unclosed, or malformed top-level JSON arrays/objects.
pub(crate) fn parse_bench_json_array(text: &str) -> Result<Vec<BenchRecord>, String> {
    let trimmed = text.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("expected a complete JSON array '[...]'".to_string());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = Vec::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
    let mut depth = 0usize;
    let mut obj_start: Option<usize> = None;
    let mut in_str = false;
    let mut escape = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                if depth == 0 {
                    return Err("unexpected '}' in benchmark JSON".to_string());
                }
                depth -= 1;
                if depth == 0
                    && let Some(start) = obj_start.take()
                {
                    out.push(parse_bench_object(&inner[start..=i]));
                }
            }
            b',' | b' ' | b'\n' | b'\r' | b'\t' if depth == 0 => {}
            _ if depth == 0 => {
                return Err(format!(
                    "unexpected token '{}' outside JSON object in benchmark array",
                    b as char
                ));
            }
            _ => {}
        }
        i += 1;
    }
    if in_str || depth != 0 || obj_start.is_some() {
        return Err("incomplete or truncated JSON object in benchmark file".to_string());
    }
    Ok(out)
}

fn find_json_field<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let mut search = obj;
    while let Some(idx) = search.find(&needle) {
        let after = search[idx + needle.len()..].trim_start();
        if let Some(rest) = after.strip_prefix(':') {
            return Some(rest.trim_start());
        }
        search = &search[idx + needle.len()..];
    }
    None
}

fn json_str_field(obj: &str, key: &str) -> Option<String> {
    let val = find_json_field(obj, key)?;
    let rest = val.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => {
                if let Some(esc) = chars.next() {
                    out.push(esc);
                }
            }
            other => out.push(other),
        }
    }
    None
}

fn json_f64_field(obj: &str, key: &str) -> Option<f64> {
    let val = find_json_field(obj, key)?;
    if val.starts_with("null") {
        return None;
    }
    let end = val
        .find([',', '}', ']', ' ', '\n', '\r', '\t'])
        .unwrap_or(val.len());
    val[..end].parse::<f64>().ok()
}

fn json_bool_field(obj: &str, key: &str) -> Option<bool> {
    let val = find_json_field(obj, key)?;
    if val.starts_with("true") {
        Some(true)
    } else if val.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_bench_object(obj: &str) -> BenchRecord {
    BenchRecord {
        model: json_str_field(obj, "model").unwrap_or_else(|| "unknown".to_string()),
        mode: canonical_mode(&json_str_field(obj, "mode").unwrap_or_default()),
        workload: json_str_field(obj, "workload").unwrap_or_default(),
        decode_tok_per_s: json_f64_field(obj, "decode_tok_per_s"),
        ttft_s: json_f64_field(obj, "ttft_s"),
        turn2_ttft_s: json_f64_field(obj, "turn2_ttft_s"),
        ttft_speedup_x: json_f64_field(obj, "ttft_speedup_x"),
        peak_rss_mib: json_f64_field(obj, "peak_rss_mib"),
        swap_delta_mib: json_f64_field(obj, "swap_delta_mib"),
        joules_per_tok: json_f64_field(obj, "joules_per_tok"),
        using_gpu: json_bool_field(obj, "using_gpu"),
        backend_name: json_str_field(obj, "backend_name"),
    }
}

type RecordKey = (String, String, String);

fn load_round_records(round_dir: &Path) -> Result<BTreeMap<RecordKey, BenchRecord>, String> {
    let mut records = BTreeMap::new();
    let mut paths: Vec<PathBuf> = fs::read_dir(round_dir)
        .map_err(|e| format!("failed to read {}: {e}", round_dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();

    for path in paths {
        if path.file_name().and_then(|s| s.to_str()) == Some("hardware.json") {
            continue;
        }
        let meta = fs::metadata(&path).map_err(|e| format!("stat {}: {e}", path.display()))?;
        if meta.len() == 0 {
            continue;
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if !text.trim_start().starts_with('[') {
            continue;
        }
        let parsed = parse_bench_json_array(&text)
            .map_err(|e| format!("malformed benchmark JSON {}: {e}", path.display()))?;
        for rec in parsed {
            let key = (rec.model.clone(), rec.mode.clone(), rec.workload.clone());
            records.insert(key, rec);
        }
    }
    Ok(records)
}

fn extract_graph_splits(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.contains("sched_reserve:") || !line.contains("graph:") {
            continue;
        }
        if let Some(idx) = line.find("splits") {
            let after = line[idx + "splits".len()..].trim_start();
            if let Some(eq) = after.strip_prefix('=') {
                let digits: String = eq
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                if let Ok(n) = digits.parse::<usize>() {
                    out.push(n);
                }
            }
        }
    }
    out
}

fn has_nvme_timeout(text: &str) -> bool {
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("timeout") && (lower.contains("nvme") || lower.contains("i/o tag")) {
            return true;
        }
    }
    false
}

pub(crate) fn scan_logs(logs_dir: &Path, max_splits: usize) -> Vec<String> {
    let mut failures = Vec::new();
    let Ok(entries) = fs::read_dir(logs_dir) else {
        return failures;
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("log"))
        .collect();
    paths.sort();

    for log_path in paths {
        let name = log_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("log");
        let Ok(bytes) = fs::read(&log_path) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        let cpu_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("assigned to device CPU"))
            .collect();
        if !cpu_lines.is_empty() {
            failures.push(format!(
                "FAIL [log:{name}] {} layer(s) silently assigned to CPU (first: {})",
                cpu_lines.len(),
                cpu_lines[0].trim()
            ));
        }
        for splits in extract_graph_splits(&text) {
            if splits > max_splits {
                failures.push(format!(
                    "FAIL [log:{name}] ggml graph splits={splits} exceeds max {max_splits}"
                ));
                break;
            }
        }
        if has_nvme_timeout(&text) {
            failures.push(format!(
                "FAIL [log:{name}] NVMe/PCIe I/O timeout detected in log"
            ));
        }
    }
    failures
}

struct RegressionThresholds {
    max_tps_drop_pct: f64,
    max_ttft_rise_pct: f64,
    max_rss_jump_mib: f64,
    max_swap_mib: f64,
    max_splits: usize,
    strict: bool,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            max_tps_drop_pct: 10.0,
            max_ttft_rise_pct: 15.0,
            max_rss_jump_mib: 350.0,
            max_swap_mib: 32.0,
            max_splits: 4,
            strict: false,
        }
    }
}

fn check_pair(
    key: &RecordKey,
    base: Option<&BenchRecord>,
    cand: &BenchRecord,
    cfg: &RegressionThresholds,
) -> (String, Vec<String>, Vec<String>) {
    let (model, mode, workload) = key;
    let mut warns = Vec::new();
    let mut fails = Vec::new();

    let c_swap = cand.swap_delta_mib.unwrap_or(0.0);
    if c_swap > cfg.max_swap_mib {
        fails.push(format!(
            "FAIL [{model} {workload}] swap delta +{c_swap:.1} MiB exceeds {:.1} MiB",
            cfg.max_swap_mib
        ));
    }

    if cand.using_gpu == Some(true)
        && cand
            .backend_name
            .as_deref()
            .is_some_and(|b| b.eq_ignore_ascii_case("cpu"))
    {
        fails.push(format!(
            "FAIL [{model} {workload}] GPU requested but backend_name is CPU"
        ));
    }

    if workload == "multi_turn_prefix_reuse" {
        let b_t2 = base
            .and_then(|b| b.turn2_ttft_s)
            .map_or_else(|| "—".to_string(), |v| format!("{v:.3}"));
        let c_t2 = cand
            .turn2_ttft_s
            .map_or_else(|| "—".to_string(), |v| format!("{v:.3}"));
        let spd = cand
            .ttft_speedup_x
            .map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
        let row = format!(
            "| `{model}` | `{mode}` | `multi_turn` | {b_t2} s | {c_t2} s | {spd}x TTFT | — | — |"
        );
        return (row, warns, fails);
    }

    let b_tps = base.and_then(|b| b.decode_tok_per_s);
    let c_tps = cand.decode_tok_per_s;
    let b_ttft = base.and_then(|b| b.ttft_s);
    let c_ttft = cand.ttft_s;
    let b_rss = base.and_then(|b| b.peak_rss_mib);
    let c_rss = cand.peak_rss_mib;
    let c_j = cand.joules_per_tok;

    let mut delta_str = "new".to_string();
    if let (Some(bt), Some(ct)) = (b_tps, c_tps)
        && bt > 0.0
    {
        let pct = (ct / bt - 1.0) * 100.0;
        delta_str = format!("{pct:+.1}%");
        if pct < -cfg.max_tps_drop_pct {
            warns.push(format!(
                "WARN [{model} {workload}] decode tok/s dropped {pct:+.1}% ({bt:.3} -> {ct:.3})"
            ));
        }
    }

    if let (Some(bt), Some(ct)) = (b_ttft, c_ttft)
        && bt > 0.0
    {
        let ttft_pct = (ct / bt - 1.0) * 100.0;
        if ttft_pct > cfg.max_ttft_rise_pct {
            warns.push(format!(
                "WARN [{model} {workload}] TTFT rose {ttft_pct:+.1}% ({bt:.2}s -> {ct:.2}s)"
            ));
        }
    }

    if let (Some(br), Some(cr)) = (b_rss, c_rss) {
        let rss_delta = cr - br;
        if rss_delta > cfg.max_rss_jump_mib {
            fails.push(format!(
                "FAIL [{model} {workload}] peak RSS jumped +{rss_delta:.0} MiB ({br:.0} -> {cr:.0} MiB)"
            ));
        }
    }

    let b_tps_s = b_tps.map_or_else(|| "—".to_string(), |v| format!("{v:.3}"));
    let c_tps_s = c_tps.map_or_else(|| "—".to_string(), |v| format!("{v:.3}"));
    let c_rss_s = c_rss.map_or_else(|| "—".to_string(), |v| format!("{v:.0} MiB"));
    let c_j_s = c_j.map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
    let row = format!(
        "| `{model}` | `{mode}` | `{workload}` | {b_tps_s} | {c_tps_s} | {delta_str} | {c_rss_s} | {c_j_s} |"
    );
    (row, warns, fails)
}

fn run_check_regressions(
    baseline: &Path,
    candidate: &Path,
    logs_dir: Option<&Path>,
    cfg: &RegressionThresholds,
) -> Result<(), String> {
    let base_map = load_round_records(baseline)?;
    let cand_map = load_round_records(candidate)?;
    if cand_map.is_empty() {
        return Err(format!(
            "no benchmark JSON records found in {}",
            candidate.display()
        ));
    }

    println!(
        "| Model | Mode | Workload | Baseline `tok/s` | Candidate `tok/s` | Delta | Peak RSS | `J/tok` |"
    );
    println!("|---|---|---|---:|---:|---:|---:|---:|");

    let mut all_warns = Vec::new();
    let mut all_fails = Vec::new();
    for (key, cand) in &cand_map {
        let (row, warns, fails) = check_pair(key, base_map.get(key), cand, cfg);
        println!("{row}");
        all_warns.extend(warns);
        all_fails.extend(fails);
    }

    let logs = logs_dir.unwrap_or(candidate);
    if logs.is_dir() {
        all_fails.extend(scan_logs(logs, cfg.max_splits));
    }

    println!();
    if !all_warns.is_empty() {
        println!("### Warnings");
        for w in &all_warns {
            println!("- {w}");
        }
    }
    if !all_fails.is_empty() {
        println!("### Failures");
        for f in &all_fails {
            println!("- {f}");
        }
    }

    if !all_fails.is_empty() {
        return Err(format!(
            "regression check failed with {} failure(s)",
            all_fails.len()
        ));
    }
    if cfg.strict && !all_warns.is_empty() {
        return Err(format!(
            "strict regression check failed with {} warning(s)",
            all_warns.len()
        ));
    }
    let base_name = baseline
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("baseline");
    println!(
        "PASS: {} records verified against {base_name}.",
        cand_map.len()
    );
    Ok(())
}

fn next_flag_val<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {flag}"))
}

pub fn cmd_check_regressions(args: &[String]) -> Result<(), String> {
    let mut baseline: Option<PathBuf> = None;
    let mut candidate: Option<PathBuf> = None;
    let mut logs_dir: Option<PathBuf> = None;
    let mut cfg = RegressionThresholds::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => {
                baseline = Some(PathBuf::from(next_flag_val(args, &mut i, "--baseline")?))
            }
            "--candidate" => {
                candidate = Some(PathBuf::from(next_flag_val(args, &mut i, "--candidate")?))
            }
            "--logs-dir" => {
                logs_dir = Some(PathBuf::from(next_flag_val(args, &mut i, "--logs-dir")?))
            }
            "--max-tps-drop-pct" => {
                cfg.max_tps_drop_pct = next_flag_val(args, &mut i, "--max-tps-drop-pct")?
                    .parse()
                    .map_err(|_| "invalid --max-tps-drop-pct".to_string())?;
            }
            "--max-ttft-rise-pct" => {
                cfg.max_ttft_rise_pct = next_flag_val(args, &mut i, "--max-ttft-rise-pct")?
                    .parse()
                    .map_err(|_| "invalid --max-ttft-rise-pct".to_string())?;
            }
            "--max-rss-jump-mib" => {
                cfg.max_rss_jump_mib = next_flag_val(args, &mut i, "--max-rss-jump-mib")?
                    .parse()
                    .map_err(|_| "invalid --max-rss-jump-mib".to_string())?;
            }
            "--max-swap-mib" => {
                cfg.max_swap_mib = next_flag_val(args, &mut i, "--max-swap-mib")?
                    .parse()
                    .map_err(|_| "invalid --max-swap-mib".to_string())?;
            }
            "--max-splits" => {
                cfg.max_splits = next_flag_val(args, &mut i, "--max-splits")?
                    .parse()
                    .map_err(|_| "invalid --max-splits".to_string())?;
            }
            "--strict" => cfg.strict = true,
            other => return Err(format!("unknown check-regressions flag: {other}")),
        }
        i += 1;
    }

    let baseline = baseline
        .ok_or("usage: cargo xtask check-regressions --baseline <dir> --candidate <dir>")?;
    let candidate = candidate
        .ok_or("usage: cargo xtask check-regressions --baseline <dir> --candidate <dir>")?;
    run_check_regressions(&baseline, &candidate, logs_dir.as_deref(), &cfg)
}

fn run_case(
    bench_bin: &Path,
    out_dir: &Path,
    name: &str,
    model: &Path,
    extra_args: &[String],
    cooldown_s: u64,
) -> Result<(), String> {
    let json_path = out_dir.join(format!("{name}.json"));
    let log_path = out_dir.join(format!("{name}.log"));
    println!("==> Running {name} ...");
    let jf =
        File::create(&json_path).map_err(|e| format!("create {}: {e}", json_path.display()))?;
    let lf = File::create(&log_path).map_err(|e| format!("create {}: {e}", log_path.display()))?;

    let status = Command::new(bench_bin)
        .arg("--model")
        .arg(model)
        .args(extra_args)
        .stdout(Stdio::from(jf))
        .stderr(Stdio::from(lf))
        .status()
        .map_err(|e| format!("spawn {}: {e}", bench_bin.display()))?;
    if !status.success() {
        return Err(format!("bench case '{name}' exited with status {status}"));
    }
    if cooldown_s > 0 {
        thread::sleep(Duration::from_secs(cooldown_s));
    }
    Ok(())
}

pub fn cmd_bench_round(args: &[String]) -> Result<(), String> {
    let mut round: Option<String> = None;
    let mut bench_bin: Option<PathBuf> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut baseline: Option<PathBuf> = None;
    let mut mtp_model: Option<PathBuf> = None;
    let mut hybrid_model: Option<PathBuf> = None;
    let mut cooldown_s: u64 = 3;
    let mut min_mem_mib: f64 = 3800.0;
    let mut max_zram_phys_mib: f64 = 256.0;
    let mut max_disk_swap_mib: f64 = 64.0;
    let mut allow_dirty_swap = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--round" => round = Some(next_flag_val(args, &mut i, "--round")?.to_string()),
            "--bench" => bench_bin = Some(PathBuf::from(next_flag_val(args, &mut i, "--bench")?)),
            "--models-dir" => {
                models_dir = Some(PathBuf::from(next_flag_val(args, &mut i, "--models-dir")?));
            }
            "--out-dir" => out_dir = Some(PathBuf::from(next_flag_val(args, &mut i, "--out-dir")?)),
            "--baseline" => {
                baseline = Some(PathBuf::from(next_flag_val(args, &mut i, "--baseline")?))
            }
            "--mtp-model" => {
                mtp_model = Some(PathBuf::from(next_flag_val(args, &mut i, "--mtp-model")?))
            }
            "--hybrid-model" => {
                hybrid_model = Some(PathBuf::from(next_flag_val(
                    args,
                    &mut i,
                    "--hybrid-model",
                )?));
            }
            "--cooldown" => {
                cooldown_s = next_flag_val(args, &mut i, "--cooldown")?
                    .parse()
                    .map_err(|_| "invalid --cooldown".to_string())?;
            }
            "--min-mem-mib" => {
                min_mem_mib = next_flag_val(args, &mut i, "--min-mem-mib")?
                    .parse()
                    .map_err(|_| "invalid --min-mem-mib".to_string())?;
            }
            "--max-zram-phys-mib" => {
                max_zram_phys_mib = next_flag_val(args, &mut i, "--max-zram-phys-mib")?
                    .parse()
                    .map_err(|_| "invalid --max-zram-phys-mib".to_string())?;
            }
            "--max-disk-swap-mib" => {
                max_disk_swap_mib = next_flag_val(args, &mut i, "--max-disk-swap-mib")?
                    .parse()
                    .map_err(|_| "invalid --max-disk-swap-mib".to_string())?;
            }
            "--allow-dirty-swap" => allow_dirty_swap = true,
            other => return Err(format!("unknown bench-round flag: {other}")),
        }
        i += 1;
    }

    let r = round.ok_or("usage: cargo xtask bench-round --round <r> --bench <bin> --models-dir <dir> --out-dir <dir>")?;
    let bench_bin = bench_bin.ok_or("missing --bench <path>")?;
    let m = models_dir.ok_or("missing --models-dir <dir>")?;
    let out_dir = out_dir.ok_or("missing --out-dir <dir>")?;

    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        let (mem_avail, swap_used) = parse_meminfo_mib(&meminfo);
        if let Some(avail) = mem_avail {
            println!("Host MemAvailable: {avail:.0} MiB");
            if avail < min_mem_mib {
                eprintln!(
                    "WARNING: MemAvailable ({avail:.0} MiB) < {min_mem_mib:.0} MiB — \
                     stop heavy background services (e.g. mesh-llm, frigate) for clean numbers!"
                );
            }
        }
        let (zram_orig_mib, zram_phys_mib) = read_zram_stats_mib();
        if zram_phys_mib > 0.0 {
            println!(
                "Host zram: {zram_phys_mib:.1} MiB physical RAM ({zram_orig_mib:.0} MiB logical)"
            );
        }
        if let Some(total_swap) = swap_used {
            let disk_swap = (total_swap - zram_orig_mib).max(0.0);
            println!("Host SwapUsed: {total_swap:.0} MiB (disk={disk_swap:.0} MiB)");
            let dirty_zram = zram_phys_mib > max_zram_phys_mib;
            let dirty_disk = disk_swap > max_disk_swap_mib;
            if dirty_zram || dirty_disk {
                let msg = format!(
                    "Host swap is dirty (zram_phys={zram_phys_mib:.1} MiB > {max_zram_phys_mib:.0} MiB or \
                     disk_swap={disk_swap:.0} MiB > {max_disk_swap_mib:.0} MiB). \
                     Clear swap (e.g. `sudo swapoff /dev/zram0 && sudo swapon /dev/zram0` or reboot) \
                     or pass --allow-dirty-swap to override."
                );
                if allow_dirty_swap {
                    eprintln!("WARNING: {msg}");
                } else {
                    return Err(msg);
                }
            }
        }
    }

    if out_dir.is_dir() {
        let has_existing = fs::read_dir(&out_dir)
            .map_err(|e| format!("read_dir {}: {e}", out_dir.display()))?
            .filter_map(Result::ok)
            .any(|e| {
                matches!(
                    e.path().extension().and_then(|s| s.to_str()),
                    Some("json" | "log")
                )
            });
        if has_existing {
            return Err(format!(
                "output directory {} already contains .json/.log results; use a fresh --out-dir to avoid mixing rounds",
                out_dir.display()
            ));
        }
    }
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create_dir_all {}: {e}", out_dir.display()))?;

    let s = |v: &str| v.to_string();
    let draft_q25 = m
        .join("qwen2.5-0.5b-instruct-q4_k_m.gguf")
        .display()
        .to_string();
    let draft_q3 = m.join("Qwen3-0.6B-Q4_K_M.gguf").display().to_string();

    let mut cases: Vec<(String, PathBuf, Vec<String>)> = vec![
        (
            format!("{r}_olmoe_r0_cached"),
            m.join("OLMoE-1B-7B-0125-Instruct-Q2_K.gguf"),
            vec![
                s("--mode"),
                format!("{r}_olmoe_r0_sparse_cached"),
                s("--resident-layers"),
                s("0"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--keep-expert-cache"),
                s("--warmup"),
                s("--reps"),
                s("3"),
            ],
        ),
        (
            format!("{r}_14b_b5400_draft05b"),
            m.join("Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
            vec![
                s("--draft-model"),
                draft_q25.clone(),
                s("--mode"),
                format!("{r}_14b_b5400_draft05b"),
                s("--vram-budget-mb"),
                s("5400"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--warmup"),
                s("--reps"),
                s("3"),
            ],
        ),
        (
            format!("{r}_14b_auto_draft05b"),
            m.join("Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
            vec![
                s("--draft-model"),
                draft_q25.clone(),
                s("--mode"),
                format!("{r}_14b_auto_draft05b"),
                s("--vram-budget-mb"),
                s("auto"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--multi-turn"),
                s("--warmup"),
                s("--reps"),
                s("3"),
            ],
        ),
        (
            format!("{r}_qwen3_30b_a3b_draft06b_open"),
            m.join("Qwen3-30B-A3B-Q4_K_M.gguf"),
            vec![
                s("--draft-model"),
                draft_q3.clone(),
                s("--mode"),
                format!("{r}_qwen3_30b_a3b_r0_draft06b"),
                s("--resident-layers"),
                s("0"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--chat"),
                s("--workload"),
                s("open_ended"),
                s("--max-tokens"),
                s("32"),
                s("--reps"),
                s("2"),
            ],
        ),
        (
            format!("{r}_qwen3_30b_a3b_draft06b_code"),
            m.join("Qwen3-30B-A3B-Q4_K_M.gguf"),
            vec![
                s("--draft-model"),
                draft_q3,
                s("--mode"),
                format!("{r}_qwen3_30b_a3b_r0_draft06b"),
                s("--resident-layers"),
                s("0"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--chat"),
                s("--workload"),
                s("code_edit"),
                s("--max-tokens"),
                s("32"),
                s("--reps"),
                s("2"),
            ],
        ),
        (
            format!("{r}_32b_b5400_draft05b"),
            m.join("Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
            vec![
                s("--draft-model"),
                draft_q25,
                s("--mode"),
                format!("{r}_32b_b5400_draft05b"),
                s("--vram-budget-mb"),
                s("5400"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--reps"),
                s("2"),
            ],
        ),
    ];

    if let Some(mtp) = mtp_model {
        cases.push((
            format!("{r}_mtp_self_spec"),
            mtp,
            vec![
                s("--mode"),
                format!("{r}_mtp_self_spec"),
                s("--vram-budget-mb"),
                s("auto"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--chat"),
                s("--warmup"),
                s("--reps"),
                s("2"),
            ],
        ));
    }

    if let Some(hybrid) = hybrid_model {
        cases.push((
            format!("{r}_hybrid_rs_rollback"),
            hybrid,
            vec![
                s("--mode"),
                format!("{r}_hybrid_rs_rollback"),
                s("--vram-budget-mb"),
                s("auto"),
                s("--kv-cache-type"),
                s("q8_0"),
                s("--ngram"),
                s("--chat"),
                s("--warmup"),
                s("--reps"),
                s("2"),
            ],
        ));
    }

    for (name, model, extra) in &cases {
        if !model.is_file() {
            println!("SKIP {name}: missing model {}", model.display());
            continue;
        }
        if let Some(pos) = extra.iter().position(|a| a == "--draft-model")
            && let Some(draft) = extra.get(pos + 1)
            && !Path::new(draft).is_file()
        {
            println!("SKIP {name}: missing draft model {draft}");
            continue;
        }
        run_case(&bench_bin, &out_dir, name, model, extra, cooldown_s)?;
    }

    if let Some(base_dir) = baseline {
        println!(
            "\n==> Running regression check against {} ...",
            base_dir.display()
        );
        run_check_regressions(
            &base_dir,
            &out_dir,
            Some(&out_dir),
            &RegressionThresholds::default(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_mode_strips_round_prefix() {
        assert_eq!(
            canonical_mode("r12_olmoe_r0_sparse_cached"),
            "olmoe_r0_sparse_cached"
        );
        assert_eq!(canonical_mode("r9_14b_auto"), "14b_auto");
        assert_eq!(canonical_mode("baseline_mode"), "baseline_mode");
    }

    #[test]
    fn parses_meminfo_and_zram_mm_stat() {
        let meminfo = "MemTotal: 7658712 kB\nMemAvailable: 5836800 kB\nSwapTotal: 52246260 kB\nSwapFree: 52049652 kB\n";
        let (avail, swap) = parse_meminfo_mib(meminfo);
        assert!((avail.unwrap() - 5700.0).abs() < 1.0);
        assert!((swap.unwrap() - 192.0).abs() < 1.0);

        let mm = "201850880 16567296 55889920 0 55889920 12 0 1234 5678\n";
        assert_eq!(parse_zram_mm_stat(mm), Some((201_850_880, 55_889_920)));
        assert_eq!(parse_zram_mm_stat("bad"), None);
    }

    #[test]
    fn parses_bench_json_and_detects_regressions_and_log_errors() {
        let sample = r#"[
          {
            "model": "OLMoE.gguf",
            "mode": "r12_olmoe",
            "workload": "code_edit",
            "decode_tok_per_s": 15.1,
            "ttft_s": 1.02,
            "peak_rss_mib": 3300.0,
            "swap_delta_mib": 0.0,
            "joules_per_tok": 1.15,
            "using_gpu": true,
            "backend_name": "CUDA0"
          }
        ]"#;
        let recs = parse_bench_json_array(sample).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].model, "OLMoE.gguf");
        assert_eq!(recs[0].mode, "olmoe");
        assert_eq!(recs[0].decode_tok_per_s, Some(15.1));
        assert_eq!(recs[0].using_gpu, Some(true));

        // Truncated or malformed JSON is rejected instead of returning partial records.
        assert!(parse_bench_json_array(r#"[{"model":"a"}, {"model":"b""#).is_err());
        assert!(parse_bench_json_array(r#"{"model":"a"}"#).is_err());

        let key = ("OLMoE.gguf".into(), "olmoe".into(), "code_edit".into());
        let mut bad = recs[0].clone();
        bad.swap_delta_mib = Some(128.0);
        let (_row, _warns, fails) =
            check_pair(&key, Some(&recs[0]), &bad, &RegressionThresholds::default());
        assert_eq!(fails.len(), 1);
        assert!(fails[0].contains("swap delta"));

        // Common swap and CPU-fallback checks also apply to multi-turn records.
        let mt_key = (
            "Qwen14B.gguf".into(),
            "14b_auto".into(),
            "multi_turn_prefix_reuse".into(),
        );
        let mut mt_bad = recs[0].clone();
        mt_bad.workload = "multi_turn_prefix_reuse".into();
        mt_bad.swap_delta_mib = Some(64.0);
        mt_bad.using_gpu = Some(true);
        mt_bad.backend_name = Some("CPU".into());
        let (_mt_row, _mt_warns, mt_fails) =
            check_pair(&mt_key, None, &mt_bad, &RegressionThresholds::default());
        assert_eq!(mt_fails.len(), 2);

        assert_eq!(
            extract_graph_splits("sched_reserve: graph: nodes = 10, splits = 6"),
            vec![6]
        );
        assert!(has_nvme_timeout("kernel: nvme0: I/O tag 12 timeout"));
        assert!(!has_nvme_timeout("normal log line"));
    }
}
