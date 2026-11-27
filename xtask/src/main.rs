//! Pure-Rust workspace task runner (`cargo xtask <cmd>`).
//!
//! Replaces embedded bash/awk/Python recipes previously living in `Justfile`.
//! Uses only the Rust standard library so it compiles in < 1s with zero dependencies.

mod wheel_audit;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn {:?}: {e}", cmd.get_program()))?;
    if !status.success() {
        return Err(format!(
            "command {:?} exited with status {}",
            cmd.get_program(),
            status
        ));
    }
    Ok(())
}

fn run_in(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let mut cmd = Command::new(program);
    cmd.current_dir(root).args(args);
    run(&mut cmd)
}

fn nix_develop(root: &Path, shell: Option<&str>, script: &str) -> Result<(), String> {
    let mut cmd = Command::new("nix");
    cmd.current_dir(root).arg("develop");
    if let Some(s) = shell {
        cmd.arg(s);
    }
    cmd.args(["-c", "bash", "-euo", "pipefail", "-c", script]);
    run(&mut cmd)
}

fn extract_toml_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("version = \"")
            && let Some(end) = rest.find('"')
        {
            return Some(rest[..end].to_string());
        }
    }
    None
}

fn extract_json_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(idx) = line.find("\"version\"") {
            let after_key = &line[idx + "\"version\"".len()..];
            if let Some(colon) = after_key.find(':') {
                let after_colon = after_key[colon + 1..].trim_start();
                if let Some(rest) = after_colon.strip_prefix('"')
                    && let Some(end) = rest.find('"')
                {
                    return Some(rest[..end].to_string());
                }
            }
        }
    }
    None
}

fn read_manifest_version(root: &Path, rel: &str, is_json: bool) -> Result<String, String> {
    let text =
        fs::read_to_string(root.join(rel)).map_err(|e| format!("failed to read {rel}: {e}"))?;
    let ver = if is_json {
        extract_json_version(&text)
    } else {
        extract_toml_version(&text)
    };
    ver.ok_or_else(|| format!("version field not found in {rel}"))
}

fn extract_flake_llama_sha(flake_nix: &str) -> Option<String> {
    let marker = "github:ggml-org/llama.cpp/";
    let idx = flake_nix.find(marker)?;
    let rest = &flake_nix[idx + marker.len()..];
    let sha: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    (sha.len() == 40).then_some(sha)
}

fn extract_flake_lock_llama_locked_rev(flake_lock: &str) -> Option<String> {
    let node_idx = flake_lock.find("\"llama-cpp-src\"")?;
    let after_node = &flake_lock[node_idx..];
    let locked_idx = after_node.find("\"locked\"")?;
    let after_locked = &after_node[locked_idx..];
    let open_brace = after_locked.find('{')?;
    let close_brace = after_locked[open_brace..].find('}')?;
    let locked_block = &after_locked[open_brace..open_brace + close_brace];
    for line in locked_block.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("\"rev\":") {
            let val = rest.trim().trim_end_matches(',').trim().trim_matches('"');
            if val.len() == 40 && val.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(val.to_string());
            }
        }
    }
    None
}

fn normalize_yaml_scalar(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(s) = trimmed
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
    {
        return s;
    }
    if let Some(s) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return s;
    }
    trimmed
}

fn extract_workflow_llama_refs(workflow_yaml: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = workflow_yaml.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed == "with:" {
            let with_indent = line.len() - line.trim_start().len();
            let mut is_llama_repo = false;
            let mut found_ref: Option<(usize, String)> = None;
            let mut j = i + 1;
            while j < lines.len() {
                let child = lines[j];
                let child_trimmed = child.trim();
                if child_trimmed.is_empty() || child_trimmed.starts_with('#') {
                    j += 1;
                    continue;
                }
                let child_indent = child.len() - child.trim_start().len();
                if child_indent <= with_indent {
                    break;
                }
                if let Some(rest) = child_trimmed.strip_prefix("repository:") {
                    if normalize_yaml_scalar(rest) == "ggml-org/llama.cpp" {
                        is_llama_repo = true;
                    }
                } else if let Some(rest) = child_trimmed.strip_prefix("ref:") {
                    let val = normalize_yaml_scalar(rest);
                    found_ref = Some((j + 1, val.to_string()));
                }
                j += 1;
            }
            if is_llama_repo && let Some(entry) = found_ref {
                out.push(entry);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

fn check_llama_cpp_pins(root: &Path) -> Result<(), String> {
    let flake_nix = fs::read_to_string(root.join("flake.nix"))
        .map_err(|e| format!("failed to read flake.nix: {e}"))?;
    let expected = extract_flake_llama_sha(&flake_nix)
        .ok_or_else(|| "40-char ggml-org/llama.cpp SHA not found in flake.nix".to_string())?;

    let flake_lock = fs::read_to_string(root.join("flake.lock"))
        .map_err(|e| format!("failed to read flake.lock: {e}"))?;
    let locked_rev = extract_flake_lock_llama_locked_rev(&flake_lock)
        .ok_or_else(|| "llama-cpp-src locked.rev not found in flake.lock".to_string())?;
    if locked_rev != expected {
        return Err(format!(
            "llama.cpp pin mismatch: flake.nix={expected} vs flake.lock llama-cpp-src.locked.rev={locked_rev}"
        ));
    }

    let workflows_dir = root.join(".github/workflows");
    let mut wf_paths: Vec<PathBuf> = fs::read_dir(&workflows_dir)
        .map_err(|e| format!("failed to read .github/workflows: {e}"))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| ext == "yml" || ext == "yaml")
        })
        .collect();
    wf_paths.sort();

    let mut checked_refs = 0usize;
    for path in wf_paths {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        for (line_no, found) in extract_workflow_llama_refs(&text) {
            checked_refs += 1;
            if found != expected {
                return Err(format!(
                    "llama.cpp pin mismatch: flake.nix={expected} vs {rel}:{line_no}={found}"
                ));
            }
        }
    }
    if checked_refs == 0 {
        return Err("no ggml-org/llama.cpp checkout refs found in .github/workflows".to_string());
    }
    println!(
        "llama.cpp pin={expected} synced across flake.nix, flake.lock, and {checked_refs} workflow ref(s)"
    );
    Ok(())
}

fn cmd_check_versions(root: &Path) -> Result<(), String> {
    let cargo_ver = read_manifest_version(root, "Cargo.toml", false)?;
    let plugin_ver = read_manifest_version(root, "plugin.toml", false)?;
    let py_cargo_ver = read_manifest_version(root, "bindings/python/Cargo.toml", false)?;
    let py_proj_ver = read_manifest_version(root, "bindings/python/pyproject.toml", false)?;
    let py_cuda_ver = read_manifest_version(root, "bindings/python-cuda13/pyproject.toml", false)?;
    let node_cargo_ver = read_manifest_version(root, "bindings/node/Cargo.toml", false)?;
    let node_pkg_ver = read_manifest_version(root, "bindings/node/package.json", true)?;

    println!(
        "Cargo={cargo_ver} plugin={plugin_ver} py={py_proj_ver} py_cuda={py_cuda_ver} node={node_pkg_ver}"
    );
    let checks = [
        ("plugin.toml", &plugin_ver),
        ("bindings/python/Cargo.toml", &py_cargo_ver),
        ("bindings/python/pyproject.toml", &py_proj_ver),
        ("bindings/python-cuda13/pyproject.toml", &py_cuda_ver),
        ("bindings/node/Cargo.toml", &node_cargo_ver),
        ("bindings/node/package.json", &node_pkg_ver),
    ];
    for (name, ver) in checks {
        if *ver != cargo_ver {
            return Err(format!(
                "version mismatch: Cargo.toml={cargo_ver} vs {name}={ver}"
            ));
        }
    }
    check_llama_cpp_pins(root)?;
    Ok(())
}

fn cmd_check_loc(root: &Path) -> Result<(), String> {
    let out = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "*.rs", "*.cpp", "*.hpp", "*.py", "*.js", "*.ts"])
        .output()
        .map_err(|e| format!("git ls-files failed: {e}"))?;
    if !out.status.success() {
        return Err("git ls-files returned non-zero status".to_string());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut bad = Vec::new();
    for rel in stdout.lines().filter(|l| !l.is_empty()) {
        let path = root.join(rel);
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let lines = bytes.iter().filter(|&&b| b == b'\n').count();
        if lines > 2000 {
            bad.push((lines, rel.to_string()));
        }
    }
    if !bad.is_empty() {
        eprintln!("ERROR: source files exceeding 2,000 LoC limit:");
        for (n, f) in &bad {
            eprintln!("  {n:5} {f}");
        }
        return Err(format!("{} file(s) exceed 2,000 LoC limit", bad.len()));
    }
    println!("All tracked source files are <= 2,000 LoC.");
    Ok(())
}

fn is_valid_semver(ver: &str) -> bool {
    let parts: Vec<&str> = ver.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

fn replace_line_prefix(text: &str, prefix: &str, suffix: &str, new_val: &str) -> Option<String> {
    let mut changed = false;
    let mut out = String::with_capacity(text.len() + 16);
    for segment in text.split_inclusive('\n') {
        let (line, nl) = segment
            .strip_suffix('\n')
            .map_or((segment, ""), |l| (l, "\n"));
        if let Some(rest) = line.strip_prefix(prefix)
            && let Some(end) = rest.find(suffix)
        {
            out.push_str(prefix);
            out.push_str(new_val);
            out.push_str(&rest[end..]);
            out.push_str(nl);
            changed = true;
            continue;
        }
        out.push_str(segment);
    }
    changed.then_some(out)
}

fn replace_substring_delimited(text: &str, marker: &str, new_val: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    let mut changed = false;
    while let Some(idx) = rest.find(marker) {
        out.push_str(&rest[..idx + marker.len()]);
        let after = &rest[idx + marker.len()..];
        let old_len = after
            .bytes()
            .take_while(|b| b.is_ascii_digit() || *b == b'.')
            .count();
        if old_len > 0 {
            out.push_str(new_val);
            rest = &after[old_len..];
            changed = true;
        } else {
            rest = after;
        }
    }
    out.push_str(rest);
    changed.then_some(out)
}

fn cmd_bump(root: &Path, ver: &str) -> Result<(), String> {
    if !is_valid_semver(ver) {
        return Err(format!("invalid semver: {ver}"));
    }

    let toml_files = [
        "Cargo.toml",
        "plugin.toml",
        "bindings/python/Cargo.toml",
        "bindings/python/pyproject.toml",
        "bindings/python-cuda13/pyproject.toml",
        "bindings/node/Cargo.toml",
    ];
    for rel in toml_files {
        let p = root.join(rel);
        let text = fs::read_to_string(&p).map_err(|e| format!("read {rel}: {e}"))?;
        let updated = replace_line_prefix(&text, "version = \"", "\"", ver)
            .ok_or_else(|| format!("version line not found in {rel}"))?;
        fs::write(&p, updated).map_err(|e| format!("write {rel}: {e}"))?;
    }

    // Update pinned `layerstream-cuda13==<ver>` in bindings/python/pyproject.toml
    {
        let rel = "bindings/python/pyproject.toml";
        let p = root.join(rel);
        let text = fs::read_to_string(&p).map_err(|e| format!("read {rel}: {e}"))?;
        let updated = replace_substring_delimited(&text, "layerstream-cuda13==", ver)
            .ok_or_else(|| format!("layerstream-cuda13 pin not found in {rel}"))?;
        fs::write(&p, updated).map_err(|e| format!("write {rel}: {e}"))?;
    }

    // Update internal path-dependency version in crates/hf-gguf/Cargo.toml
    {
        let rel = "crates/hf-gguf/Cargo.toml";
        let p = root.join(rel);
        let text = fs::read_to_string(&p).map_err(|e| format!("read {rel}: {e}"))?;
        let updated = replace_substring_delimited(
            &text,
            "gguf-direct-io = { path = \"../gguf-direct-io\", version = \"",
            ver,
        )
        .ok_or_else(|| format!("gguf-direct-io path-dep version not found in {rel}"))?;
        fs::write(&p, updated).map_err(|e| format!("write {rel}: {e}"))?;
    }

    // Update `__version__ = "<ver>"` in python-cuda13
    {
        let rel = "bindings/python-cuda13/python/layerstream_cuda13/__init__.py";
        let p = root.join(rel);
        let text = fs::read_to_string(&p).map_err(|e| format!("read {rel}: {e}"))?;
        let updated = replace_line_prefix(&text, "__version__ = \"", "\"", ver)
            .ok_or_else(|| format!("__version__ not found in {rel}"))?;
        fs::write(&p, updated).map_err(|e| format!("write {rel}: {e}"))?;
    }

    // Update `"version": "<ver>"` in bindings/node/package.json
    {
        let rel = "bindings/node/package.json";
        let p = root.join(rel);
        let text = fs::read_to_string(&p).map_err(|e| format!("read {rel}: {e}"))?;
        let updated = replace_line_prefix(&text, "  \"version\": \"", "\"", ver)
            .ok_or_else(|| format!("version field not found in {rel}"))?;
        fs::write(&p, updated).map_err(|e| format!("write {rel}: {e}"))?;
    }

    for subdir in [".", "bindings/python", "bindings/node"] {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(root.join(subdir))
            .args(["metadata", "--format-version", "1", "--offline"])
            .stdout(Stdio::null());
        run(&mut cmd)?;
    }

    cmd_check_versions(root)
}

fn cmd_preflight(root: &Path) -> Result<(), String> {
    cmd_check_versions(root)?;
    cmd_check_loc(root)?;
    run_in(root, "cargo", &["fmt", "--all", "--", "--check"])?;
    run_in(
        &root.join("bindings/python"),
        "cargo",
        &["fmt", "--", "--check"],
    )?;
    run_in(
        &root.join("bindings/node"),
        "cargo",
        &["fmt", "--", "--check"],
    )?;
    nix_develop(
        root,
        Some(".#python"),
        "cd bindings/python && ruff check python tests ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py ../../benchmarks && ruff format --check python tests ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py ../../benchmarks && pyright python/layerstream tests",
    )?;
    nix_develop(root, Some(".#node"), "cd bindings/node && biome check .")?;
    run_in(root, "nix", &["fmt", "--", "--ci"])?;
    if Command::new("nix-check-workarounds")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        run_in(root, "nix-check-workarounds", &["."])?;
    }
    Ok(())
}

fn cmd_fmt(root: &Path) -> Result<(), String> {
    run_in(root, "cargo", &["fmt", "--all"])?;
    run_in(&root.join("bindings/python"), "cargo", &["fmt"])?;
    run_in(&root.join("bindings/node"), "cargo", &["fmt"])?;
    nix_develop(
        root,
        Some(".#python"),
        "cd bindings/python && ruff format python tests ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py ../../benchmarks",
    )?;
    nix_develop(
        root,
        Some(".#node"),
        "cd bindings/node && biome check --write .",
    )?;
    run_in(root, "nix", &["fmt"])
}

fn cmd_test_rust(root: &Path) -> Result<(), String> {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let target_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/layerstream-target"));
    let mut clippy = Command::new("chrt");
    clippy
        .current_dir(root)
        .env("CARGO_TARGET_DIR", &target_dir)
        .args([
            "--idle",
            "0",
            "ionice",
            "-c3",
            "cargo",
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "-j",
            "4",
            "--",
            "-D",
            "warnings",
        ]);
    run(&mut clippy)?;
    let mut test = Command::new("chrt");
    test.current_dir(root)
        .env("CARGO_TARGET_DIR", &target_dir)
        .args([
            "--idle",
            "0",
            "ionice",
            "-c3",
            "cargo",
            "test",
            "--locked",
            "--workspace",
            "-j",
            "4",
            "--",
            "--test-threads",
            "1",
        ]);
    run(&mut test)
}

fn cmd_lint_cpp(root: &Path) -> Result<(), String> {
    nix_develop(
        root,
        None,
        r#"clang-tidy crates/layerstream-core/cpp/backend_loader.cpp crates/layerstream-core/cpp/stream_engine.cpp -- -std=c++23 -Icrates/layerstream-core/cpp -isystem"$LLAMA_CPP_DIR/include" -isystem"$LLAMA_CPP_DIR/src" -isystem"$LLAMA_CPP_DIR/ggml/include" -isystem"$LLAMA_CPP_DIR/ggml/src""#,
    )
}

fn cmd_coverage_cpp(root: &Path) -> Result<(), String> {
    nix_develop(
        root,
        None,
        r#"export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-cpp-cov}"
export LAYERSTREAM_CPP_COVERAGE=1
find "$CARGO_TARGET_DIR" -name "*.gcda" -delete 2>/dev/null || true
chrt --idle 0 ionice -c3 cargo test --locked --target x86_64-unknown-linux-gnu -p layerstream-core -j 4 -- --test-threads 1
python3 - <<'PY'
import os, subprocess, pathlib, json
root = pathlib.Path(os.environ["CARGO_TARGET_DIR"])
gcdas = sorted(root.rglob("*-backend_loader.gcda")) + sorted(root.rglob("*-stream_engine.gcda"))
by_file = {}
for g in gcdas:
    data = json.loads(subprocess.check_output(["gcov", "--json-format", "--stdout", g.name], cwd=g.parent, text=True))
    for f in data["files"]:
        if not f["file"].startswith("cpp/"):
            continue
        rec = by_file.setdefault(f["file"], {})
        for l in f["lines"]:
            n = l["line_number"]
            rec[n] = max(rec.get(n, 0), l["count"])
th = ta = 0
for k, m in sorted(by_file.items()):
    h = sum(c > 0 for c in m.values())
    a = len(m)
    th += h
    ta += a
    print(f"{k:32s} {h:4d}/{a:4d} ({100*h/a:5.2f}%)")
print(f"TOTAL                            {th:4d}/{ta:4d} ({100*th/ta:5.2f}%)")
PY"#,
    )
}

fn cmd_gpu_smoke(root: &Path, backend: &str) -> Result<(), String> {
    match backend {
        "cuda" => nix_develop(
            root,
            Some(".#cuda"),
            r#"CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-gpu-smoke}" chrt --idle 0 ionice -c3 cargo test --locked -p layerstream-core --features cuda --test parity_test -j 4 -- --test-threads 1"#,
        ),
        "vulkan" => nix_develop(
            root,
            None,
            r#"CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-gpu-smoke}" chrt --idle 0 ionice -c3 cargo test --locked -p layerstream-core --features vulkan --test parity_test -j 4 -- --test-threads 1"#,
        ),
        other => Err(format!(
            "unsupported backend '{other}' (use 'vulkan' or 'cuda')"
        )),
    }
}

fn print_usage() {
    println!(
        "Usage: cargo xtask <command> [args]\n\
         \n\
         Commands:\n\
           check-versions      Verify all 7 workspace, plugin, Python, and Node versions match\n\
           check-loc           Verify no tracked source file exceeds 2,000 LoC\n\
           preflight           Run formatting, lint, type, version, and LoC checks\n\
           fmt                 Format Rust, Python, Node/TS, and Nix files\n\
           test-rust           Run Rust workspace clippy + tests at idle priority\n\
           coverage            Measure LLVM coverage for pure-Rust workspace crates\n\
           test-python-fast    Run fast pure-Python unit tests\n\
           coverage-python     Measure Python SDK line coverage (100% floor)\n\
           coverage-node       Measure Node.js SDK V8 line coverage (100% floor)\n\
           lint-cpp            Run clang-tidy static analysis on C++ shim\n\
           coverage-cpp        Measure C++ gcov line coverage for crates/layerstream-core/cpp/\n\
           gpu-smoke [backend] Run real GPU parity smoke test (vulkan | cuda)\n\
           audit-wheel-elf     Verify Linux wheel ELF symbols, RUNPATH, and PyPI 100 MiB cap\n\
           bump <version>      Bump semver across all manifests and refresh Cargo.lock files"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let root = repo_root();
    let res = match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => {
            print_usage();
            Ok(())
        }
        Some("check-versions") => cmd_check_versions(&root),
        Some("check-loc") => cmd_check_loc(&root),
        Some("preflight") => cmd_preflight(&root),
        Some("fmt") => cmd_fmt(&root),
        Some("test-rust") => cmd_test_rust(&root),
        Some("coverage") => run_in(
            &root,
            "cargo",
            &[
                "llvm-cov",
                "--locked",
                "-p",
                "json-schema-gbnf",
                "-p",
                "gguf-direct-io",
                "-p",
                "hf-gguf",
                "--html",
                "--",
                "--test-threads",
                "1",
            ],
        ),
        Some("test-python-fast") => nix_develop(
            &root,
            Some(".#python"),
            "cd bindings/python && pytest -v tests/test_cuda13_plugin.py",
        ),
        Some("coverage-python") => nix_develop(
            &root,
            Some(".#python"),
            "cd bindings/python && PYTHONPATH=python pytest --cov=layerstream --cov-report=term-missing --cov-fail-under=100",
        ),
        Some("coverage-node") => nix_develop(
            &root,
            Some(".#node"),
            "cd bindings/node && npm run test:coverage",
        ),
        Some("lint-cpp") => cmd_lint_cpp(&root),
        Some("coverage-cpp") => cmd_coverage_cpp(&root),
        Some("gpu-smoke") => {
            let backend = args.get(1).map_or("vulkan", String::as_str);
            cmd_gpu_smoke(&root, backend)
        }
        Some("audit-wheel-elf") => wheel_audit::cmd_audit_wheel_elf(&args[1..]),
        Some("bump") => match args.get(1) {
            Some(ver) => cmd_bump(&root, ver),
            None => Err("usage: cargo xtask bump <major.minor.patch>".to_string()),
        },
        Some(other) => Err(format!("unknown xtask command '{other}' (run --help)")),
    };

    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_semver_strings() {
        assert!(is_valid_semver("0.5.0"));
        assert!(is_valid_semver("12.34.56"));
        assert!(!is_valid_semver("0.5"));
        assert!(!is_valid_semver("0.5.0-rc1"));
        assert!(!is_valid_semver("v0.5.0"));
    }

    #[test]
    fn extracts_and_replaces_versions() {
        let toml = "[workspace.package]\nversion = \"0.4.1\"\nedition = \"2024\"\n";
        assert_eq!(extract_toml_version(toml).as_deref(), Some("0.4.1"));
        let bumped = replace_line_prefix(toml, "version = \"", "\"", "0.5.0");
        assert_eq!(
            bumped.as_deref(),
            Some("[workspace.package]\nversion = \"0.5.0\"\nedition = \"2024\"\n")
        );

        let json = "{\n  \"name\": \"layerstream\",\n  \"version\": \"0.4.1\"\n}\n";
        assert_eq!(extract_json_version(json).as_deref(), Some("0.4.1"));

        let pyproj = "cuda13 = [\"layerstream-cuda13==0.4.1\"]\n";
        let bumped_py = replace_substring_delimited(pyproj, "layerstream-cuda13==", "0.5.0");
        assert_eq!(
            bumped_py.as_deref(),
            Some("cuda13 = [\"layerstream-cuda13==0.5.0\"]\n")
        );
    }

    #[test]
    fn check_versions_and_loc_succeed_on_repo() {
        let root = repo_root();
        assert!(cmd_check_versions(&root).is_ok());
        assert!(cmd_check_loc(&root).is_ok());
    }

    #[test]
    fn extracts_llama_cpp_sha_and_workflow_refs() {
        let flake =
            "url = \"github:ggml-org/llama.cpp/0123456789abcdef0123456789abcdef01234567\";\n";
        assert_eq!(
            extract_flake_llama_sha(flake).as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(
            extract_flake_llama_sha("github:ggml-org/llama.cpp/short"),
            None
        );

        let wf = r#"
        with:
          repository: ggml-org/llama.cpp
          ref: 0123456789abcdef0123456789abcdef01234567
        with:
          repository: 'ggml-org/llama.cpp'
          ref: 'fedcba9876543210fedcba9876543210fedcba98'
        with:
          ref: "1111111111111111111111111111111111111111"
          path: third_party/llama.cpp
          fetch-depth: 1
          submodules: false
          repository: "ggml-org/llama.cpp"
"#;
        let refs = extract_workflow_llama_refs(wf);
        assert_eq!(
            refs,
            vec![
                (4, "0123456789abcdef0123456789abcdef01234567".to_string()),
                (7, "fedcba9876543210fedcba9876543210fedcba98".to_string()),
                (9, "1111111111111111111111111111111111111111".to_string()),
            ]
        );

        // locked.rev is extracted specifically, even when original.rev differs.
        let lock = r#"{
          "nodes": {
            "llama-cpp-src": {
              "locked": {
                "rev": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              },
              "original": {
                "rev": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
              }
            }
          }
        }"#;
        assert_eq!(
            extract_flake_lock_llama_locked_rev(lock).as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }
}
