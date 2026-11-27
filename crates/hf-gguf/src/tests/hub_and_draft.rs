use super::ENV_LOCK;
use crate::*;

#[test]
fn parses_repo_and_revision_forms() {
    assert_eq!(
        parse_repo_and_revision("Qwen/Qwen2.5-14B-Instruct-GGUF", None).unwrap(),
        ("Qwen/Qwen2.5-14B-Instruct-GGUF".into(), "main".into())
    );
    assert_eq!(
        parse_repo_and_revision("hf://bartowski/Llama-3-GGUF@v2", None).unwrap(),
        ("bartowski/Llama-3-GGUF".into(), "v2".into())
    );
    assert_eq!(
        parse_repo_and_revision("https://huggingface.co/unsloth/Qwen3-GGUF/", Some("rev1"))
            .unwrap(),
        ("unsloth/Qwen3-GGUF".into(), "rev1".into())
    );
    assert!(parse_repo_and_revision("invalid-no-slash", None).is_err());
}

#[test]
fn groups_shards_skips_mmproj_and_ranks_quants() {
    let gib = 1024u64 * 1024 * 1024;
    let entries = vec![
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "mmproj-model-f16.gguf".into(),
            size: gib,
            lfs: None,
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "model-q4_k_m-00001-of-00003.gguf".into(),
            size: 0,
            lfs: Some(HfLfsInfo { size: 3 * gib }),
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "model-q4_k_m-00002-of-00003.gguf".into(),
            size: 0,
            lfs: Some(HfLfsInfo { size: 3 * gib }),
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "model-q4_k_m-00003-of-00003.gguf".into(),
            size: 0,
            lfs: Some(HfLfsInfo { size: 2 * gib }),
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "model-q8_0.gguf".into(),
            size: 16 * gib,
            lfs: None,
        },
    ];

    let cands = group_gguf_tree_entries(&entries);
    assert_eq!(cands.len(), 2, "mmproj skipped and 3 shards grouped");
    assert_eq!(cands[0].quant, "Q4_K_M");
    assert_eq!(cands[0].shard_count, 3);
    assert_eq!(cands[0].total_bytes, 8 * gib);
    assert_eq!(cands[0].primary_file, "model-q4_k_m-00001-of-00003.gguf");

    // On an 8 GB Jetson (5.2 GiB free, 2800 MiB/s), 8 GiB Q4_K_M should beat 16 GiB Q8_0.
    let recs = recommend_candidates(
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "main",
        &cands,
        (5.2 * gib as f64) as u64,
        2800.0,
        true,
    );
    let best = recs.iter().find(|r| r.recommended).unwrap();
    assert_eq!(best.candidate.quant, "Q4_K_M");
    assert_eq!(best.verdict, "GOOD");
    assert_eq!(
        best.hf_uri,
        "hf://Qwen/Qwen2.5-14B-Instruct-GGUF:model-q4_k_m-00001-of-00003.gguf"
    );
}

#[test]
fn xet_redirect_uses_hub_x_linked_etag_not_cdn_xet_etag() {
    let _env_guard = ENV_LOCK.lock().unwrap();
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    let payload: Vec<u8> = (0..16_384).map(|i| ((i * 29 + 11) % 251) as u8).collect();
    let real_sha256 = to_hex_lower(&Sha256::digest(&payload));
    let fake_xet_cas_etag = "0663b826e09042345d6e2069d9655cd3ae84812d8f39a5894bed7b24fc0e86cc";
    assert_ne!(real_sha256, fake_xet_cas_etag);

    // Simulate HF Hub 302 with `X-Linked-Etag` (real SHA-256) redirecting to a CDN 200
    // whose `ETag` is a different 64-hex Xet CAS hash (and verify Authorization is stripped).
    let cdn_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let cdn_addr = cdn_listener.local_addr().unwrap();
    let cdn_payload = payload.clone();
    let cdn_thread = std::thread::spawn(move || {
        let (mut sock, _) = cdn_listener.accept().unwrap();
        let mut reader = BufReader::new(sock.try_clone().unwrap());
        let mut saw_auth = false;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            if line.to_ascii_lowercase().starts_with("authorization:") {
                saw_auth = true;
            }
        }
        assert!(
            !saw_auth,
            "Authorization header must not leak to redirected CDN"
        );
        let header = format!(
            "HTTP/1.1 200 OK\r\nETag: \"{fake_xet_cas_etag}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            cdn_payload.len()
        );
        sock.write_all(header.as_bytes()).unwrap();
        sock.write_all(&cdn_payload).unwrap();
        let _ = sock.flush();
    });

    let hub_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let hub_addr = hub_listener.local_addr().unwrap();
    let hub_sha = real_sha256.clone();
    let hub_thread = std::thread::spawn(move || {
        let (mut sock, _) = hub_listener.accept().unwrap();
        let mut reader = BufReader::new(sock.try_clone().unwrap());
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                break;
            }
        }
        let header = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://{cdn_addr}/xet-bridge/blob\r\nX-Linked-Etag: \"{hub_sha}\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        sock.write_all(header.as_bytes()).unwrap();
        let _ = sock.flush();
    });

    let tmp = std::env::temp_dir().join(format!("layerstream-xet-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{hub_addr}"));
        std::env::set_var("HF_TOKEN", "hf_test_secret_token");
    }

    let res = ensure_hf_downloaded("hf://testorg/xetrepo:model.gguf", &tmp);

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("HF_TOKEN");
    }
    hub_thread.join().unwrap();
    cdn_thread.join().unwrap();

    let local = res.expect("Xet redirect download must verify against Hub X-Linked-Etag");
    assert_eq!(fs::read(&local).unwrap(), payload);
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn resolve_pretrained_local_file_and_missing_local_fail_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf = tmp.path().join("tiny.gguf");
    fs::write(&gguf, b"GGUF").unwrap();

    let opts = PretrainedResolveOptions {
        cache_dir: Some(tmp.path().join("cache")),
        ..Default::default()
    };
    let resolved = resolve_pretrained_model(gguf.to_str().unwrap(), &opts).unwrap();
    assert_eq!(resolved.local_path, gguf);
    assert!(resolved.source_uri.is_none());

    let missing = tmp.path().join("does-not-exist.gguf");
    let err = resolve_pretrained_model(missing.to_str().unwrap(), &opts).unwrap_err();
    assert!(
        err.to_string().contains("local GGUF model not found"),
        "unexpected error: {err}"
    );
}

#[test]
fn resolve_pretrained_gguf_repo_and_base_repo_sibling() {
    let _env_guard = ENV_LOCK.lock().unwrap();
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    let payload = b"GGUF_MOCK_PAYLOAD_BYTES".to_vec();
    let sha = to_hex_lower(&Sha256::digest(&payload));
    let tree_gguf = format!(
        r#"[
              {{"type":"file","path":"Model-Q4_K_M.gguf","size":{},"lfs":{{"size":{}}}}},
              {{"type":"file","path":"Model-Q8_0.gguf","size":{},"lfs":{{"size":{}}}}}
            ]"#,
        4 * 1024 * 1024 * 1024u64,
        4 * 1024 * 1024 * 1024u64,
        9 * 1024 * 1024 * 1024u64,
        9 * 1024 * 1024 * 1024u64,
    );
    let tree_safetensors = r#"[{"type":"file","path":"model.safetensors","size":1000}]"#;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_payload = payload.clone();
    let srv_sha = sha.clone();
    let server = std::thread::spawn(move || {
        // Handle 5 HTTP requests across the test cases.
        for stream in listener.incoming().take(5) {
            let Ok(mut sock) = stream else { break };
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            let mut req_line = String::new();
            if reader.read_line(&mut req_line).unwrap_or(0) == 0 {
                continue;
            }
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let path = req_line.split_whitespace().nth(1).unwrap_or("");
            if path == "/api/models/Org/BaseModel/tree/main?recursive=true" {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    tree_safetensors.len(),
                    tree_safetensors
                );
                sock.write_all(resp.as_bytes()).unwrap();
            } else if path == "/api/models/Org/BaseModel-GGUF/tree/main?recursive=true" {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    tree_gguf.len(),
                    tree_gguf
                );
                sock.write_all(resp.as_bytes()).unwrap();
            } else if path == "/Org/BaseModel-GGUF/resolve/main/Model-Q4_K_M.gguf" {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nETag: \"{srv_sha}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    srv_payload.len()
                );
                sock.write_all(header.as_bytes()).unwrap();
                sock.write_all(&srv_payload).unwrap();
            } else if path == "/api/models/Org/NoGguf/tree/main?recursive=true" {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    tree_safetensors.len(),
                    tree_safetensors
                );
                sock.write_all(resp.as_bytes()).unwrap();
            } else {
                sock.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
            }
            let _ = sock.flush();
        }
    });

    let tmp = tempfile::tempdir().unwrap();
    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };

    let opts = PretrainedResolveOptions {
        cache_dir: Some(tmp.path().to_path_buf()),
        vram_budget_bytes: Some(6 * 1024 * 1024 * 1024),
        use_gpu: false,
        has_draft: false,
        ..Default::default()
    };

    // 1. Base repo `Org/BaseModel` -> auto-resolves to `Org/BaseModel-GGUF` and picks Q4_K_M (3 requests).
    let res = resolve_pretrained_model("Org/BaseModel", &opts).unwrap();
    assert_eq!(res.resolved_repo.as_deref(), Some("Org/BaseModel-GGUF"));
    assert_eq!(res.selected_quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(fs::read(&res.local_path).unwrap(), payload);

    // 2. Base repo with no GGUFs and 404 sibling -> clear guidance error (2 requests).
    let err = resolve_pretrained_model("Org/NoGguf", &opts).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Org/NoGguf-GGUF"), "unexpected error: {msg}");
    assert!(
        msg.contains("bartowski/NoGguf-GGUF"),
        "missing guidance: {msg}"
    );

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

#[test]
fn auto_draft_family_mapping_and_progress_callbacks() {
    assert_eq!(
        select_auto_draft_repo("qwen2", "Qwen/Qwen2.5-14B-Instruct-GGUF"),
        Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("qwen3moe", "unsloth/Qwen3-30B-A3B-GGUF"),
        Some("Qwen/Qwen3-0.6B-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("llama", "bartowski/Meta-Llama-3.3-70B-Instruct-GGUF"),
        Some("bartowski/Llama-3.2-1B-Instruct-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("gemma3", "ggml-org/gemma-3-27b-it-GGUF"),
        Some("ggml-org/gemma-3-1b-it-GGUF")
    );
    assert_eq!(select_auto_draft_repo("unknown_arch", "custom/model"), None);

    // Verify progress callback receives start + completion + cache-hit events and propagates abort.
    let _env_guard = ENV_LOCK.lock().unwrap();
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Mutex;

    let payload = vec![0x42u8; 32 * 1024];
    let sha = to_hex_lower(&Sha256::digest(&payload));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_payload = payload.clone();
    let srv_sha = sha.clone();
    let server = std::thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let Ok(mut sock) = stream else { break };
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nETag: \"{srv_sha}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                srv_payload.len()
            );
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(&srv_payload);
            let _ = sock.flush();
        }
    });

    let tmp = tempfile::tempdir().unwrap();
    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };

    let events: Arc<Mutex<Vec<DownloadProgress>>> = Arc::new(Mutex::new(Vec::new()));
    let ev_clone = events.clone();
    let cb: ProgressCallback = Arc::new(move |ev| {
        ev_clone.lock().unwrap().push(ev.clone());
        Ok(())
    });

    let p1 = ensure_hf_downloaded_with_progress("hf://Org/Repo:test.gguf", tmp.path(), Some(&cb))
        .unwrap();
    assert!(p1.is_file());
    {
        let recorded = events.lock().unwrap();
        assert!(recorded.len() >= 2);
        assert!(!recorded.first().unwrap().done);
        let last = recorded.last().unwrap();
        assert!(last.done && !last.cached);
        assert_eq!(last.downloaded_bytes, payload.len() as u64);
    }

    // Second call hits cache and emits `cached: true, done: true`.
    events.lock().unwrap().clear();
    let _ = ensure_hf_downloaded_with_progress("hf://Org/Repo:test.gguf", tmp.path(), Some(&cb))
        .unwrap();
    {
        let recorded = events.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].cached && recorded[0].done);
    }

    // Callback returning Err aborts uncached download immediately.
    let abort_cb: ProgressCallback =
        Arc::new(|_ev| Err(HfError::Cancelled("user aborted download".into())));
    let err =
        ensure_hf_downloaded_with_progress("hf://Org/Repo:other.gguf", tmp.path(), Some(&abort_cb))
            .unwrap_err();
    assert!(err.to_string().contains("user aborted download"));

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

#[test]
fn covers_quant_labels_verdicts_local_paths_redirects_headers_and_progress_bar() {
    use crate::hub::{
        extract_quant_label, looks_like_local_path, quant_quality_rank, select_candidate_for_repo,
    };
    // Quant extraction & ranking across all tiers (including UD- bonus and unknown fallback).
    for (fname, expected) in [
        ("Model-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("Model-Q5_K_M.gguf", "Q5_K_M"),
        ("Model-IQ4_XS.gguf", "IQ4_XS"),
        ("Model-Q6_K.gguf", "Q6_K"),
        ("Model-Q8_0.gguf", "Q8_0"),
        ("Model-Q5_0.gguf", "Q5_0"),
        ("Model-Q3_K_M.gguf", "Q3_K_M"),
        ("Model-IQ3_M.gguf", "IQ3_M"),
        ("Model-IQ3_XS.gguf", "IQ3_XS"),
        ("Model-Q2_K_XL.gguf", "Q2_K_XL"),
        ("Model-Q2_K.gguf", "Q2_K"),
        ("Model-BF16.gguf", "BF16"),
        ("Model-F16.gguf", "F16"),
        ("weird-custom.gguf", "CUSTOM"),
    ] {
        let q = extract_quant_label(fname);
        assert_eq!(q, expected, "fname={fname}");
        assert!(quant_quality_rank(&q) >= 50);
    }

    // `recommend_candidates` verdicts: TOO_SLOW_DISK, GOOD, SLOW, VERY_SLOW, zero-size, has_draft.
    let big = HfGgufCandidate {
        quant: "Q4_K_M".to_string(),
        primary_file: "dir/Big-Q4_K_M.gguf".to_string(),
        shard_count: 1,
        total_bytes: 40 * 1024 * 1024 * 1024,
    };
    let slow_disk = recommend_candidates(
        "o/r",
        "main",
        std::slice::from_ref(&big),
        8 << 30,
        50.0,
        false,
    );
    assert_eq!(slow_disk[0].verdict, "TOO_SLOW_DISK");

    let good_stream = recommend_candidates(
        "o/r",
        "dev",
        std::slice::from_ref(&big),
        8 << 30,
        50_000.0,
        true,
    );
    assert_eq!(good_stream[0].verdict, "GOOD");
    assert!(good_stream[0].hf_uri.ends_with("@dev"));

    let slow_stream = recommend_candidates(
        "o/r",
        "main",
        std::slice::from_ref(&big),
        8 << 30,
        12_000.0,
        true,
    );
    assert!(matches!(
        slow_stream[0].verdict.as_str(),
        "GOOD" | "SLOW" | "VERY_SLOW"
    ));

    let very_slow = recommend_candidates(
        "o/r",
        "main",
        std::slice::from_ref(&big),
        4 << 30,
        200.0,
        false,
    );
    assert_eq!(very_slow[0].verdict, "VERY_SLOW");

    let zero_cand = HfGgufCandidate {
        quant: "Q4_K_M".to_string(),
        primary_file: "Zero.gguf".to_string(),
        shard_count: 1,
        total_bytes: 0,
    };
    let zero_recs = recommend_candidates("o/r", "main", &[zero_cand], 8 << 30, 2000.0, false);
    assert_eq!(zero_recs[0].pinned_pct, 0);

    // `select_candidate_for_repo`: exact path, basename, quant, and missing quant error.
    assert!(select_candidate_for_repo(&slow_disk, Some("dir/Big-Q4_K_M.gguf"), "o/r").is_ok());
    assert!(select_candidate_for_repo(&slow_disk, Some("Big-Q4_K_M.gguf"), "o/r").is_ok());
    assert!(select_candidate_for_repo(&slow_disk, Some("q4_k_m"), "o/r").is_ok());
    assert!(matches!(
        select_candidate_for_repo(&slow_disk, Some("Q8_0"), "o/r").unwrap_err(),
        HfError::NotFound(_)
    ));
    assert!(matches!(
        select_candidate_for_repo(&[], None, "o/r").unwrap_err(),
        HfError::NotFound(_)
    ));

    // `looks_like_local_path` across Unix, Windows, relative, and URL forms.
    assert!(looks_like_local_path("/abs/model.gguf"));
    assert!(looks_like_local_path("./rel.gguf"));
    assert!(looks_like_local_path("../up.gguf"));
    assert!(looks_like_local_path("~/m.gguf"));
    assert!(looks_like_local_path("C:\\models\\m.gguf"));
    assert!(looks_like_local_path("foo.gguf"));
    assert!(!looks_like_local_path("hf://org/repo:m.gguf"));
    assert!(!looks_like_local_path("https://huggingface.co/org/repo"));
    assert!(!looks_like_local_path("org/repo"));

    // `resolve_redirect_url`, `extract_header_sha256`, `content_range_matches_start`.
    assert_eq!(
        resolve_redirect_url("https://hf.co/a/b", "https://cdn.example/x").unwrap(),
        "https://cdn.example/x"
    );
    assert_eq!(
        resolve_redirect_url("https://hf.co/a/b", "/rel/path").unwrap(),
        "https://hf.co/rel/path"
    );
    assert_eq!(
        resolve_redirect_url("https://hf.co/a/b", "leaf").unwrap(),
        "https://hf.co/leaf"
    );
    assert!(resolve_redirect_url("not-a-uri", "/x").is_err());

    let mut hdrs = ureq::http::HeaderMap::new();
    let sha = "a".repeat(64);
    hdrs.insert("etag", format!("W/\"{sha}\"").parse().unwrap());
    assert_eq!(
        extract_header_sha256(&hdrs, "etag").as_deref(),
        Some(sha.as_str())
    );
    hdrs.insert("x-bad", "short".parse().unwrap());
    assert_eq!(extract_header_sha256(&hdrs, "x-bad"), None);

    hdrs.insert("content-range", "bytes 100-199/200".parse().unwrap());
    assert!(content_range_matches_start(&hdrs, 100));
    assert!(!content_range_matches_start(&hdrs, 50));

    // `render_tty_progress_bar` with total_bytes, without total_bytes, and multi-shard.
    let _ = render_tty_progress_bar(&DownloadProgress {
        repo_id: "o/r".into(),
        filename: "sub/m.gguf".into(),
        shard_index: 1,
        shard_count: 2,
        downloaded_bytes: 512,
        total_bytes: Some(1024),
        mib_per_s: 12.5,
        cached: false,
        done: false,
    });
    let _ = render_tty_progress_bar(&DownloadProgress {
        repo_id: "o/r".into(),
        filename: "m.gguf".into(),
        shard_index: 1,
        shard_count: 1,
        downloaded_bytes: 512,
        total_bytes: None,
        mib_per_s: 12.5,
        cached: false,
        done: false,
    });

    // `Debug` for `PretrainedResolveOptions`, `HfError::from(io::Error)`, `default_cache_dir`,
    // `is_path_or_hf_cached`, and `resolve_path_or_hf`.
    let dbg = format!("{:?}", PretrainedResolveOptions::default());
    assert!(dbg.contains("PretrainedResolveOptions"));
    let io_err: HfError = std::io::Error::other("disk").into();
    assert!(matches!(io_err, HfError::Io { .. }));
    assert!(!default_cache_dir().as_os_str().is_empty());
    let tmp = tempfile::tempdir().unwrap();
    let local = tmp.path().join("x.gguf");
    fs::write(&local, b"GGUF").unwrap();
    assert!(is_path_or_hf_cached(local.to_str().unwrap(), tmp.path()));
    assert_eq!(
        resolve_path_or_hf(local.to_str().unwrap(), tmp.path()).unwrap(),
        local
    );
}

#[test]
fn covers_resolve_url_form_auto_draft_fallbacks_and_link_edge_cases() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    // `hf://org/repo/resolve/<rev>/file.gguf` form extracts revision automatically.
    let parsed = HfSpec::parse("hf://Org/Repo/resolve/v2/sub/model.gguf").unwrap();
    assert_eq!(parsed.repo_id, "Org/Repo");
    assert_eq!(parsed.revision, "v2");
    assert_eq!(parsed.filename, "sub/model.gguf");

    // `extract_expected_sha256` never trusts a redirected CDN's ETag without hub_sha256.
    let mut cdn_hdrs = ureq::http::HeaderMap::new();
    cdn_hdrs.insert("etag", format!("\"{}\"", "b".repeat(64)).parse().unwrap());
    assert_eq!(extract_expected_sha256(&cdn_hdrs, true, None), None);
    assert!(extract_expected_sha256(&cdn_hdrs, false, None).is_some());

    // Mock server testing:
    // 1. `resolve_auto_draft_model_with_probe` fallback when `Q4_K_M` is absent (retries `quant: None`)
    //    AND draft has incompatible vocabulary -> returns `Ok(None)` cleanly!
    // 2. `resolve_auto_draft_model_with_probe` when draft repo returns 404 -> returns `Ok(None)` cleanly!
    // 3. `resolve_pretrained_model_with_probe` with `use_gpu: false` and `vram_budget_bytes: None`.
    let _env_guard = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();

    let target_p = tmp.path().join("big-qwen2.gguf");
    make_sparse_gguf(&target_p, "qwen2", &["a", "b", "c"], 3 * 1024 * 1024 * 1024);
    let target = ResolvedModel {
        local_path: target_p,
        source_uri: None,
        resolved_repo: None,
        selected_quant: None,
    };

    let bad_draft_src = tmp.path().join("bad_draft_src.gguf");
    make_sparse_gguf(&bad_draft_src, "qwen2", &["X", "Y", "Z"], 1024);
    let bad_draft_bytes = fs::read(&bad_draft_src).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bad_srv = bad_draft_bytes.clone();
    let server = std::thread::spawn(move || {
        for stream_res in listener.incoming() {
            let Ok(mut stream) = stream_res else { break };
            let mut req_buf = [0u8; 4096];
            let n = stream.read(&mut req_buf).unwrap_or(0);
            if n == 0 {
                continue;
            }
            let req = String::from_utf8_lossy(&req_buf[..n]);
            let path = req
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("");
            if path.ends_with("/__stop") {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            } else if path.contains("/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/") {
                // Only Q8_0 available (no Q4_K_M -> triggers quant=None retry!).
                let body = r#"[{"type":"file","path":"Qwen2.5-0.5B-Q8_0.gguf","size":500000000}]"#;
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nLink: <http://{addr}/prev>; rel=\"prev\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(body.as_bytes());
            } else if path.ends_with("Qwen2.5-0.5B-Q8_0.gguf") {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bad_srv.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&bad_srv);
            } else {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            }
        }
    });

    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
    }
    let cache = tmp.path().join("cache");

    // 1. Q4_K_M missing -> retries quant=None -> downloads Q8_0 -> vocab incompatible -> Ok(None).
    assert!(
        resolve_auto_draft_model_with_probe(
            &target,
            Some(cache.clone()),
            Some(2 * 1024 * 1024 * 1024),
            false,
            None,
            || None,
        )
        .unwrap()
        .is_none()
    );

    // 2. Draft repo 404 -> falls back cleanly to Ok(None).
    let gemma_p = tmp.path().join("big-gemma3.gguf");
    make_sparse_gguf(&gemma_p, "gemma3", &["a", "b"], 3 * 1024 * 1024 * 1024);
    let gemma_target = ResolvedModel {
        local_path: gemma_p,
        source_uri: None,
        resolved_repo: None,
        selected_quant: None,
    };
    assert!(
        resolve_auto_draft_model_with_probe(
            &gemma_target,
            Some(cache.clone()),
            None,
            false,
            None,
            || None,
        )
        .unwrap()
        .is_none()
    );

    // 3. `resolve_pretrained_model_with_probe` with `use_gpu: false` & `vram_budget_bytes: None`.
    let cpu_opts = PretrainedResolveOptions {
        cache_dir: Some(cache),
        use_gpu: false,
        vram_budget_bytes: None,
        ..Default::default()
    };
    let cpu_res =
        resolve_pretrained_model_with_probe("Qwen/Qwen2.5-0.5B-Instruct-GGUF", &cpu_opts, || None)
            .unwrap();
    assert_eq!(cpu_res.selected_quant.as_deref(), Some("Q8_0"));

    let _ = ureq::get(&format!("http://{addr}/__stop")).call();
    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

fn make_sparse_gguf(path: &Path, arch: &str, tokens: &[&str], sparse_size: u64) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&2u64.to_le_bytes()); // n_kv

    let write_s = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    write_s(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    write_s(&mut buf, arch);

    write_s(&mut buf, "tokenizer.ggml.tokens");
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for tok in tokens {
        write_s(&mut buf, tok);
    }

    let mut f = File::create(path).unwrap();
    std::io::Write::write_all(&mut f, &buf).unwrap();
    if sparse_size > buf.len() as u64 {
        f.set_len(sparse_size).unwrap();
    }
}

#[test]
fn resolve_auto_draft_model_with_probe_covers_guards_matching_and_incompatibility() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    // Cover all `select_auto_draft_repo` branches including empty-arch hints.
    assert_eq!(
        select_auto_draft_repo("", "my-qwen3-8b.gguf"),
        Some("Qwen/Qwen3-0.6B-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("", "QwQ-32B.gguf"),
        Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("", "gemma-3-27b.gguf"),
        Some("ggml-org/gemma-3-1b-it-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("gemma3", ""),
        Some("ggml-org/gemma-3-1b-it-GGUF")
    );
    assert_eq!(
        select_auto_draft_repo("llama", "DeepSeek-R1-Distill-Llama-8B"),
        Some("bartowski/Llama-3.2-1B-Instruct-GGUF")
    );
    assert_eq!(select_auto_draft_repo("mistral", "Mistral-7B"), None);

    let tmp = tempfile::tempdir().unwrap();
    let vocab = &["<bos>", "hello", "world"];

    // Small target (< 2 GiB) -> Ok(None).
    let small_p = tmp.path().join("small.gguf");
    make_sparse_gguf(&small_p, "qwen3", vocab, 4096);
    let small_target = ResolvedModel {
        local_path: small_p,
        source_uri: None,
        resolved_repo: None,
        selected_quant: None,
    };
    assert!(
        resolve_auto_draft_model(
            &small_target,
            Some(tmp.path().to_path_buf()),
            None,
            false,
            None
        )
        .unwrap()
        .is_none()
    );

    // Large 3 GiB sparse target (uses 0 disk blocks on Linux).
    let big_p = tmp.path().join("big-qwen3.gguf");
    make_sparse_gguf(&big_p, "qwen3", vocab, 3 * 1024 * 1024 * 1024);
    let big_target = ResolvedModel {
        local_path: big_p,
        source_uri: None,
        resolved_repo: Some("Qwen/Qwen3-8B-GGUF".into()),
        selected_quant: Some("Q4_K_M".into()),
    };

    // Low VRAM (< 1.5 GiB) -> Ok(None).
    assert!(
        resolve_auto_draft_model_with_probe(
            &big_target,
            Some(tmp.path().to_path_buf()),
            None,
            true,
            None,
            || Some(512 * 1024 * 1024),
        )
        .unwrap()
        .is_none()
    );

    // Huge VRAM (target fits 100% resident) -> Ok(None).
    assert!(
        resolve_auto_draft_model_with_probe(
            &big_target,
            Some(tmp.path().to_path_buf()),
            None,
            true,
            None,
            || Some(16 * 1024 * 1024 * 1024),
        )
        .unwrap()
        .is_none()
    );

    // Unknown architecture -> Ok(None).
    let mistral_p = tmp.path().join("big-mistral.gguf");
    make_sparse_gguf(&mistral_p, "mistral", vocab, 3 * 1024 * 1024 * 1024);
    let mistral_target = ResolvedModel {
        local_path: mistral_p,
        source_uri: None,
        resolved_repo: None,
        selected_quant: None,
    };
    assert!(
        resolve_auto_draft_model_with_probe(
            &mistral_target,
            Some(tmp.path().to_path_buf()),
            Some(2 * 1024 * 1024 * 1024),
            true,
            None,
            || None,
        )
        .unwrap()
        .is_none()
    );

    // End-to-end mock HTTP server for `resolve_auto_draft_model_with_probe` + `resolve_pretrained_model_with_probe`.
    let _env_guard = ENV_LOCK.lock().unwrap();
    let draft_ok_file = tmp.path().join("draft_ok_src.gguf");
    make_sparse_gguf(&draft_ok_file, "qwen3", vocab, 1024);
    let draft_ok_bytes = fs::read(&draft_ok_file).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let draft_srv = draft_ok_bytes.clone();
    let server = std::thread::spawn(move || {
        for stream_res in listener.incoming() {
            let Ok(mut stream) = stream_res else { break };
            let mut req_buf = [0u8; 4096];
            let n = stream.read(&mut req_buf).unwrap_or(0);
            if n == 0 {
                continue;
            }
            let req = String::from_utf8_lossy(&req_buf[..n]);
            let path = req
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("");
            if path.ends_with("/__stop") {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            } else if path.contains("/api/models/Qwen/Qwen3-0.6B-GGUF/tree/") {
                let json_body =
                    r#"[{"type":"file","path":"Qwen3-0.6B-Q4_K_M.gguf","size":500000000}]"#;
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    json_body.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(json_body.as_bytes());
            } else if path.contains("/api/models/Org/Empty-GGUF/tree/") {
                let json_body = "[]";
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    json_body.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(json_body.as_bytes());
            } else if path.ends_with("Qwen3-0.6B-Q4_K_M.gguf") {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    draft_srv.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&draft_srv);
            } else {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            }
        }
    });

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };
    let cache = tmp.path().join("cache");

    // 1. Compatible Qwen3 auto-draft resolves and passes vocab compatibility check!
    let resolved_draft = resolve_auto_draft_model_with_probe(
        &big_target,
        Some(cache.clone()),
        Some(2 * 1024 * 1024 * 1024),
        true,
        Some(Arc::new(|_| Ok(()))),
        || None,
    )
    .unwrap();
    assert!(resolved_draft.is_some());

    // 2. Progress callback abort propagates `HfError::Cancelled`.
    let cache2 = tmp.path().join("cache2");
    let abort_cb: ProgressCallback = Arc::new(|_| Err(HfError::Cancelled("aborted".into())));
    let err = resolve_auto_draft_model_with_probe(
        &big_target,
        Some(cache2),
        Some(2 * 1024 * 1024 * 1024),
        true,
        Some(abort_cb),
        || None,
    )
    .unwrap_err();
    assert!(matches!(err, HfError::Cancelled(_)));

    // 3. Explicit `hf://` URI with revision override + `-GGUF` repo with no files -> NotFound.
    let mut uri_opts = PretrainedResolveOptions {
        revision: Some("v1".into()),
        cache_dir: Some(cache.clone()),
        use_gpu: true,
        ..Default::default()
    };
    let res_uri = resolve_pretrained_model_with_probe(
        "hf://Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf",
        &uri_opts,
        || Some(8 * 1024 * 1024 * 1024),
    )
    .unwrap();
    assert!(res_uri.source_uri.unwrap().ends_with("@v1"));

    uri_opts.revision = None;
    let empty_err =
        resolve_pretrained_model_with_probe("Org/Empty-GGUF", &uri_opts, || Some(8 << 30))
            .unwrap_err();
    assert!(matches!(empty_err, HfError::NotFound(_)));

    let _ = ureq::get(&format!("http://{addr}/__stop")).call();
    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

#[test]
fn group_gguf_tree_entries_covers_directories_subdirs_and_bad_shard_suffixes() {
    let entries = vec![
        HfTreeEntry {
            entry_type: Some("directory".into()),
            path: "Q4_K_M".into(),
            size: 0,
            lfs: None,
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "README.md".into(),
            size: 100,
            lfs: None,
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "mmproj-F16.gguf".into(),
            size: 500,
            lfs: None,
        },
        // Subdirectory multi-part shards (out-of-order shard 2 before shard 1).
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf".into(),
            size: 1000,
            lfs: Some(HfLfsInfo { size: 2000 }),
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf".into(),
            size: 1000,
            lfs: Some(HfLfsInfo { size: 3000 }),
        },
        // Non-shard filenames that look similar to shard suffixes (tests strip_shard_suffix guards).
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "Model-00001-of-999.gguf".into(),
            size: 111,
            lfs: None,
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "Model-00001-of-00001.gguf".into(),
            size: 222,
            lfs: None,
        },
        HfTreeEntry {
            entry_type: Some("file".into()),
            path: "Model-12-of-00002.gguf".into(),
            size: 333,
            lfs: None,
        },
    ];

    let cands = group_gguf_tree_entries(&entries);
    assert_eq!(cands.len(), 4);
    let multi = cands
        .iter()
        .find(|c| c.shard_count == 2)
        .expect("multi-shard candidate");
    assert_eq!(
        multi.primary_file,
        "Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf"
    );
    assert_eq!(multi.total_bytes, 5000);
}

#[test]
fn parse_collection_slug_accepts_forms_and_rejects_bad_inputs() {
    assert_eq!(
        parse_collection_slug("kleinbem/layerstream-edge-123").unwrap(),
        "kleinbem/layerstream-edge-123"
    );
    assert_eq!(
        parse_collection_slug(
            "https://huggingface.co/collections/kleinbem/layerstream-edge-123/?utm_source=share#top"
        )
        .unwrap(),
        "kleinbem/layerstream-edge-123"
    );
    assert_eq!(
        parse_collection_slug("collections/kleinbem/layerstream-edge-123").unwrap(),
        "kleinbem/layerstream-edge-123"
    );
    assert!(parse_collection_slug("no-slash").is_err());
    assert!(parse_collection_slug("a/b/c").is_err());
    assert!(parse_collection_slug("../evil").is_err());
}

#[test]
fn fetch_hf_collection_handles_success_not_found_http_error_and_bad_json() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let server = std::thread::spawn(move || {
        for _ in 0..4 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut first_line = String::new();
            let _ = reader.read_line(&mut first_line);
            let target = first_line.split_whitespace().nth(1).unwrap_or("");
            let (status, body) = match target {
                "/api/collections/kleinbem/ok-col" => (
                    "200 OK",
                    r#"{"slug":"kleinbem/ok-col","title":"Edge Benchmarks","description":"desc","items":[{"id":"Qwen/Qwen2.5-14B-Instruct","type":"model","note":{"text":"4.87 tok/s"}},{"id":"kleinbem/layerstream-benchmarks","type":"space"},{"id":"allenai/OLMoE-1B-7B-0125-Instruct","type":"model"}]}"#,
                ),
                "/api/collections/kleinbem/missing-col" => ("404 Not Found", "not found"),
                "/api/collections/kleinbem/err-col" => ("500 Internal Server Error", "boom"),
                "/api/collections/kleinbem/bad-json" => ("200 OK", "not-json"),
                _ => ("400 Bad Request", "unexpected path"),
            };
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });

    let endpoint = format!("http://{addr}");
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .http_status_as_error(false)
        .proxy(None)
        .build()
        .new_agent();

    let ok = fetch_hf_collection_with_agent("kleinbem/ok-col", &endpoint, &agent).unwrap();
    assert_eq!(ok.title, "Edge Benchmarks");
    assert_eq!(
        ok.model_ids(),
        vec![
            "Qwen/Qwen2.5-14B-Instruct",
            "allenai/OLMoE-1B-7B-0125-Instruct"
        ]
    );
    assert!(matches!(
        fetch_hf_collection_with_agent("kleinbem/missing-col", &endpoint, &agent),
        Err(HfError::NotFound(_))
    ));
    match fetch_hf_collection_with_agent("kleinbem/err-col", &endpoint, &agent) {
        Err(HfError::Http(msg)) => assert!(msg.contains("HTTP 500"), "unexpected message: {msg}"),
        other => panic!("expected HfError::Http(500), got {other:?}"),
    }
    assert!(matches!(
        fetch_hf_collection_with_agent("kleinbem/bad-json", &endpoint, &agent),
        Err(HfError::Json { .. })
    ));
    assert!(matches!(
        fetch_hf_collection("invalid-no-slash"),
        Err(HfError::InvalidSpec(_))
    ));
    let prev_ep = std::env::var("HF_ENDPOINT").ok();
    // SAFETY: serialized under ENV_LOCK.
    unsafe {
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1");
    }
    let wrapper_res = fetch_hf_collection("kleinbem/unreachable-col");
    // SAFETY: serialized under ENV_LOCK.
    unsafe {
        match prev_ep {
            Some(v) => std::env::set_var("HF_ENDPOINT", v),
            None => std::env::remove_var("HF_ENDPOINT"),
        }
    }
    assert!(matches!(wrapper_res, Err(HfError::Http(_))));

    server.join().unwrap();
}
