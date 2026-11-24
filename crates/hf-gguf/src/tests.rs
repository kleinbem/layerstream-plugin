use super::*;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn parses_colon_slash_and_revision_specs() {
    let s1 =
        HfSpec::parse("hf://bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q4_K_M.gguf")
            .unwrap();
    assert_eq!(s1.repo_id, "bartowski/Qwen2.5-14B-Instruct-GGUF");
    assert_eq!(s1.filename, "Qwen2.5-14B-Instruct-Q4_K_M.gguf");
    assert_eq!(s1.revision, "main");
    assert!(s1.download_url().ends_with(
        "/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    ));

    let s2 = HfSpec::parse("hf://unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M/model.gguf@v1.0").unwrap();
    assert_eq!(s2.repo_id, "unsloth/Qwen3-30B-A3B-GGUF");
    assert_eq!(s2.filename, "Q4_K_M/model.gguf");
    assert_eq!(s2.revision, "v1.0");

    let s3 = HfSpec::parse("hf://org/repo/model.gguf").unwrap();
    assert_eq!(s3.repo_id, "org/repo");
    assert_eq!(s3.filename, "model.gguf");
}

#[test]
fn expands_multipart_gguf_shards() {
    let s = HfSpec::parse("hf://org/big-gguf:Q4_K_M/big-00001-of-00003.gguf").unwrap();
    let shards = s.expand_shards();
    assert_eq!(shards.len(), 3);
    assert_eq!(shards[0].filename, "Q4_K_M/big-00001-of-00003.gguf");
    assert_eq!(shards[1].filename, "Q4_K_M/big-00002-of-00003.gguf");
    assert_eq!(shards[2].filename, "Q4_K_M/big-00003-of-00003.gguf");
}

#[test]
fn rejects_path_traversal_and_bad_repo() {
    assert!(HfSpec::parse("hf://badrepo:file.gguf").is_err());
    assert!(HfSpec::parse("hf://org/repo:../escape.gguf").is_err());
    assert!(HfSpec::parse("/local/path.gguf").is_err());
}

#[test]
fn revision_cache_paths_are_isolated() {
    let tmp = Path::new("/tmp/ls-cache");
    let a = HfSpec::parse("hf://org/repo:model.gguf@v1").unwrap();
    let b = HfSpec::parse("hf://org/repo:model.gguf@v2").unwrap();
    let m = HfSpec::parse("hf://org/repo:model.gguf").unwrap();
    assert_ne!(a.cache_path(tmp), b.cache_path(tmp));
    assert_ne!(a.cache_path(tmp), m.cache_path(tmp));
}

#[test]
fn resumable_range_download_etag_sha256_and_offline_cache_hit() {
    let _env_guard = ENV_LOCK.lock().unwrap();
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let payload_v1: Vec<u8> = (0..32_768).map(|i| ((i * 31 + 7) % 251) as u8).collect();
    let payload_v2: Vec<u8> = (0..32_768).map(|i| ((i * 17 + 3) % 241) as u8).collect();
    let sha_v1 = to_hex_lower(&Sha256::digest(&payload_v1));
    let sha_v2 = to_hex_lower(&Sha256::digest(&payload_v2));
    let etag_v1 = format!("\"{sha_v1}\"");
    let etag_v2 = format!("\"{sha_v2}\"");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_bg = hits.clone();
    let pv1 = payload_v1.clone();
    let pv2 = payload_v2.clone();
    let ev1 = etag_v1.clone();
    let ev2 = etag_v2.clone();

    // Request sequence:
    // 1: Range + matching If-Range(v1) -> 206 Partial Content of v1
    // 2: Range + oversized offset -> 416
    // 3: Full GET after 416 -> 200 OK of v1
    // 4: Range + stale If-Range(v1) while server now serves v2 -> 200 OK of v2 (no splice!)
    let server = std::thread::spawn(move || {
        for (req_no, stream) in listener.incoming().take(4).enumerate() {
            let Ok(mut sock) = stream else { break };
            hits_bg.fetch_add(1, Ordering::SeqCst);
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            let mut range_start: Option<usize> = None;
            let mut if_range: Option<String> = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(rest) = lower.strip_prefix("range: bytes=")
                    && let Some((start_s, _)) = rest.trim().split_once('-')
                    && let Ok(s) = start_s.parse::<usize>()
                {
                    range_start = Some(s);
                } else if let Some(rest) = lower.strip_prefix("if-range:") {
                    if_range = Some(rest.trim().to_string());
                }
            }
            let (active_payload, active_etag) = if req_no < 3 {
                (&pv1, &ev1)
            } else {
                (&pv2, &ev2)
            };
            let etag_matches = if_range.as_deref() == Some(active_etag.as_str());
            match range_start {
                Some(start) if start < active_payload.len() && etag_matches => {
                    let slice = &active_payload[start..];
                    let header = format!(
                        "HTTP/1.1 206 Partial Content\r\n\
                             ETag: {active_etag}\r\n\
                             Content-Range: bytes {}-{}/{}\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n",
                        start,
                        active_payload.len() - 1,
                        active_payload.len(),
                        slice.len()
                    );
                    sock.write_all(header.as_bytes()).unwrap();
                    sock.write_all(slice).unwrap();
                }
                Some(start) if start >= active_payload.len() && etag_matches => {
                    let header = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    sock.write_all(header.as_bytes()).unwrap();
                }
                _ => {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\n\
                             ETag: {active_etag}\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n",
                        active_payload.len()
                    );
                    sock.write_all(header.as_bytes()).unwrap();
                    sock.write_all(active_payload).unwrap();
                }
            }
            let _ = sock.flush();
        }
    });

    let tmp = std::env::temp_dir().join(format!("layerstream-hf-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };

    // 1. Pre-populate partial `.part` + matching `.part.meta` ETag -> verifies 206 resume + SHA-256.
    let uri = "hf://testorg/testrepo:model.gguf";
    let spec = HfSpec::parse(uri).unwrap();
    let dest = spec.cache_path(&tmp);
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    fs::write(part_path_for(&dest), &payload_v1[..4096]).unwrap();
    write_part_meta(
        &meta_path_for(&dest),
        &PartMeta {
            etag: Some(etag_v1.clone()),
            sha256: Some(sha_v1.clone()),
        },
    );

    let local = ensure_hf_downloaded(uri, &tmp).unwrap();
    assert_eq!(local, dest);
    assert_eq!(fs::read(&local).unwrap(), payload_v1);
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    // 2. Second call must be a 0-network cache hit.
    let local2 = ensure_hf_downloaded(uri, &tmp).unwrap();
    assert_eq!(local2, dest);
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    // 3. Oversized stale `.part` with matching ETag triggers 416 recovery + clean download (2 requests).
    fs::remove_file(&dest).unwrap();
    fs::write(part_path_for(&dest), vec![0u8; payload_v1.len() + 1024]).unwrap();
    write_part_meta(
        &meta_path_for(&dest),
        &PartMeta {
            etag: Some(etag_v1.clone()),
            sha256: Some(sha_v1.clone()),
        },
    );
    let local3 = ensure_hf_downloaded(uri, &tmp).unwrap();
    assert_eq!(fs::read(&local3).unwrap(), payload_v1);
    assert_eq!(hits.load(Ordering::SeqCst), 3);

    // 4. Stale `.part` (v1 prefix + v1 ETag) against updated upstream v2 -> restarts cleanly (NEVER splices v1+v2!).
    fs::remove_file(&dest).unwrap();
    fs::write(part_path_for(&dest), &payload_v1[..4096]).unwrap();
    write_part_meta(
        &meta_path_for(&dest),
        &PartMeta {
            etag: Some(etag_v1),
            sha256: Some(sha_v1),
        },
    );
    let local4 = ensure_hf_downloaded(uri, &tmp).unwrap();
    assert_eq!(fs::read(&local4).unwrap(), payload_v2);
    assert_eq!(hits.load(Ordering::SeqCst), 4);

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    let _ = fs::remove_dir_all(&tmp);
    server.join().unwrap();
}

#[test]
fn stalls_time_out_cleanly() {
    let _env_guard = ENV_LOCK.lock().unwrap();
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let mut req = [0u8; 1024];
            let _ = sock.read(&mut req);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\nConnection: close\r\n\r\n",
            );
            let _ = sock.flush();
            std::thread::sleep(Duration::from_secs(3));
        }
    });

    let tmp = std::env::temp_dir().join(format!("layerstream-hf-stall-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
        std::env::set_var("LAYERSTREAM_HF_READ_TIMEOUT_SECS", "1");
    }
    let err = ensure_hf_downloaded("hf://stallorg/stallrepo:model.gguf", &tmp);
    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("LAYERSTREAM_HF_READ_TIMEOUT_SECS");
    }
    assert!(err.is_err(), "stalled download must time out");
    let _ = fs::remove_dir_all(&tmp);
    let _ = server.join();
}

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
    let abort_cb: ProgressCallback = Arc::new(|_ev| bail!("user aborted download"));
    let err =
        ensure_hf_downloaded_with_progress("hf://Org/Repo:other.gguf", tmp.path(), Some(&abort_cb))
            .unwrap_err();
    assert!(err.to_string().contains("user aborted download"));

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

#[test]
fn cache_path_prevents_repo_and_revision_collisions() {
    let tmp = Path::new("/tmp/ls-cache");
    let a = HfSpec::parse("hf://org/a--b:model.gguf@main").unwrap();
    let b = HfSpec::parse("hf://org--a/b:model.gguf@main").unwrap();
    assert_ne!(a.cache_path(tmp), b.cache_path(tmp));

    let r1 = HfSpec::parse("hf://org/repo:model.gguf@feat/a").unwrap();
    let r2 = HfSpec::parse("hf://org/repo:model.gguf@feat_a").unwrap();
    assert_ne!(r1.cache_path(tmp), r2.cache_path(tmp));
}

#[test]
fn find_shards_in_hf_hub_root_resolves_ref_sha_and_rejects_partial_shards() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path();
    let sha = "50968a4468ef4233ed78cd7c3de230dd1d61a56b";
    let model_dir = hub.join("models--unsloth--Qwen3-0.6B-GGUF");
    let refs_dir = model_dir.join("refs");
    let snap_dir = model_dir.join("snapshots").join(sha);
    fs::create_dir_all(&refs_dir).unwrap();
    fs::create_dir_all(&snap_dir).unwrap();
    fs::write(refs_dir.join("main"), format!("{sha}\n")).unwrap();
    let gguf = snap_dir.join("Qwen3-0.6B-Q4_K_M.gguf");
    fs::write(&gguf, b"GGUF\x03\x00\x00\x00dummy").unwrap();

    // 1) Branch ref resolution (`main` -> `refs/main` -> `snapshots/<sha>/...`)
    let spec_ref = HfSpec::parse("hf://unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf").unwrap();
    let found = find_shards_in_hf_hub_root(&spec_ref, &spec_ref.expand_shards(), hub).unwrap();
    assert_eq!(found, vec![gguf.clone()]);

    // 2) Direct 40-hex commit SHA resolution
    let spec_sha = HfSpec::parse(&format!(
        "hf://unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf@{sha}"
    ))
    .unwrap();
    let found_sha = find_shards_in_hf_hub_root(&spec_sha, &spec_sha.expand_shards(), hub).unwrap();
    assert_eq!(found_sha, vec![gguf]);

    // 3) Multi-shard checkpoint must require all shards to be present
    let s1 = snap_dir.join("big-00001-of-00002.gguf");
    fs::write(&s1, b"GGUF\x03\x00\x00\x00shard1").unwrap();
    let spec_multi = HfSpec::parse("hf://unsloth/Qwen3-0.6B-GGUF:big-00001-of-00002.gguf").unwrap();
    assert!(find_shards_in_hf_hub_root(&spec_multi, &spec_multi.expand_shards(), hub).is_none());

    let s2 = snap_dir.join("big-00002-of-00002.gguf");
    fs::write(&s2, b"GGUF\x03\x00\x00\x00shard2").unwrap();
    let found_multi =
        find_shards_in_hf_hub_root(&spec_multi, &spec_multi.expand_shards(), hub).unwrap();
    assert_eq!(found_multi, vec![s1, s2]);
}
