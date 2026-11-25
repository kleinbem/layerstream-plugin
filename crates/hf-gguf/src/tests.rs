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

#[test]
fn typed_hf_errors_and_alias_resolution_and_cache_check() {
    let tmp = tempfile::tempdir().unwrap();
    let local_gguf = tmp.path().join("local.gguf");
    fs::write(&local_gguf, b"GGUF\x03\x00\x00\x00test").unwrap();

    // Typed InvalidSpec on malformed hf:// URI and empty spec.
    assert!(matches!(
        HfSpec::parse("not-hf://org/repo:a.gguf").unwrap_err(),
        HfError::InvalidSpec(_)
    ));
    assert!(matches!(
        resolve_pretrained_model("   ", &PretrainedResolveOptions::default()).unwrap_err(),
        HfError::InvalidSpec(_)
    ));

    // Typed NotFound on missing local path.
    let missing = tmp.path().join("missing.gguf");
    assert!(matches!(
        resolve_pretrained_model(
            missing.to_str().unwrap(),
            &PretrainedResolveOptions::default()
        )
        .unwrap_err(),
        HfError::NotFound(_)
    ));

    // Local paths pass through `resolve_model_spec` preserving `alias=`.
    let aliased = format!("my-model={}", local_gguf.display());
    assert_eq!(resolve_model_spec(&aliased, tmp.path()).unwrap(), aliased);
    assert!(is_model_spec_cached(
        local_gguf.to_str().unwrap(),
        tmp.path()
    ));
    assert!(!is_model_spec_cached(
        "hf://nonexistent-org/nonexistent-repo:missing-unique-xyz.gguf",
        tmp.path()
    ));

    // `recommend_candidates` selects quant matching available VRAM budget.
    let cands = vec![
        HfGgufCandidate {
            quant: "Q4_K_M".to_string(),
            primary_file: "Model-Q4_K_M.gguf".to_string(),
            shard_count: 1,
            total_bytes: 16 * 1024 * 1024 * 1024,
        },
        HfGgufCandidate {
            quant: "Q2_K".to_string(),
            primary_file: "Model-Q2_K.gguf".to_string(),
            shard_count: 1,
            total_bytes: 6 * 1024 * 1024 * 1024,
        },
    ];
    let recs_small_vram = recommend_candidates(
        "org/Model-GGUF",
        "main",
        &cands,
        12 * 1024 * 1024 * 1024,
        2500.0,
        false,
    );
    let best_small = recs_small_vram.iter().find(|r| r.recommended).unwrap();
    assert_eq!(best_small.candidate.quant, "Q2_K");
    assert_eq!(best_small.verdict, "RESIDENT");

    let recs_large_vram = recommend_candidates(
        "org/Model-GGUF",
        "main",
        &cands,
        48 * 1024 * 1024 * 1024,
        2500.0,
        false,
    );
    let best_large = recs_large_vram.iter().find(|r| r.recommended).unwrap();
    assert_eq!(best_large.candidate.quant, "Q4_K_M");
    assert_eq!(best_large.verdict, "RESIDENT");
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
fn download_handles_416_range_mismatch_etag_change_truncation_sha_mismatch_and_redirects() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let _env_guard = ENV_LOCK.lock().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = b"GGUF_PAYLOAD_0123456789ABCDEF".to_vec();
    let body_sha = to_hex_lower(&Sha256::digest(&body));
    let req_count = Arc::new(AtomicUsize::new(0));
    let body_srv = body.clone();
    let sha_srv = body_sha.clone();
    let cnt_srv = req_count.clone();

    let server = std::thread::spawn(move || {
        for stream_res in listener.incoming() {
            let Ok(mut stream) = stream_res else { break };
            let mut req_buf = [0u8; 4096];
            let n = stream.read(&mut req_buf).unwrap_or(0);
            if n == 0 {
                continue;
            }
            let req = String::from_utf8_lossy(&req_buf[..n]);
            let first = req.lines().next().unwrap_or("");
            let path = first.split_whitespace().nth(1).unwrap_or("");
            let has_range = req.to_ascii_lowercase().contains("\nrange:");
            cnt_srv.fetch_add(1, Ordering::SeqCst);

            if path.ends_with("/__stop") {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            } else if path.ends_with("/err404.gguf") {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            } else if path.ends_with("/loop.gguf") {
                let resp = format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://{addr}/o/r/resolve/main/loop.gguf\r\nContent-Length: 0\r\n\r\n"
                );
                let _ = stream.write_all(resp.as_bytes());
            } else if path.ends_with("/trunc.gguf") {
                let resp =
                    b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\nConnection: close\r\n\r\nshort";
                let _ = stream.write_all(resp);
            } else if path.ends_with("/badsha.gguf") {
                let wrong = "0".repeat(64);
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"{wrong}\"\r\nConnection: close\r\n\r\n",
                    body_srv.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&body_srv);
            } else if path.ends_with("/r416.gguf") {
                if has_range {
                    let _ = stream.write_all(
                        b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                } else {
                    let hdr = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"{sha_srv}\"\r\nConnection: close\r\n\r\n",
                        body_srv.len()
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(&body_srv);
                }
            } else if path.ends_with("/etag_flip.gguf") {
                if has_range {
                    let hdr = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-{}/{}\r\nContent-Length: {}\r\nETag: \"new-etag\"\r\nConnection: close\r\n\r\n",
                        body_srv.len() - 1,
                        body_srv.len(),
                        body_srv.len() - 5
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(&body_srv[5..]);
                } else {
                    let hdr = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"new-etag\"\r\nConnection: close\r\n\r\n",
                        body_srv.len()
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(&body_srv);
                }
            } else if path.contains("/api/models/Org/Paged/tree/main") && !path.contains("cursor=2")
            {
                let page1 = r#"[{"type":"file","path":"Model-Q4_K_M.gguf","size":1000}]"#;
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nLink: <http://{addr}/api/models/Org/Paged/tree/main?cursor=2>; rel=\"next\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    page1.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(page1.as_bytes());
            } else if path.contains("cursor=2") {
                let page2 = r#"[{"type":"file","path":"Model-Q8_0.gguf","size":2000}]"#;
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    page2.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(page2.as_bytes());
            } else if path.contains("/api/models/Org/ApiErr/tree/main") {
                let _ = stream
                    .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
            }
        }
    });

    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };
    let tmp = tempfile::tempdir().unwrap();

    // 1. HTTP 404 -> HfError::Http
    assert!(matches!(
        ensure_hf_downloaded("hf://o/r:err404.gguf", tmp.path()).unwrap_err(),
        HfError::Http(_)
    ));

    // 2. Redirect loop -> HfError::Http
    assert!(matches!(
        ensure_hf_downloaded("hf://o/r:loop.gguf", tmp.path()).unwrap_err(),
        HfError::Http(_)
    ));

    // 3. Truncated body -> HfError::Io (UnexpectedEof from HTTP reader) or DownloadFailed
    assert!(matches!(
        ensure_hf_downloaded("hf://o/r:trunc.gguf", tmp.path()).unwrap_err(),
        HfError::Io { .. } | HfError::DownloadFailed(_)
    ));

    // 4. SHA-256 mismatch -> HfError::DownloadFailed & partial cleaned up
    assert!(matches!(
        ensure_hf_downloaded("hf://o/r:badsha.gguf", tmp.path()).unwrap_err(),
        HfError::DownloadFailed(_)
    ));

    // 5. HTTP 416 on stale .part restarts cleanly from byte 0
    let spec416 = HfSpec::parse("hf://o/r:r416.gguf").unwrap();
    let dest416 = spec416.cache_path(tmp.path());
    fs::create_dir_all(dest416.parent().unwrap()).unwrap();
    fs::write(part_path_for(&dest416), b"stale-bytes").unwrap();
    write_part_meta(
        &meta_path_for(&dest416),
        &PartMeta {
            etag: Some("\"old\"".into()),
            sha256: None,
        },
    );
    let p416 = ensure_hf_downloaded("hf://o/r:r416.gguf", tmp.path()).unwrap();
    assert_eq!(fs::read(&p416).unwrap(), body);

    // 6. ETag change during 206 resume restarts cleanly from byte 0
    let spec_flip = HfSpec::parse("hf://o/r:etag_flip.gguf").unwrap();
    let dest_flip = spec_flip.cache_path(tmp.path());
    fs::write(part_path_for(&dest_flip), b"12345").unwrap();
    write_part_meta(
        &meta_path_for(&dest_flip),
        &PartMeta {
            etag: Some("\"old-etag\"".into()),
            sha256: None,
        },
    );
    let p_flip = ensure_hf_downloaded("hf://o/r:etag_flip.gguf", tmp.path()).unwrap();
    assert_eq!(fs::read(&p_flip).unwrap(), body);

    // 7. Paginated Hugging Face tree listing via `Link: <...>; rel="next"` + API 500 error
    let cands = list_hf_gguf_candidates("Org/Paged", "main").unwrap();
    assert_eq!(cands.len(), 2);
    assert!(matches!(
        list_hf_gguf_candidates("Org/ApiErr", "main").unwrap_err(),
        HfError::Http(_)
    ));

    // Signal server shutdown.
    let _ = ureq::get(&format!("http://{addr}/__stop")).call();
    // SAFETY: Serialized by `ENV_LOCK` so no concurrent test thread reads or mutates process env vars.
    unsafe { std::env::remove_var("HF_ENDPOINT") };
    server.join().unwrap();
}

#[test]
fn covers_env_cache_dirs_hub_cache_hit_tokens_reentry_and_resume_edge_cases() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let _env_guard = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();

    // 1. `default_cache_dir` env priority: LAYERSTREAM_CACHE_DIR -> XDG_CACHE_HOME -> HOME.
    let saved_lc = std::env::var_os("LAYERSTREAM_CACHE_DIR");
    let saved_xdg = std::env::var_os("XDG_CACHE_HOME");
    let saved_hf_hub = std::env::var_os("HF_HUB_CACHE");
    let saved_hf_home = std::env::var_os("HF_HOME");
    let saved_hf_tok = std::env::var_os("HF_TOKEN");
    let saved_hub_tok = std::env::var_os("HUGGING_FACE_HUB_TOKEN");

    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::set_var("LAYERSTREAM_CACHE_DIR", tmp.path().join("custom-lc"));
    }
    assert_eq!(default_cache_dir(), tmp.path().join("custom-lc"));

    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::remove_var("LAYERSTREAM_CACHE_DIR");
        std::env::set_var("XDG_CACHE_HOME", tmp.path().join("xdg"));
    }
    assert_eq!(
        default_cache_dir(),
        tmp.path().join("xdg").join("layerstream").join("models")
    );

    // 2. `load_hf_token` via HF_TOKEN, HUGGING_FACE_HUB_TOKEN, and $HF_HOME/token file.
    let hf_home = tmp.path().join("hf-home");
    fs::create_dir_all(&hf_home).unwrap();
    fs::write(hf_home.join("token"), "file-tok-123\n").unwrap();
    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::set_var("HF_HOME", &hf_home);
        std::env::set_var("HF_TOKEN", "env-tok-1");
    }
    assert_eq!(load_hf_token().as_deref(), Some("env-tok-1"));
    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::remove_var("HF_TOKEN");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "env-tok-2");
    }
    assert_eq!(load_hf_token().as_deref(), Some("env-tok-2"));
    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
    assert_eq!(load_hf_token().as_deref(), Some("file-tok-123"));

    // 3. `hf_hub_cache_roots` + `ensure_hf_downloaded_with_progress` hitting HF Hub cache directly,
    //    plus `resolve_model_spec` (`alias=hf://...` and `hf://...`) and `is_model_spec_cached`.
    let hub_cache = tmp.path().join("hub-cache");
    let sha = "1234567890abcdef1234567890abcdef12345678";
    let snap = hub_cache
        .join("models--Org--CachedRepo")
        .join("snapshots")
        .join(sha);
    let refs = hub_cache.join("models--Org--CachedRepo").join("refs");
    fs::create_dir_all(&snap).unwrap();
    fs::create_dir_all(&refs).unwrap();
    fs::write(refs.join("main"), format!("{sha}\n")).unwrap();
    fs::write(refs.join("badref"), "../escape\n").unwrap();
    let hub_gguf = snap.join("m.gguf");
    fs::write(&hub_gguf, b"GGUF_HUB_CACHED").unwrap();

    // Invalid ref path and invalid ref contents return None cleanly.
    let bad_rev_spec = HfSpec {
        repo_id: "Org/CachedRepo".into(),
        filename: "m.gguf".into(),
        revision: "../bad".into(),
    };
    assert!(
        find_shards_in_hf_hub_root(
            &bad_rev_spec,
            std::slice::from_ref(&bad_rev_spec),
            &hub_cache
        )
        .is_none()
    );
    let bad_ref_content = HfSpec {
        repo_id: "Org/CachedRepo".into(),
        filename: "m.gguf".into(),
        revision: "badref".into(),
    };
    assert!(
        find_shards_in_hf_hub_root(
            &bad_ref_content,
            std::slice::from_ref(&bad_ref_content),
            &hub_cache
        )
        .is_none()
    );

    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::set_var("HF_HUB_CACHE", &hub_cache);
    }
    let empty_cache = tmp.path().join("ls-cache");
    let uri = "hf://Org/CachedRepo:m.gguf";
    assert!(is_model_spec_cached(
        &format!("myalias={uri}"),
        &empty_cache
    ));
    let hit_cb_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hit_flag = hit_cb_called.clone();
    let cb: ProgressCallback = Arc::new(move |ev| {
        assert!(ev.cached && ev.done);
        hit_flag.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    });
    let resolved = ensure_hf_downloaded_with_progress(uri, &empty_cache, Some(&cb)).unwrap();
    assert_eq!(resolved, hub_gguf);
    assert!(hit_cb_called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(
        resolve_model_spec(&format!("myalias={uri}"), &empty_cache).unwrap(),
        format!("myalias={}", hub_gguf.display())
    );
    assert_eq!(
        resolve_model_spec(uri, &empty_cache).unwrap(),
        hub_gguf.to_string_lossy()
    );
    assert_eq!(resolve_path_or_hf(uri, &empty_cache).unwrap(), hub_gguf);
    assert!(!is_path_or_hf_cached("hf://bad", &empty_cache));

    // 4. Mock server for mismatched Content-Range restart, weak ETag discard, relative redirect,
    //    Authorization header forwarding, and re-entrant progress-callback deadlock guard.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = b"GGUF_CONTENT_RANGE_AND_REENTRY_TEST".to_vec();
    let body_srv = body.clone();
    let saw_auth = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let auth_srv = saw_auth.clone();

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
            let lower = req.to_ascii_lowercase();
            let has_range = lower.contains("\nrange:");
            if lower.contains("authorization: bearer file-tok-123") {
                auth_srv.store(true, std::sync::atomic::Ordering::SeqCst);
            }

            if path.ends_with("/__stop") {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            } else if path.ends_with("/badcr.gguf") {
                if has_range {
                    // Return mismatched Content-Range start (99 != 5) -> client restarts from byte 0.
                    let hdr = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 99-110/200\r\nContent-Length: 12\r\nConnection: close\r\n\r\n123456789012";
                    let _ = stream.write_all(hdr.as_bytes());
                } else {
                    let hdr = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body_srv.len()
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(&body_srv);
                }
            } else if path.ends_with("/rel_redir.gguf") {
                let _ = stream.write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: /o/r/resolve/main/final_rel.gguf\r\nContent-Length: 0\r\n\r\n",
                );
            } else if path.ends_with("/final_rel.gguf") || path.ends_with("/reentry.gguf") {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body_srv.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&body_srv);
            }
        }
    });

    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
    }

    // Mismatched Content-Range on resume restarts cleanly from byte 0.
    let spec_cr = HfSpec::parse("hf://o/r:badcr.gguf").unwrap();
    let dest_cr = spec_cr.cache_path(&empty_cache);
    fs::create_dir_all(dest_cr.parent().unwrap()).unwrap();
    fs::write(part_path_for(&dest_cr), b"12345").unwrap();
    write_part_meta(
        &meta_path_for(&dest_cr),
        &PartMeta {
            etag: Some("\"strong-etag\"".into()),
            sha256: None,
        },
    );
    let p_cr = ensure_hf_downloaded("hf://o/r:badcr.gguf", &empty_cache).unwrap();
    assert_eq!(fs::read(&p_cr).unwrap(), body);
    assert!(saw_auth.load(std::sync::atomic::Ordering::SeqCst));

    // Weak ETag partial without SHA-256 is discarded and restarted via relative redirect.
    let spec_rel = HfSpec::parse("hf://o/r:rel_redir.gguf").unwrap();
    let dest_rel = spec_rel.cache_path(&empty_cache);
    fs::write(part_path_for(&dest_rel), b"weak-stale").unwrap();
    write_part_meta(
        &meta_path_for(&dest_rel),
        &PartMeta {
            etag: Some("W/\"weak\"".into()),
            sha256: None,
        },
    );
    let p_rel = ensure_hf_downloaded("hf://o/r:rel_redir.gguf", &empty_cache).unwrap();
    assert_eq!(fs::read(&p_rel).unwrap(), body);

    // Re-entrant download from inside progress callback fails closed instead of deadlocking.
    let reentry_cache = empty_cache.clone();
    let reentry_cb: ProgressCallback = Arc::new(move |_ev| {
        ensure_hf_downloaded("hf://o/r:reentry.gguf", &reentry_cache).map(|_| ())
    });
    let reentry_err = ensure_hf_downloaded_with_progress(
        "hf://o/r:reentry.gguf",
        &empty_cache,
        Some(&reentry_cb),
    )
    .unwrap_err();
    assert!(matches!(reentry_err, HfError::DownloadFailed(_)));
    assert!(reentry_err.to_string().contains("re-entrant"));

    let _ = ureq::get(&format!("http://{addr}/__stop")).call();
    server.join().unwrap();

    // Restore all modified environment variables.
    // SAFETY: Serialized by `ENV_LOCK`.
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
        match saved_lc {
            Some(v) => std::env::set_var("LAYERSTREAM_CACHE_DIR", v),
            None => std::env::remove_var("LAYERSTREAM_CACHE_DIR"),
        }
        match saved_xdg {
            Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
            None => std::env::remove_var("XDG_CACHE_HOME"),
        }
        match saved_hf_hub {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
        match saved_hf_home {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
        match saved_hf_tok {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
        match saved_hub_tok {
            Some(v) => std::env::set_var("HUGGING_FACE_HUB_TOKEN", v),
            None => std::env::remove_var("HUGGING_FACE_HUB_TOKEN"),
        }
    }
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

#[test]
fn covers_flock_race_emit_download_progress_and_link_parser_edge_cases() {
    use crate::hub::parse_next_link_url;

    // 1. `emit_download_progress` for both TTY and non-TTY, with and without `total_bytes`.
    let mut ev = DownloadProgress {
        repo_id: "o/r".into(),
        filename: "m.gguf".into(),
        shard_index: 1,
        shard_count: 1,
        downloaded_bytes: 512,
        total_bytes: Some(1024),
        mib_per_s: 25.0,
        cached: false,
        done: false,
    };
    emit_download_progress(&ev, true);
    emit_download_progress(&ev, false);
    ev.total_bytes = None;
    emit_download_progress(&ev, true);
    emit_download_progress(&ev, false);

    // 2. Concurrent flock race: thread 1 holds `PartLock`, thread 2 enters `download_with_resume`,
    //    thread 1 writes the completed file and drops the lock -> thread 2 observes cache hit!
    let tmp = tempfile::tempdir().unwrap();
    let spec = HfSpec::parse("hf://Org/Race:model.gguf").unwrap();
    let dest = spec.cache_path(tmp.path());
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let lock = PartLock::acquire(&lock_path_for(&dest)).unwrap();

    let cache_dir = tmp.path().to_path_buf();
    let saw_cached = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = saw_cached.clone();
    let t2 = std::thread::spawn(move || {
        let cb: ProgressCallback = Arc::new(move |p| {
            if p.cached && p.done {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        });
        ensure_hf_downloaded_with_progress("hf://Org/Race:model.gguf", &cache_dir, Some(&cb))
    });
    std::thread::sleep(std::time::Duration::from_millis(30));
    fs::write(&dest, b"COMPLETED_BY_OTHER_PROCESS").unwrap();
    drop(lock);
    let resolved = t2.join().unwrap().unwrap();
    assert_eq!(resolved, dest);
    assert!(saw_cached.load(std::sync::atomic::Ordering::SeqCst));

    // 3. `parse_next_link_url` edge cases: relative next URL, comma-separated links, missing rel.
    let mut h1 = ureq::http::HeaderMap::new();
    h1.insert(
        "link",
        "</api/page?cursor=3>; rel=\"next\", </prev>; rel=\"prev\""
            .parse()
            .unwrap(),
    );
    assert_eq!(
        parse_next_link_url(&h1, "https://huggingface.co/api/page").as_deref(),
        Some("https://huggingface.co/api/page?cursor=3")
    );
    let mut h2 = ureq::http::HeaderMap::new();
    h2.insert("link", "malformed; rel=\"next\"".parse().unwrap());
    assert_eq!(parse_next_link_url(&h2, "https://huggingface.co/api"), None);
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
