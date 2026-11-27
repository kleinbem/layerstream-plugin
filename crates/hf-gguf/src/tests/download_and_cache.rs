use super::ENV_LOCK;
use crate::*;

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
