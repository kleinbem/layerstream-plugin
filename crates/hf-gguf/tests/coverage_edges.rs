#![allow(clippy::unwrap_used, clippy::undocumented_unsafe_blocks)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use hf_gguf::{
    HfSpec, PretrainedResolveOptions, default_cache_dir, list_hf_gguf_candidates,
    resolve_pretrained_model,
};

fn spawn_one_shot_http(status: &str, body: &str) -> (String, thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut req_buf = [0u8; 4096];
        let n = stream.read(&mut req_buf).unwrap_or(0);
        let _ = stream.write_all(resp.as_bytes());
        let _ = stream.flush();
        String::from_utf8_lossy(&req_buf[..n]).into_owned()
    });
    (format!("http://{addr}"), handle)
}

#[test]
fn hub_tilde_cache_expansion_auth_header_and_invalid_json_paths() {
    // 1. Tilde HOME expansion in local model path lookup.
    let tilde_err = resolve_pretrained_model(
        "~/definitely-missing-layerstream-tilde-test.gguf",
        &PretrainedResolveOptions::default(),
    )
    .unwrap_err();
    let _ = default_cache_dir();
    assert!(format!("{tilde_err:#}").contains("definitely-missing-layerstream-tilde-test.gguf"));

    // 2. Same-origin Bearer auth header + invalid tree JSON error.
    let (endpoint, handle) = spawn_one_shot_http("200 OK", "not valid json");
    unsafe {
        std::env::set_var("HF_ENDPOINT", &endpoint);
        std::env::set_var("HF_TOKEN", "hf_test_secret_token");
    }
    let err = list_hf_gguf_candidates("Org/Broken-GGUF", "main").unwrap_err();
    let req_text = handle.join().unwrap();
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("HF_TOKEN");
    }
    assert!(
        req_text.contains("Authorization: Bearer hf_test_secret_token")
            || req_text.contains("authorization: Bearer hf_test_secret_token"),
        "expected Bearer header in request: {req_text}"
    );
    assert!(format!("{err:#}").contains("parse Hugging Face tree JSON"));

    // 3. HfSpec::parse + local missing file rejection.
    let parsed = HfSpec::parse("hf://Org/Repo:model.gguf@rev1").unwrap();
    assert_eq!(parsed.repo_id, "Org/Repo");
    assert_eq!(parsed.filename, "model.gguf");
    assert_eq!(parsed.revision, "rev1");

    let missing = resolve_pretrained_model(
        "/definitely/missing/local-model.gguf",
        &PretrainedResolveOptions::default(),
    );
    assert!(missing.is_err());
}

#[test]
fn download_truncation_progress_callback_and_sha256_mismatch_paths() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();

    // 1. Truncated download (advertises Content-Length: 1024, closes after 32 bytes)
    let l1 = TcpListener::bind("127.0.0.1:0").unwrap();
    let a1 = l1.local_addr().unwrap();
    let h1 = thread::spawn(move || {
        if let Ok((mut s, _)) = l1.accept() {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let _ = s
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n");
            let _ = s.write_all(&[0xAB; 32]);
            let _ = s.flush();
        }
    });
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{a1}"));
    }
    let err_trunc = hf_gguf::ensure_hf_downloaded("hf://Org/Trunc:m.gguf", dir.path()).unwrap_err();
    assert!(!format!("{err_trunc:#}").is_empty());
    let _ = h1.join();

    // 2. Slow chunked download triggers on_progress callback (>100ms between chunks)
    let l2 = TcpListener::bind("127.0.0.1:0").unwrap();
    let a2 = l2.local_addr().unwrap();
    let h2 = thread::spawn(move || {
        if let Ok((mut s, _)) = l2.accept() {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let _ =
                s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\nConnection: close\r\n\r\n");
            let _ = s.write_all(&[0x11; 32]);
            let _ = s.flush();
            thread::sleep(Duration::from_millis(130));
            let _ = s.write_all(&[0x22; 32]);
            let _ = s.flush();
        }
    });
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{a2}"));
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();
    let cb: hf_gguf::ProgressCallback = Arc::new(move |_p| {
        calls_cb.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let ok_path =
        hf_gguf::ensure_hf_downloaded_with_progress("hf://Org/Slow:m.gguf", dir.path(), Some(&cb))
            .unwrap();
    assert!(ok_path.is_file());
    assert!(calls.load(Ordering::Relaxed) >= 2);
    let _ = h2.join();

    // 3. SHA-256 mismatch via X-Linked-Etag removes corrupt .part file and errors
    let l3 = TcpListener::bind("127.0.0.1:0").unwrap();
    let a3 = l3.local_addr().unwrap();
    let fake_sha = "0".repeat(64);
    let h3 = thread::spawn(move || {
        if let Ok((mut s, _)) = l3.accept() {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let hdr = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nX-Linked-Etag: \"{fake_sha}\"\r\nConnection: close\r\n\r\n"
            );
            let _ = s.write_all(hdr.as_bytes());
            let _ = s.write_all(&[0x55; 16]);
            let _ = s.flush();
        }
    });
    unsafe {
        std::env::set_var("HF_ENDPOINT", format!("http://{a3}"));
    }
    let err_sha = hf_gguf::ensure_hf_downloaded("hf://Org/BadSha:m.gguf", dir.path()).unwrap_err();
    assert!(format!("{err_sha:#}").contains("SHA-256 mismatch"));
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
    }
    let _ = h3.join();
}

#[test]
fn fuzz_gguf_header_and_hf_spec_parsers_never_panic() {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    let mut next_u64 = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    // 1. Fuzz remote GGUF header parser across 256 random/mutated inputs.
    for case in 0..256u32 {
        let len = (next_u64() as usize) % 512;
        let mut raw = Vec::with_capacity(len + 8);
        if case % 2 == 0 {
            raw.extend_from_slice(b"GGUF");
            raw.extend_from_slice(&3u32.to_le_bytes());
        }
        for _ in 0..len {
            raw.push((next_u64() & 0xff) as u8);
        }
        let _ = hf_gguf::parse_gguf_header_metadata(&raw);
    }

    // 2. Fuzz HfSpec::parse and ensure path-traversal attempts are always rejected.
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789/:.@_-\\";
    for _ in 0..256 {
        let len = (next_u64() as usize) % 96;
        let mut s = String::from("hf://");
        for _ in 0..len {
            let idx = (next_u64() as usize) % alphabet.len();
            s.push(alphabet[idx] as char);
        }
        if let Ok(spec) = hf_gguf::HfSpec::parse(&s) {
            assert!(!spec.repo_id.contains(".."));
            assert!(!spec.filename.contains(".."));
            assert!(!spec.filename.starts_with('/'));
            assert!(!spec.filename.contains('\\'));
        }
    }

    // 3. Incomplete multi-part shard groups (missing shard 2 or missing shard 1) are filtered out.
    let entries: Vec<hf_gguf::HfTreeEntry> = serde_json::from_value(serde_json::json!([
        {"type": "file", "path": "complete-Q4_K_M-00001-of-00002.gguf", "size": 1000},
        {"type": "file", "path": "complete-Q4_K_M-00002-of-00002.gguf", "size": 2000},
        {"type": "file", "path": "partial-Q8_0-00001-of-00002.gguf", "size": 3000},
        {"type": "file", "path": "orphan-Q6_K-00002-of-00002.gguf", "size": 4000},
        {"type": "file", "path": "badidx-Q4_0-00000-of-00002.gguf", "size": 500}
    ]))
    .unwrap();
    let grouped = hf_gguf::group_gguf_tree_entries(&entries);
    assert_eq!(grouped.len(), 1);
    assert_eq!(
        grouped[0].primary_file,
        "complete-Q4_K_M-00001-of-00002.gguf"
    );
    assert_eq!(grouped[0].total_bytes, 3000);
    assert_eq!(grouped[0].shard_count, 2);
}
