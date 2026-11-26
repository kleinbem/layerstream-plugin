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
