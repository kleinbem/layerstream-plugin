#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::io::Write;

use gguf_direct_io::{
    AlignedBuf, DIRECT_IO_ALIGN, DirectSpanSlice, GgufIndex, GgufIoError, LayerIoPool, ModelMeta,
    TensorLocation, align_up, probe_dir_storage_mib_per_s, probe_storage_mib_per_s,
};

fn make_synthetic_pool(
    payload_len: usize,
) -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>, LayerIoPool) {
    let dir = tempfile::tempdir().unwrap();
    let shard_path = dir.path().join("direct_test.gguf");
    let mut payload = vec![0u8; payload_len];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = ((i * 31) ^ (i >> 7)) as u8;
    }
    std::fs::File::create(&shard_path)
        .unwrap()
        .write_all(&payload)
        .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "blk.0.w0".to_string(),
        TensorLocation {
            name: "blk.0.w0".to_string(),
            shard_idx: 0,
            abs_offset: 4096,
            ggml_type: 0,
            dims: vec![1024], // 4096 bytes
        },
    );
    tensors.insert(
        "blk.0.w1".to_string(),
        TensorLocation {
            name: "blk.0.w1".to_string(),
            shard_idx: 0,
            abs_offset: 12288,
            ggml_type: 0,
            dims: vec![1024], // 4096 bytes
        },
    );

    let index = GgufIndex {
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: Some("direct-test".to_string()),
            n_layers: 1,
            context_length: Some(256),
            total_file_bytes: payload.len() as u64,
        },
        shards: vec![shard_path.clone()],
        shard_sizes: vec![payload.len() as u64],
        tensors,
    };
    let pool = LayerIoPool::new(index, true).unwrap();
    (dir, shard_path, payload, pool)
}

#[test]
fn load_layer_direct_into_zero_copy_and_fallbacks_work() {
    let (_dir, _path, payload, pool) = make_synthetic_pool(64 * 1024);
    assert_eq!(pool.index().meta.architecture, "llama");

    let mut aligned = AlignedBuf::new(4096).unwrap();
    aligned.ensure_capacity(4096).unwrap(); // no-op when already >= 4096
    aligned.ensure_capacity(32 * 1024).unwrap(); // grow
    let pin = aligned.as_mut_slice(32 * 1024);

    let reqs = [("blk.0.w0", 0usize, 4096usize), ("blk.0.w1", 4096, 4096)];
    let mut slices = [DirectSpanSlice::default(); 4];

    // Negative layer index returns Ok(0) (fallback).
    assert_eq!(
        pool.load_layer_direct_into(-1, Some(&reqs), pin, &mut slices)
            .unwrap(),
        0
    );
    // Uncached layer with `requests = None` returns Ok(0).
    assert_eq!(
        pool.load_layer_direct_into(0, None, pin, &mut slices)
            .unwrap(),
        0
    );
    // Unaligned destination buffer returns Ok(0).
    assert_eq!(
        pool.load_layer_direct_into(0, Some(&reqs), &mut pin[1..], &mut slices)
            .unwrap(),
        0
    );

    // Valid aligned zero-copy direct load succeeds and returns 2 slices.
    let n = pool
        .load_layer_direct_into(0, Some(&reqs), pin, &mut slices)
        .unwrap();
    assert_eq!(n, 2);
    for s in &slices[..n] {
        let src = &pin[s.src_offset as usize..(s.src_offset + s.len) as usize];
        if s.dst_offset == 0 {
            assert_eq!(src, &payload[4096..8192]);
        } else {
            assert_eq!(src, &payload[12288..16384]);
        }
    }
    assert!(pool.stats().span_reads >= 1);

    // Second sweep uses cached layer plan with `requests = None`.
    let n2 = pool
        .load_layer_direct_into(0, None, pin, &mut slices)
        .unwrap();
    assert_eq!(n2, 2);

    // Undersized pin_buf or out_slices falls back cleanly with Ok(0).
    let mut tiny_slices = [DirectSpanSlice::default(); 1];
    assert_eq!(
        pool.load_layer_direct_into(0, None, pin, &mut tiny_slices)
            .unwrap(),
        0
    );
    assert_eq!(
        pool.load_layer_direct_into(0, None, &mut pin[..DIRECT_IO_ALIGN as usize], &mut slices)
            .unwrap(),
        0
    );

    // Empty `load_slices_into` is a no-op; missing tensor returns MissingTensor; dst overflow returns OutOfBounds.
    let mut dst = vec![0u8; 4096];
    pool.load_slices_into(&[], &mut dst).unwrap();
    assert!(matches!(
        pool.load_slices_into(&[("nope", 0, 0, 16, 0)], &mut dst)
            .unwrap_err(),
        GgufIoError::MissingTensor(_)
    ));
    assert!(matches!(
        pool.load_slices_into(&[("blk.0.w0", 0, 0, 8192, 0)], &mut dst)
            .unwrap_err(),
        GgufIoError::OutOfBounds(_)
    ));

    pool.clear_expert_cache();
}

#[test]
fn storage_probes_handle_tiny_missing_and_empty_directories() {
    let dir = tempfile::tempdir().unwrap();
    // Empty directory creates and removes an ephemeral probe file.
    assert!(probe_dir_storage_mib_per_s(dir.path()).is_some());

    // File smaller than 4 KiB returns (0.0, false).
    let tiny = dir.path().join("tiny.gguf");
    std::fs::write(&tiny, b"small").unwrap();
    assert_eq!(probe_storage_mib_per_s(&tiny, false).unwrap(), (0.0, false));

    // Missing file returns typed GgufIoError::Io.
    assert!(matches!(
        probe_storage_mib_per_s(&dir.path().join("missing.gguf"), false).unwrap_err(),
        GgufIoError::Io { .. }
    ));

    // Large >= 4 MiB aligned read exercises `align_up` + direct probe path.
    let big = dir.path().join("big.gguf");
    let size = align_up(4 * 1024 * 1024 + 4096, DIRECT_IO_ALIGN) as usize;
    std::fs::write(&big, vec![0xA5u8; size]).unwrap();
    let (mib_s, _) = probe_storage_mib_per_s(&big, true).unwrap();
    assert!(mib_s >= 0.0);
}

#[test]
fn large_layer_read_exercises_parallel_span_and_negative_cached_layer() {
    let dir = tempfile::tempdir().unwrap();
    let shard_path = dir.path().join("big_layer.gguf");
    let tensor_bytes = 5 * 1024 * 1024; // > 4 MiB PARALLEL_READ_MIN_BYTES
    let mut payload = vec![0u8; tensor_bytes + 8192];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = ((i * 7) ^ (i >> 11)) as u8;
    }
    std::fs::write(&shard_path, &payload).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "blk.0.big".to_string(),
        TensorLocation {
            name: "blk.0.big".to_string(),
            shard_idx: 0,
            abs_offset: 4096,
            ggml_type: 0,
            dims: vec![(tensor_bytes / 4) as u64],
        },
    );
    let index = GgufIndex {
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: None,
            n_layers: 1,
            context_length: None,
            total_file_bytes: payload.len() as u64,
        },
        shards: vec![shard_path],
        shard_sizes: vec![payload.len() as u64],
        tensors,
    };
    let pool = LayerIoPool::new(index, true).unwrap();
    assert!(!pool.load_cached_layer_into(-1, &mut []).unwrap());

    let mut dst = vec![0u8; tensor_bytes];
    pool.load_layer_tensors_into(0, &[("blk.0.big", 0, tensor_bytes)], &mut dst)
        .unwrap();
    assert_eq!(&dst[..4096], &payload[4096..8192]);
    assert_eq!(
        &dst[tensor_bytes - 4096..],
        &payload[4096 + tensor_bytes - 4096..4096 + tensor_bytes]
    );
}
