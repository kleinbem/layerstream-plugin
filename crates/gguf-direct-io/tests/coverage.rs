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
    // Exercise multi-span sparse io_uring batch path (2 distant slices > gap threshold).
    let mut sparse_dst = vec![0u8; 8192];
    pool.load_slices_into(
        &[
            ("blk.0.big", 0, 0, 4096, 0),
            ("blk.0.big", 2 * 1024 * 1024, 4096, 4096, 1),
        ],
        &mut sparse_dst,
    )
    .unwrap();
    assert_eq!(&sparse_dst[..4096], &payload[4096..8192]);
    assert_eq!(
        &sparse_dst[4096..],
        &payload[4096 + 2 * 1024 * 1024..4096 + 2 * 1024 * 1024 + 4096]
    );

    // Exercise env-configured parallel O_DIRECT pread fallback (`LAYERSTREAM_IO_URING=0`,
    // `LAYERSTREAM_IO_THREADS=2`), `LAYERSTREAM_MOE_IO_THREADS`, `LAYERSTREAM_SPARSE_DIRECT_IO=1`,
    // `LAYERSTREAM_IO_CHUNK_MB=4`, `LAYERSTREAM_EXPERT_CACHE_MB=4`, and
    // `LAYERSTREAM_PROFILE=1` (drop profile summary).
    // SAFETY: Tests run with `--test-threads 1` so env mutation is serialized.
    unsafe {
        std::env::set_var("LAYERSTREAM_IO_URING", "0");
        std::env::set_var("LAYERSTREAM_IO_THREADS", "2");
        std::env::set_var("LAYERSTREAM_MOE_IO_THREADS", "2");
        std::env::set_var("LAYERSTREAM_SPARSE_DIRECT_IO", "1");
        std::env::set_var("LAYERSTREAM_IO_CHUNK_MB", "4");
        std::env::set_var("LAYERSTREAM_EXPERT_CACHE_MB", "4");
        std::env::set_var("LAYERSTREAM_PROFILE", "1");
    }
    let mut tensors2 = HashMap::new();
    tensors2.insert(
        "blk.0.big".to_string(),
        TensorLocation {
            name: "blk.0.big".to_string(),
            shard_idx: 0,
            abs_offset: 4096,
            ggml_type: 0,
            dims: vec![(tensor_bytes / 4) as u64],
        },
    );
    let index2 = GgufIndex {
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: None,
            n_layers: 1,
            context_length: None,
            total_file_bytes: payload.len() as u64,
        },
        shards: vec![dir.path().join("big_layer.gguf")],
        shard_sizes: vec![payload.len() as u64],
        tensors: tensors2,
    };
    {
        let pool2 = LayerIoPool::new(index2, true).unwrap();
        let mut dst2 = vec![0u8; tensor_bytes];
        pool2
            .load_layer_tensors_into(0, &[("blk.0.big", 0, tensor_bytes)], &mut dst2)
            .unwrap();
        assert_eq!(dst2, dst);
        // Also exercise sparse O_DIRECT path and expert cache hit/miss counters before drop.
        let mut slice_dst = vec![0u8; 4096];
        pool2
            .load_slices_into(&[("blk.0.big", 0, 0, 4096, 0)], &mut slice_dst)
            .unwrap();
        assert_eq!(&slice_dst[..], &payload[4096..8192]);
        // Exercise persistent hot-expert profile save, clear, and sequential warm-up.
        let prof_path = dir.path().join("profiles/big_layer.lsep");
        let saved = pool2.save_expert_profile(&prof_path).unwrap();
        assert!(saved >= 1);
        pool2.clear_expert_cache();
        let warmed = pool2.warm_expert_profile(&prof_path).unwrap();
        assert_eq!(warmed, saved);
        let hits_before = pool2.stats().cache_hits;
        pool2
            .load_slices_into(&[("blk.0.big", 0, 0, 4096, 0)], &mut slice_dst)
            .unwrap();
        assert!(pool2.stats().cache_hits > hits_before);
        // Drop prints `[layerstream-io-profile]` summary.
    }
    // SAFETY: Tests run with `--test-threads 1` so env mutation is serialized.
    unsafe {
        std::env::remove_var("LAYERSTREAM_IO_URING");
        std::env::remove_var("LAYERSTREAM_IO_THREADS");
        std::env::remove_var("LAYERSTREAM_MOE_IO_THREADS");
        std::env::remove_var("LAYERSTREAM_SPARSE_DIRECT_IO");
        std::env::remove_var("LAYERSTREAM_IO_CHUNK_MB");
        std::env::remove_var("LAYERSTREAM_EXPERT_CACHE_MB");
        std::env::remove_var("LAYERSTREAM_PROFILE");
    }
}

#[test]
fn gguf_index_scalar_types_overflow_and_vocab_error_paths() {
    let dir = tempfile::tempdir().unwrap();

    // 1. Valid GGUF with every scalar metadata type (U8, I8, U16, I16, U64, I64, F64)
    //    and U64 block_count.
    let mut buf = Vec::new();
    let write_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    let mut kv = Vec::new();
    let mut kv_n = 0u64;
    // general.architecture = "llama" (STRING = 8)
    write_str(&mut kv, "general.architecture");
    kv.extend_from_slice(&8u32.to_le_bytes());
    write_str(&mut kv, "llama");
    kv_n += 1;
    // llama.block_count as U64 (10)
    write_str(&mut kv, "llama.block_count");
    kv.extend_from_slice(&10u32.to_le_bytes());
    kv.extend_from_slice(&2u64.to_le_bytes());
    kv_n += 1;
    // U8 (0), I8 (1), U16 (2), I16 (3), I64 (11), F64 (12), BOOL (7)
    for (name, ty, bytes) in [
        ("test.u8", 0u32, vec![7u8]),
        ("test.i8", 1u32, vec![0xffu8]),
        ("test.u16", 2u32, 42u16.to_le_bytes().to_vec()),
        ("test.i16", 3u32, (-5i16).to_le_bytes().to_vec()),
        ("test.bool", 7u32, vec![1u8]),
        ("test.i64", 11u32, (-99i64).to_le_bytes().to_vec()),
        ("test.f64", 12u32, 3.5f64.to_le_bytes().to_vec()),
    ] {
        write_str(&mut kv, name);
        kv.extend_from_slice(&ty.to_le_bytes());
        kv.extend_from_slice(&bytes);
        kv_n += 1;
    }
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    buf.extend_from_slice(&kv_n.to_le_bytes());
    buf.extend_from_slice(&kv);
    buf.resize((buf.len() + 31) & !31, 0);
    let ok_path = dir.path().join("scalars.gguf");
    std::fs::write(&ok_path, &buf).unwrap();
    let idx = GgufIndex::open(&ok_path).unwrap();
    assert_eq!(idx.meta.n_layers, 2);

    // 2. U64 block_count exceeding u32::MAX fails cleanly.
    let mut bad_bc = Vec::new();
    let mut kv2 = Vec::new();
    write_str(&mut kv2, "general.architecture");
    kv2.extend_from_slice(&8u32.to_le_bytes());
    write_str(&mut kv2, "llama");
    write_str(&mut kv2, "llama.block_count");
    kv2.extend_from_slice(&10u32.to_le_bytes());
    kv2.extend_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
    bad_bc.extend_from_slice(b"GGUF");
    bad_bc.extend_from_slice(&3u32.to_le_bytes());
    bad_bc.extend_from_slice(&0u64.to_le_bytes());
    bad_bc.extend_from_slice(&2u64.to_le_bytes());
    bad_bc.extend_from_slice(&kv2);
    let bad_bc_path = dir.path().join("bad_bc.gguf");
    std::fs::write(&bad_bc_path, &bad_bc).unwrap();
    // U64 block_count > u32::MAX returns None from as_u32() and falls back to tensor inference (0 layers).
    assert_eq!(GgufIndex::open(&bad_bc_path).unwrap().meta.n_layers, 0);

    // 3. Array length exceeding MAX_GGUF_ARRAY_LEN fails cleanly.
    let mut bad_arr = Vec::new();
    let mut kv3 = Vec::new();
    write_str(&mut kv3, "test.huge_arr");
    kv3.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    kv3.extend_from_slice(&0u32.to_le_bytes()); // U8
    kv3.extend_from_slice(&(100_000_000u64).to_le_bytes());
    bad_arr.extend_from_slice(b"GGUF");
    bad_arr.extend_from_slice(&3u32.to_le_bytes());
    bad_arr.extend_from_slice(&0u64.to_le_bytes());
    bad_arr.extend_from_slice(&1u64.to_le_bytes());
    bad_arr.extend_from_slice(&kv3);
    let bad_arr_path = dir.path().join("bad_arr.gguf");
    std::fs::write(&bad_arr_path, &bad_arr).unwrap();
    assert!(matches!(
        GgufIndex::open(&bad_arr_path),
        Err(GgufIoError::InvalidFormat(_))
    ));

    // 4. Truncated array (skip past EOF) fails cleanly.
    let mut eof_arr = Vec::new();
    let mut kv4 = Vec::new();
    write_str(&mut kv4, "test.eof_arr");
    kv4.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    kv4.extend_from_slice(&10u32.to_le_bytes()); // U64 (8 bytes each)
    kv4.extend_from_slice(&500u64.to_le_bytes());
    eof_arr.extend_from_slice(b"GGUF");
    eof_arr.extend_from_slice(&3u32.to_le_bytes());
    eof_arr.extend_from_slice(&0u64.to_le_bytes());
    eof_arr.extend_from_slice(&1u64.to_le_bytes());
    eof_arr.extend_from_slice(&kv4);
    let eof_arr_path = dir.path().join("eof_arr.gguf");
    std::fs::write(&eof_arr_path, &eof_arr).unwrap();
    assert!(matches!(
        GgufIndex::open(&eof_arr_path),
        Err(GgufIoError::InvalidFormat(_))
    ));

    // 5. check_gguf_draft_compatible error paths (missing file + corrupt header).
    let missing = dir.path().join("does_not_exist.gguf");
    assert!(gguf_direct_io::check_gguf_draft_compatible(&missing, &ok_path).is_err());
    let corrupt = dir.path().join("corrupt.gguf");
    std::fs::write(&corrupt, b"NOT_GGUF_DATA").unwrap();
    assert!(gguf_direct_io::check_gguf_draft_compatible(&corrupt, &ok_path).is_err());
}

#[test]
fn warm_expert_profile_skips_oob_and_oversized_entries() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("warm_bounds.gguf");
    let mut buf = Vec::new();
    let mut kv = Vec::new();
    let write_str = |out: &mut Vec<u8>, s: &str| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    write_str(&mut kv, "general.architecture");
    kv.extend_from_slice(&8u32.to_le_bytes());
    write_str(&mut kv, "llama");
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&kv);
    buf.resize(8192, 0x5A);
    std::fs::write(&model_path, &buf).unwrap();

    let idx = GgufIndex::open(&model_path).unwrap();
    let pool = LayerIoPool::new_with_options(idx, false, None, Some(4096)).unwrap();

    let profile_path = dir.path().join("profile.lsep");
    let entries = vec![
        // Valid entry within both shard_len (8192) and budget/2 (2048)
        gguf_direct_io::expert_profile::HotSliceEntry {
            shard_idx: 0,
            file_offset: 0,
            len: 512,
        },
        // Oversized (> budget/2 = 2048)
        gguf_direct_io::expert_profile::HotSliceEntry {
            shard_idx: 0,
            file_offset: 0,
            len: 4096,
        },
        // Out of shard bounds
        gguf_direct_io::expert_profile::HotSliceEntry {
            shard_idx: 0,
            file_offset: 8000,
            len: 512,
        },
        // Arithmetic overflow
        gguf_direct_io::expert_profile::HotSliceEntry {
            shard_idx: 0,
            file_offset: u64::MAX - 8,
            len: 64,
        },
        // Invalid shard index
        gguf_direct_io::expert_profile::HotSliceEntry {
            shard_idx: 9,
            file_offset: 0,
            len: 64,
        },
    ];
    gguf_direct_io::expert_profile::write_expert_profile(&profile_path, &entries).unwrap();
    assert_eq!(pool.warm_expert_profile(&profile_path).unwrap(), 1);

    // Zero-budget cache skips warming immediately.
    let idx0 = GgufIndex::open(&model_path).unwrap();
    let pool0 = LayerIoPool::new_with_options(idx0, false, None, Some(0)).unwrap();
    assert_eq!(pool0.warm_expert_profile(&profile_path).unwrap(), 0);

    // `probe_dir_storage_mib_per_s` reuses an existing >=16 MiB file in a nested subdirectory.
    let nested_dir = dir.path().join("sub");
    std::fs::create_dir_all(&nested_dir).unwrap();
    let big_file = nested_dir.join("existing.bin");
    std::fs::write(&big_file, vec![0x33u8; 16 * 1024 * 1024]).unwrap();
    assert!(gguf_direct_io::probe_dir_storage_mib_per_s(dir.path()).is_some());
}

#[test]
fn load_slices_into_rejects_oob_and_overflow_before_cache_lookup() {
    use gguf_direct_io::{ModelMeta, TensorLocation};
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let shard = dir.path().join("oob.gguf");
    std::fs::write(&shard, vec![0xAAu8; 4096]).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "small".to_string(),
        TensorLocation {
            name: "small".to_string(),
            shard_idx: 0,
            abs_offset: 64,
            ggml_type: 0, // F32: 16 elems = 64 bytes
            dims: vec![16],
        },
    );
    tensors.insert(
        "near_end".to_string(),
        TensorLocation {
            name: "near_end".to_string(),
            shard_idx: 0,
            abs_offset: 4032,
            ggml_type: 999, // unknown type -> encoded_byte_len() == None, falls back to shard check
            dims: vec![16],
        },
    );

    let idx = GgufIndex {
        shards: vec![shard],
        shard_sizes: vec![4096],
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: None,
            n_layers: 1,
            context_length: None,
            total_file_bytes: 4096,
        },
        tensors,
    };
    let pool = LayerIoPool::new_with_options(idx, false, None, Some(65536)).unwrap();
    let mut dst = vec![0u8; 256];

    // 1. Relative offset overflow.
    assert!(matches!(
        pool.load_slices_into(&[("small", u64::MAX, 0, 16, 0)], &mut dst),
        Err(GgufIoError::OutOfBounds(_))
    ));
    // 2. Exceeds tensor encoded byte length (64 bytes).
    assert!(matches!(
        pool.load_slices_into(&[("small", 32, 0, 64, 0)], &mut dst),
        Err(GgufIoError::OutOfBounds(_))
    ));
    // 3. Exceeds shard size (4032 + 128 > 4096).
    assert!(matches!(
        pool.load_slices_into(&[("near_end", 0, 0, 128, 0)], &mut dst),
        Err(GgufIoError::OutOfBounds(_))
    ));

    // 4. Absolute offset overflow (`abs_offset + rel_end` overflows u64).
    let mut overflow_tensors = HashMap::new();
    overflow_tensors.insert(
        "huge_off".to_string(),
        TensorLocation {
            name: "huge_off".to_string(),
            shard_idx: 0,
            abs_offset: u64::MAX - 8,
            ggml_type: 999,
            dims: vec![16],
        },
    );
    let idx_ov = GgufIndex {
        shards: vec![dir.path().join("oob.gguf")],
        shard_sizes: vec![4096],
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: None,
            n_layers: 1,
            context_length: None,
            total_file_bytes: 4096,
        },
        tensors: overflow_tensors,
    };
    let pool_ov = LayerIoPool::new_with_options(idx_ov, false, None, Some(65536)).unwrap();
    assert!(matches!(
        pool_ov.load_slices_into(&[("huge_off", 0, 0, 32, 0)], &mut dst),
        Err(GgufIoError::OutOfBounds(_))
    ));

    // 5. `save_expert_profile` on empty cache (returns 0), then after populating cache (returns 1).
    let empty_prof = dir.path().join("empty.lsep");
    assert_eq!(pool.save_expert_profile(&empty_prof).unwrap(), 0);
    assert!(
        pool.load_slices_into(&[("small", 0, 0, 64, 0)], &mut dst)
            .is_ok()
    );
    let pop_prof = dir.path().join("pop.lsep");
    assert_eq!(pool.save_expert_profile(&pop_prof).unwrap(), 1);

    // 6. `load_layer_direct_into` with `prefer_direct_io = false` exercises the buffered `pread_all`
    //    fallback, and a truncated shard triggers `pread_all` unexpected-EOF cleanly.
    let mut aligned_buf = gguf_direct_io::AlignedBuf::new(8192).unwrap();
    let pin_buf = aligned_buf.as_mut_slice(8192);
    let mut out_slices = vec![
        gguf_direct_io::DirectSpanSlice {
            src_offset: 0,
            dst_offset: 0,
            len: 0,
            tensor_idx: 0,
        };
        2
    ];
    let n = pool
        .load_layer_direct_into(0, Some(&[("small", 0, 64)]), pin_buf, &mut out_slices)
        .unwrap();
    assert_eq!(n, 1);

    let trunc_shard = dir.path().join("trunc.gguf");
    std::fs::write(&trunc_shard, vec![0xBBu8; 16]).unwrap();
    let mut trunc_tensors = HashMap::new();
    trunc_tensors.insert(
        "t".to_string(),
        TensorLocation {
            name: "t".to_string(),
            shard_idx: 0,
            abs_offset: 0,
            ggml_type: 0,
            dims: vec![1024], // 4096 bytes, larger than 16-byte file on disk
        },
    );
    let idx_trunc = GgufIndex {
        shards: vec![trunc_shard],
        shard_sizes: vec![4096],
        meta: ModelMeta {
            architecture: "llama".to_string(),
            name: None,
            n_layers: 1,
            context_length: None,
            total_file_bytes: 4096,
        },
        tensors: trunc_tensors,
    };
    let pool_trunc = LayerIoPool::new(idx_trunc, false).unwrap();
    assert!(
        pool_trunc
            .load_layer_direct_into(0, Some(&[("t", 0, 4096)]), pin_buf, &mut out_slices)
            .is_err()
    );

    // 7. `write_expert_profile` fails cleanly when target path is a directory (removes temp file).
    let as_dir = dir.path().join("is_a_dir");
    std::fs::create_dir_all(&as_dir).unwrap();
    assert!(gguf_direct_io::expert_profile::write_expert_profile(&as_dir, &[]).is_err());
}

#[test]
fn fuzz_read_expert_profile_never_panics_or_ooms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fuzz.lsep");
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_u64 = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for case in 0..256u32 {
        let len = (next_u64() as usize) % 256;
        let mut raw = Vec::with_capacity(len + 12);
        if case % 2 == 0 {
            // Half the cases start with valid magic + version to exercise count/payload parsing.
            raw.extend_from_slice(b"LSEP");
            raw.extend_from_slice(&1u32.to_le_bytes());
        }
        for _ in 0..len {
            raw.push((next_u64() & 0xff) as u8);
        }
        std::fs::write(&path, &raw).unwrap();
        let _ = gguf_direct_io::expert_profile::read_expert_profile(&path);
    }
}
