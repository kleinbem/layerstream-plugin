use super::*;

#[cfg(test)]
mod adaptive_cache_tests {
    use super::*;

    #[test]
    fn auto_expert_cache_budget_scales_by_memory_tier() {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;
        // 8 GiB edge host with 4 GiB available: reserve 1.5 GiB, 60% of 2.5 GiB = 1.5 GiB
        assert_eq!(
            auto_expert_cache_budget_for_meminfo(8 * GIB, 4 * GIB),
            (1536 * MIB) as usize
        );
        // 8 GiB edge host with 6 GiB available: reserve 1.5 GiB, 60% of 4.5 GiB = 2.7 GiB -> capped at 2 GiB
        assert_eq!(
            auto_expert_cache_budget_for_meminfo(8 * GIB, 6 * GIB),
            (2048 * MIB) as usize
        );
        // Edge host with tight RAM (1 GiB available < 1.5 GiB reserve) -> 0
        assert_eq!(auto_expert_cache_budget_for_meminfo(8 * GIB, GIB), 0);
        // 24 GiB standard host with 16 GiB available -> capped at 4 GiB
        assert_eq!(
            auto_expert_cache_budget_for_meminfo(24 * GIB, 16 * GIB),
            (4096 * MIB) as usize
        );
        // 64 GiB high-RAM APU with 48 GiB available -> (48 - 4) / 2 = 22 GiB
        assert_eq!(
            auto_expert_cache_budget_for_meminfo(64 * GIB, 48 * GIB),
            (22 * GIB) as usize
        );
        // 128 GiB workstation with 100 GiB available -> capped at 32 GiB
        assert_eq!(
            auto_expert_cache_budget_for_meminfo(128 * GIB, 100 * GIB),
            (32 * GIB) as usize
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_slice_cache_respects_budget_and_clears() {
        let mut cache = ExpertSliceCache::new(Some(12));
        let k1 = SliceKey {
            shard_idx: 0,
            file_offset: 0,
            len: 4,
        };
        let k2 = SliceKey {
            shard_idx: 0,
            file_offset: 4,
            len: 4,
        };
        let k3 = SliceKey {
            shard_idx: 0,
            file_offset: 8,
            len: 4,
        };
        let k4 = SliceKey {
            shard_idx: 0,
            file_offset: 12,
            len: 4,
        };

        cache.insert(k1, b"AAAA");
        cache.insert(k2, b"BBBB");
        cache.insert(k3, b"CCCC");
        assert!(cache.used_bytes() <= 12);
        assert_eq!(cache.get_and_touch(k1), Some(b"AAAA".as_slice()));

        // Oversized entries (> budget / 2) are rejected.
        let k_big = SliceKey {
            shard_idx: 0,
            file_offset: 64,
            len: 8,
        };
        cache.insert(k_big, b"12345678");
        assert_eq!(cache.get_and_touch(k_big), None);

        // Inserting k4 stays strictly within byte budget while keeping hot k1 resident.
        cache.insert(k4, b"DDDD");
        assert!(cache.used_bytes() <= 12);
        assert_eq!(cache.get_and_touch(k1), Some(b"AAAA".as_slice()));

        cache.clear();
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get_and_touch(k1), None);
    }

    #[test]
    fn expert_slice_cache_resists_cyclic_scan_thrashing() {
        // Capacity 80 bytes (20 items) scanning 30 items repeatedly: pure LRU gets 0% hits,
        // S3-FIFO keeps hot items resident across sweeps.
        let mut cache = ExpertSliceCache::new(Some(80));
        let keys: Vec<SliceKey> = (0..30)
            .map(|i| SliceKey {
                shard_idx: 0,
                file_offset: i * 4,
                len: 4,
            })
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for sweep in 0..8 {
            for &k in &keys {
                if sweep >= 2 {
                    total += 1;
                }
                if cache.get_and_touch(k).is_some() {
                    if sweep >= 2 {
                        hits += 1;
                    }
                } else {
                    cache.insert(k, b"DATA");
                }
            }
        }
        assert!(
            hits * 100 / total >= 40,
            "expected >=40% hit rate under cyclic scan, got {hits}/{total}"
        );
    }

    #[test]
    fn expert_slice_cache_disabled_at_zero_budget() {
        let mut cache = ExpertSliceCache::new(Some(0));
        let k = SliceKey {
            shard_idx: 0,
            file_offset: 0,
            len: 4,
        };
        cache.insert(k, b"TEST");
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get_and_touch(k), None);
    }

    #[test]
    fn load_slices_into_reads_single_and_coalesced_spans_parallel_and_serial() {
        use crate::gguf_index::{ModelMeta, TensorLocation};
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("synthetic.gguf");
        // 512 KiB synthetic shard with unique byte values per 4 KiB page.
        let mut payload = vec![0u8; 512 * 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i / 251) ^ (i % 251)) as u8;
        }
        std::fs::File::create(&shard_path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let mut tensors = HashMap::new();
        // t0 and t1 are 8 KiB apart (< 64 KiB merge gap -> coalesced multi-slice span).
        tensors.insert(
            "t0".to_string(),
            TensorLocation {
                name: "t0".to_string(),
                shard_idx: 0,
                abs_offset: 4096,
                ggml_type: 0,
                dims: vec![8192],
            },
        );
        tensors.insert(
            "t1".to_string(),
            TensorLocation {
                name: "t1".to_string(),
                shard_idx: 0,
                abs_offset: 16384,
                ggml_type: 0,
                dims: vec![8192],
            },
        );
        // t2 and t3 are > 128 KiB apart (-> single-slice direct-to-dst fast path).
        tensors.insert(
            "t2".to_string(),
            TensorLocation {
                name: "t2".to_string(),
                shard_idx: 0,
                abs_offset: 160 * 1024,
                ggml_type: 0,
                dims: vec![12288],
            },
        );
        tensors.insert(
            "t3".to_string(),
            TensorLocation {
                name: "t3".to_string(),
                shard_idx: 0,
                abs_offset: 384 * 1024,
                ggml_type: 0,
                dims: vec![12288],
            },
        );

        let index = GgufIndex {
            meta: ModelMeta {
                architecture: "qwen3moe".to_string(),
                name: None,
                n_layers: 1,
                context_length: Some(2048),
                total_file_bytes: payload.len() as u64,
            },
            shards: vec![shard_path],
            shard_sizes: vec![payload.len() as u64],
            tensors,
        };

        let mut pool = LayerIoPool::new(index, false).unwrap();
        pool.sparse_direct_io = false;

        let reqs = [
            ("t0", 128u64, 0usize, 4096usize, 0u32),
            ("t1", 256u64, 4096usize, 4096usize, 1u32),
            ("t2", 512u64, 8192usize, 8192usize, 2u32),
            ("t3", 0u64, 16384usize, 8192usize, 3u32),
        ];

        for threads in [1usize, 4usize] {
            pool.moe_io_threads = threads;
            pool.clear_expert_cache();
            let mut dst = vec![0u8; 24576];
            pool.load_slices_into(&reqs, &mut dst).unwrap();

            assert_eq!(&dst[0..4096], &payload[4096 + 128..4096 + 128 + 4096]);
            assert_eq!(&dst[4096..8192], &payload[16384 + 256..16384 + 256 + 4096]);
            assert_eq!(
                &dst[8192..16384],
                &payload[160 * 1024 + 512..160 * 1024 + 512 + 8192]
            );
            assert_eq!(&dst[16384..24576], &payload[384 * 1024..384 * 1024 + 8192]);

            // Second read with warm expert cache serves all 4 slices from RAM.
            let before = pool.stats();
            let mut dst2 = vec![0u8; 24576];
            pool.load_slices_into(&reqs, &mut dst2).unwrap();
            let after = pool.stats();
            assert_eq!(dst2, dst);
            assert_eq!(after.cache_hits - before.cache_hits, 4);
            assert_eq!(after.bytes_read, before.bytes_read);
        }

        // Overlapping destination ranges must still fail closed.
        let bad = [
            ("t0", 0u64, 0usize, 4096usize, 0u32),
            ("t1", 0u64, 2048usize, 4096usize, 1u32),
        ];
        let mut dst = vec![0u8; 8192];
        assert!(pool.load_slices_into(&bad, &mut dst).is_err());
    }

    #[test]
    fn scoped_worker_pool_runs_parallel_and_overflow_batches() {
        let pool = ScopedWorkerPool::new(4);
        for n_tasks in [0usize, 1, 2, 4, 7] {
            let hits: Vec<AtomicU64> = (0..n_tasks).map(|_| AtomicU64::new(0)).collect();
            pool.run(n_tasks, |idx| {
                hits[idx].fetch_add(1, Ordering::Relaxed);
            });
            for h in &hits {
                assert_eq!(h.load(Ordering::Relaxed), 1);
            }
        }
    }

    #[test]
    fn load_layer_into_cached_layer_and_storage_probe_work() {
        use crate::gguf_index::{ModelMeta, TensorLocation};
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("layer_test.gguf");
        let mut payload = vec![0u8; 64 * 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 13) ^ (i >> 5)) as u8;
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
                name: Some("test".to_string()),
                n_layers: 1,
                context_length: Some(512),
                total_file_bytes: payload.len() as u64,
            },
            shards: vec![shard_path.clone()],
            shard_sizes: vec![payload.len() as u64],
            tensors,
        };
        let pool = LayerIoPool::new(index, false).unwrap();

        // First call plans & caches layer 0 spans.
        let mut dst = vec![0u8; 8192];
        assert!(!pool.load_cached_layer_into(0, &mut dst).unwrap());
        pool.load_layer_tensors_into(
            0,
            &[("blk.0.w0", 0, 4096), ("blk.0.w1", 4096, 4096)],
            &mut dst,
        )
        .unwrap();
        assert_eq!(&dst[..4096], &payload[4096..8192]);
        assert_eq!(&dst[4096..8192], &payload[12288..16384]);

        // Uncached one-shot load (`layer_idx = -1`) also matches.
        let mut dst_oneshot = vec![0u8; 8192];
        pool.load_tensors_into(
            &[("blk.0.w0", 0, 4096), ("blk.0.w1", 4096, 4096)],
            &mut dst_oneshot,
        )
        .unwrap();
        assert_eq!(dst_oneshot, dst);

        // Second call hits cached layer spans directly.
        let mut dst2 = vec![0u8; 8192];
        assert!(pool.load_cached_layer_into(0, &mut dst2).unwrap());
        assert_eq!(dst2, dst);
        assert!(pool.stats().bytes_read > 0);
        assert!(pool.stats().span_reads >= 2);

        // Undersized destination buffer fails with typed OutOfBounds.
        let mut too_small = vec![0u8; 4096];
        let err = pool.load_cached_layer_into(0, &mut too_small).unwrap_err();
        assert!(matches!(err, GgufIoError::OutOfBounds(_)));

        // Storage probes succeed on real files/directories.
        let (mib_s, _direct) = probe_storage_mib_per_s(&shard_path, false).unwrap();
        assert!(mib_s >= 0.0);
        assert!(probe_dir_storage_mib_per_s(dir.path()).is_some());

        // `record_io_sample` triggers one-time slow-storage warning at >=128 MiB and <150 MiB/s.
        assert!(!pool.slow_storage_warned.load(Ordering::Relaxed));
        pool.record_io_sample(128 * 1024 * 1024, 2_000_000); // 64 MiB/s
        assert!(pool.slow_storage_warned.load(Ordering::Relaxed));
        pool.record_io_sample(128 * 1024 * 1024, 2_000_000); // idempotent second call
    }

    #[test]
    fn covers_cache_auto_budget_warm_profile_missing_shard_and_probe_errors() {
        // 1. Auto-budget ExpertSliceCache exercises /proc/meminfo lookup + zero-budget ensure_cache.
        let mut auto_cache = ExpertSliceCache::new(None);
        let _ = auto_cache.ensure_budget();
        let mut zero_cache = ExpertSliceCache::new(Some(0));
        assert!(zero_cache.ensure_cache().is_none());
        assert!(zero_cache.export_hot_entries().is_empty());
        zero_cache.insert(
            SliceKey {
                shard_idx: 0,
                file_offset: 0,
                len: 4,
            },
            b"abcd",
        );
        assert_eq!(zero_cache.used_bytes(), 0);

        // 2. Round-trip save_expert_profile + warm_expert_profile on a synthetic shard.
        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("warm_test.bin");
        let payload: Vec<u8> = (0..(16 * 1024)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&shard_path, &payload).unwrap();

        let make_index = |path: &std::path::Path| {
            let mut tensors = HashMap::new();
            tensors.insert(
                "blk.0.w0".to_string(),
                crate::gguf_index::TensorLocation {
                    name: "blk.0.w0".to_string(),
                    shard_idx: 0,
                    abs_offset: 4096,
                    ggml_type: 0,
                    dims: vec![1024],
                },
            );
            GgufIndex {
                meta: crate::gguf_index::ModelMeta {
                    architecture: "llama".to_string(),
                    name: Some("test".to_string()),
                    n_layers: 1,
                    context_length: Some(512),
                    total_file_bytes: payload.len() as u64,
                },
                shards: vec![path.to_path_buf()],
                shard_sizes: vec![payload.len() as u64],
                tensors,
            }
        };

        let pool =
            LayerIoPool::new_with_options(make_index(&shard_path), false, Some(1), Some(64 * 1024))
                .unwrap();
        let mut dst = vec![0u8; 4096];
        pool.load_slices_into(&[("blk.0.w0", 0, 0, 4096, 0)], &mut dst)
            .unwrap();
        assert_eq!(dst, &payload[4096..8192]);

        let prof_path = dir.path().join("hot.lsep");
        let saved = pool.save_expert_profile(&prof_path).unwrap();
        assert_eq!(saved, 1);

        let pool2 =
            LayerIoPool::new_with_options(make_index(&shard_path), false, Some(1), Some(64 * 1024))
                .unwrap();
        let warmed = pool2.warm_expert_profile(&prof_path).unwrap();
        assert_eq!(warmed, 1);
        // Zero-budget pool skips warming cleanly.
        let pool_zero =
            LayerIoPool::new_with_options(make_index(&shard_path), false, Some(1), Some(0))
                .unwrap();
        assert_eq!(pool_zero.warm_expert_profile(&prof_path).unwrap(), 0);

        // 3. Missing shard file after index open triggers LayerIoPool::new Io error.
        std::fs::remove_file(&shard_path).unwrap();
        assert!(LayerIoPool::new(make_index(&shard_path), false).is_err());

        // 4. Storage probe on missing directory returns None, missing file returns Err,
        //    and nested subdirectory with >=16 MiB file hits recursive discovery.
        let missing_dir = dir.path().join("does-not-exist");
        assert!(probe_dir_storage_mib_per_s(&missing_dir).is_none());
        assert!(probe_storage_mib_per_s(&missing_dir.join("x.gguf"), true).is_err());
        let nested_root = dir.path().join("nested_probe");
        let nested_sub = nested_root.join("sub");
        std::fs::create_dir_all(&nested_sub).unwrap();
        let big_file = std::fs::File::create(nested_sub.join("big.bin")).unwrap();
        big_file.set_len(16 * 1024 * 1024).unwrap();
        drop(big_file);
        assert!(probe_dir_storage_mib_per_s(&nested_root).is_some());

        // 5. write_expert_profile rejects entry counts exceeding MAX_PROFILE_ENTRIES.
        let too_many: Vec<crate::expert_profile::HotSliceEntry> = (0..=1_000_000u64)
            .map(|i| crate::expert_profile::HotSliceEntry {
                shard_idx: 0,
                file_offset: i * 64,
                len: 64,
            })
            .collect();
        let over_err =
            crate::expert_profile::write_expert_profile(&dir.path().join("over.lsep"), &too_many)
                .unwrap_err();
        assert_eq!(over_err.kind(), std::io::ErrorKind::InvalidInput);

        // 6. Parallel sparse read error propagation (4 spans spaced 4 MiB apart so they do not coalesce).
        let short_path = dir.path().join("short.bin");
        std::fs::write(&short_path, [0u8; 4096]).unwrap();
        let mut short_tensors = HashMap::new();
        for i in 0..4u64 {
            short_tensors.insert(
                format!("blk.0.e{i}"),
                crate::gguf_index::TensorLocation {
                    name: format!("blk.0.e{i}"),
                    shard_idx: 0,
                    abs_offset: (i + 1) * 4 * 1024 * 1024, // spaced 4 MiB apart, past EOF
                    ggml_type: 0,
                    dims: vec![1024], // 4096 bytes
                },
            );
        }
        let short_idx = GgufIndex {
            meta: crate::gguf_index::ModelMeta {
                architecture: "llama".to_string(),
                name: Some("short".to_string()),
                n_layers: 1,
                context_length: Some(512),
                total_file_bytes: 32 * 1024 * 1024,
            },
            shards: vec![short_path],
            shard_sizes: vec![32 * 1024 * 1024], // claim larger than actual file to trigger EOF in workers
            tensors: short_tensors,
        };
        let mut short_pool =
            LayerIoPool::new_with_options(short_idx, false, Some(2), Some(0)).unwrap();
        short_pool.moe_io_threads = 2;
        short_pool.profile_enabled = true;
        short_pool.bytes_read.store(4096, Ordering::Relaxed);
        let mut dst_err = vec![0u8; 4 * 4096];
        let slices_err = [
            ("blk.0.e0", 0u64, 0usize, 4096usize, 0u32),
            ("blk.0.e1", 0u64, 4096usize, 4096usize, 0u32),
            ("blk.0.e2", 0u64, 8192usize, 4096usize, 0u32),
            ("blk.0.e3", 0u64, 12288usize, 4096usize, 0u32),
        ];
        assert!(
            short_pool
                .load_slices_into(&slices_err, &mut dst_err)
                .is_err()
        );

        // 7. io_uring bad-fd CQE error path (EBADF) returns false cleanly on Linux.
        #[cfg(target_os = "linux")]
        if let Some(mut reader) = crate::uring::UringReader::try_new() {
            let mut buf = vec![0u8; 4096];
            assert!(!reader.read_aligned_span(-1, &mut buf, 0, 4096));
        }

        // 8. O_DIRECT fallback warning when opening a shard on tmpfs (/dev/shm) with prefer_direct_io=true.
        #[cfg(target_os = "linux")]
        if std::path::Path::new("/dev/shm").is_dir() {
            let shm_path = std::path::PathBuf::from(format!(
                "/dev/shm/.layerstream-direct-test-{}.bin",
                std::process::id()
            ));
            if std::fs::write(&shm_path, [0u8; 4096]).is_ok() {
                let _ = LayerIoPool::new(make_index(&shm_path), true);
                let _ = std::fs::remove_file(&shm_path);
            }
        }

        // Direct call to read_sparse_span_into_disjoint_ptr with undersized dst hits OutOfBounds.
        let mut scratch = AlignedBuf::new(4096).unwrap();
        let bad_span = CoalescedSpan {
            shard_idx: 0,
            aligned_offset: 0,
            aligned_len: 4096,
            valid_len: 4096,
            slices: vec![crate::gguf_index::SpanSlice {
                span_offset: 0,
                dst_offset: 16,
                len: 32,
                tensor_idx: 0,
            }],
        };
        let mut tiny = [0u8; 16];
        assert!(matches!(
            short_pool.read_sparse_span_into_disjoint_ptr(
                &bad_span,
                &mut scratch,
                tiny.as_mut_ptr(),
                tiny.len(),
            ),
            Err(GgufIoError::OutOfBounds(_))
        ));
        // Drop `short_pool` with `profile_enabled = true`, `bytes_read > 0`, `hits = misses = 0`.
        drop(short_pool);
    }
}
