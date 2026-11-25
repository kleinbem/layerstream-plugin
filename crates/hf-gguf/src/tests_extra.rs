use super::*;

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
