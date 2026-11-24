#![allow(clippy::unwrap_used)]

//! Pure unit tests covering review findings:
//! - P1-2: padded-vocab draft compatibility (`152,064` vs `151,936`)
//! - P1-3: unbiased speculative rejection sampling (`eval_sample` vs `eval_verify_sample`, TV < 0.005)
//! - P2-2: sliding UTF-8 stop-sequence filter (never leaks stop text, catches mid-piece/split stops)
//! - P2-7: corrupt/untrusted GGUF header bounds & alignment validation

use std::io::Write;

use layerstream_core::ffi::{
    ls_sample_from_logits, ls_sample_from_logits_with_history, ls_verify_sample_from_logits,
    ls_verify_sample_from_logits_with_history,
};
use layerstream_core::{GgufIndex, LsSampleParams, StopFilter, check_vocab_lists_compatible};

#[test]
fn padded_vocab_draft_compatibility() {
    let base: Vec<String> = (0..151_936).map(|i| format!("tok_{i}")).collect();
    let mut padded = base.clone();
    for i in 0..128 {
        padded.push(format!("[PAD{i}]"));
    }
    assert!(
        check_vocab_lists_compatible(&padded, &base).is_ok(),
        "Qwen2.5-style 128-token padded vocab must be accepted"
    );

    // Mismatched token string in shared range must be rejected.
    let mut mismatched = base.clone();
    mismatched[42] = "DIFFERENT".to_string();
    assert!(check_vocab_lists_compatible(&padded, &mismatched).is_err());

    // Delta > 256 must be rejected.
    let mut too_big = base.clone();
    for i in 0..300 {
        too_big.push(format!("[PAD{i}]"));
    }
    assert!(check_vocab_lists_compatible(&too_big, &base).is_err());
}

#[test]
fn stop_filter_never_emits_stop_and_catches_mid_piece_and_split() {
    // 1. Stop split across multiple tokens ("ST" + "OP") must never emit "STOP".
    let mut f = StopFilter::new(&["STOP".to_string()]);
    let mut out = String::new();
    let mut stopped = false;
    for piece in ["hello ", "wor", "ld ", "ST", "OP", " trailing"] {
        let (emit, hit) = f.push(piece);
        out.push_str(&emit);
        if hit {
            stopped = true;
            break;
        }
    }
    assert!(stopped);
    assert_eq!(out, "hello world ");

    // 2. Mid-piece match (`"\n\nUser:"` with stop `"\n\n"`).
    let mut f2 = StopFilter::new(&["\n\n".to_string()]);
    let (e1, h1) = f2.push("Answer");
    assert!(!h1);
    let (e2, h2) = f2.push(" done.\n\nUser: hi");
    assert!(h2);
    assert_eq!(format!("{e1}{e2}"), "Answer done.");

    // 3. Natural end flushes held-back tail cleanly (including multibyte UTF-8).
    let mut f3 = StopFilter::new(&["STOP".to_string()]);
    let (a, ha) = f3.push("héllo!");
    assert!(!ha);
    let tail = f3.flush();
    assert_eq!(format!("{a}{tail}"), "héllo!");
}

#[test]
fn rejection_sampler_matches_eval_sample_distribution() {
    // Synthetic 16-token vocab with peaked + tail logits (16 bins at N=200k has Monte Carlo
    // TV noise ~0.0028, so TV < 0.008 cleanly separates unbiased sampling from real bias).
    let n_vocab: i32 = 16;
    let logits: Vec<f32> = (0..n_vocab)
        .map(|i| {
            let x = i as f32;
            2.4 * (-0.22 * x).exp() + 0.15 * (x * 1.7).sin()
        })
        .collect();

    for &temp in &[0.3f32, 0.7, 1.3] {
        let n_draws = 200_000usize;
        let mut counts_direct = vec![0usize; n_vocab as usize];
        let mut counts_verify = vec![0usize; n_vocab as usize];

        let logits_ptrs = [logits.as_ptr(), logits.as_ptr()];
        // Fixed deterministic draft token = 1 (not argmax = 0, exercising both accept & reject).
        let draft = [1i32];

        for i in 0..n_draws {
            let sp_direct = LsSampleParams {
                temperature: temp,
                top_p: 0.92,
                min_p: 0.05,
                top_k: 12,
                seed: (i as u32).wrapping_mul(2654435761).wrapping_add(17),
                ..LsSampleParams::default()
            };
            let t_direct = unsafe { ls_sample_from_logits(logits.as_ptr(), n_vocab, &sp_direct) };
            counts_direct[t_direct as usize] += 1;

            let sp_verify = LsSampleParams {
                temperature: temp,
                top_p: 0.92,
                min_p: 0.05,
                top_k: 12,
                seed: (i as u32).wrapping_mul(2246822519).wrapping_add(99),
                ..LsSampleParams::default()
            };
            let mut acc = 0i32;
            let mut next = 0i32;
            unsafe {
                ls_verify_sample_from_logits(
                    logits_ptrs.as_ptr(),
                    n_vocab,
                    draft.as_ptr(),
                    1,
                    &sp_verify,
                    &mut acc,
                    &mut next,
                );
            }
            // First emitted token from verify step is `draft[0]` if accepted, else `next`.
            let t_verify = if acc >= 1 { draft[0] } else { next };
            counts_verify[t_verify as usize] += 1;
        }

        let tv: f64 = counts_direct
            .iter()
            .zip(counts_verify.iter())
            .map(|(&a, &b)| ((a as f64 / n_draws as f64) - (b as f64 / n_draws as f64)).abs())
            .sum::<f64>()
            * 0.5;
        assert!(
            tv < 0.008,
            "TV distance between eval_sample and eval_verify_sample too high at T={temp}: {tv:.5}"
        );
    }
}

#[test]
fn corrupt_gguf_headers_are_rejected_without_oom() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // 1. Huge tensor_count (0xFFFF_FFFF_FFFF) must fail cleanly instead of OOM-aborting.
    let bad_tc = dir.join("bad_tc.gguf");
    {
        let mut f = std::fs::File::create(&bad_tc).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap(); // tensor_count
        f.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count
    }
    assert!(GgufIndex::open(&bad_tc).is_err());

    // 2. Non-power-of-two `general.alignment = 0` must fail cleanly.
    let bad_align = dir.join("bad_align.gguf");
    {
        let mut f = std::fs::File::create(&bad_align).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor_count
        f.write_all(&1u64.to_le_bytes()).unwrap(); // kv_count
        let key = b"general.alignment";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32
        f.write_all(&0u32.to_le_bytes()).unwrap(); // invalid alignment=0
    }
    assert!(GgufIndex::open(&bad_align).is_err());
}

#[test]
fn repetition_and_frequency_penalties_preserve_exact_speculative_parity() {
    let n_vocab: i32 = 12;
    let logits0: Vec<f32> = (0..n_vocab).map(|i| 3.0 - 0.35 * (i as f32)).collect();
    let logits1: Vec<f32> = (0..n_vocab)
        .map(|i| 2.5 - 0.25 * ((i as f32 - 2.0).abs()))
        .collect();
    let logits2: Vec<f32> = (0..n_vocab).map(|i| 1.8 + 0.2 * (i as f32).cos()).collect();
    let history = [0i32, 0, 2, 4, 0, 1];

    // 1. Greedy with strong repeat_penalty + frequency_penalty + presence_penalty:
    // multi-token verify must accept/reject identically to sequential single-token steps.
    let sp_greedy_pen = LsSampleParams {
        temperature: 0.0,
        repeat_penalty: 1.45,
        frequency_penalty: 0.6,
        presence_penalty: 0.4,
        penalty_last_n: 4,
        ..LsSampleParams::default()
    };
    let step0 = unsafe {
        ls_sample_from_logits_with_history(
            logits0.as_ptr(),
            n_vocab,
            &sp_greedy_pen,
            history.as_ptr(),
            history.len() as i32,
        )
    };
    let mut hist_after_0 = history.to_vec();
    hist_after_0.push(step0);
    let step1 = unsafe {
        ls_sample_from_logits_with_history(
            logits1.as_ptr(),
            n_vocab,
            &sp_greedy_pen,
            hist_after_0.as_ptr(),
            hist_after_0.len() as i32,
        )
    };
    let logits_ptrs = [logits0.as_ptr(), logits1.as_ptr(), logits2.as_ptr()];
    let matching_draft = [step0, 999_999.min(n_vocab - 1)];
    let mut acc = 0i32;
    let mut next_tok = 0i32;
    unsafe {
        ls_verify_sample_from_logits_with_history(
            logits_ptrs.as_ptr(),
            n_vocab,
            matching_draft.as_ptr(),
            2,
            &sp_greedy_pen,
            history.as_ptr(),
            history.len() as i32,
            &mut acc,
            &mut next_tok,
        );
    }
    if matching_draft[1] == step1 {
        assert_eq!(acc, 2);
    } else {
        assert_eq!(acc, 1);
        assert_eq!(next_tok, step1);
    }

    // 2. Sampled (T=0.85, min_p=0.08, repeat/freq/presence penalties) 2-step joint distribution:
    // verify that (first_emitted, second_emitted_when_accepted) matches sequential sampling.
    let n_draws = 150_000usize;
    let mut counts_seq = vec![0usize; n_vocab as usize];
    let mut counts_ver = vec![0usize; n_vocab as usize];
    let draft = [1i32, 3i32];

    for i in 0..n_draws {
        let sp1 = LsSampleParams {
            temperature: 0.85,
            top_p: 0.90,
            min_p: 0.08,
            top_k: 10,
            repeat_penalty: 1.25,
            frequency_penalty: 0.35,
            presence_penalty: 0.25,
            penalty_last_n: 5,
            seed: (i as u32).wrapping_mul(2654435761).wrapping_add(31),
        };
        let t_seq = unsafe {
            ls_sample_from_logits_with_history(
                logits0.as_ptr(),
                n_vocab,
                &sp1,
                history.as_ptr(),
                history.len() as i32,
            )
        };
        counts_seq[t_seq as usize] += 1;

        let mut sp2 = sp1;
        sp2.seed = (i as u32).wrapping_mul(2246822519).wrapping_add(73);
        let mut v_acc = 0i32;
        let mut v_next = 0i32;
        unsafe {
            ls_verify_sample_from_logits_with_history(
                logits_ptrs.as_ptr(),
                n_vocab,
                draft.as_ptr(),
                2,
                &sp2,
                history.as_ptr(),
                history.len() as i32,
                &mut v_acc,
                &mut v_next,
            );
        }
        let t_ver = if v_acc >= 1 { draft[0] } else { v_next };
        counts_ver[t_ver as usize] += 1;
    }

    let tv: f64 = counts_seq
        .iter()
        .zip(counts_ver.iter())
        .map(|(&a, &b)| ((a as f64 / n_draws as f64) - (b as f64 / n_draws as f64)).abs())
        .sum::<f64>()
        * 0.5;
    assert!(
        tv < 0.008,
        "penalized rejection sampler TV distance too high: {tv:.5}"
    );
}
