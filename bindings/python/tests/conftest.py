from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest


def write_synthetic_mixed_dense_moe_gguf(
    dest: Path,
    *,
    n_vocab: int = 64,
    include_json_tokens: bool = False,
) -> Path:
    """Build a tiny deterministic 6-layer mixed Dense (0..3) + Sparse MoE (4..5) GGUF v3 file.

    Mirrors `crates/layerstream-core/tests/parity_test.rs::write_synthetic_mixed_dense_moe_gguf`
    so Python integration tests exercise real llama.cpp streaming, MoE active-expert streaming,
    chat templating, KV reuse, and GIL release without external model downloads.
    """
    n_embd = 64
    n_head = 2
    n_head_kv = 2
    n_ff = 128
    n_ff_exp = 64
    n_layers = 6
    n_dense_lead = 4
    n_experts = 4
    n_experts_used = 2
    align = 32

    def write_str(buf: bytearray, s: str) -> None:
        raw = s.encode("utf-8")
        buf += struct.pack("<Q", len(raw))
        buf += raw

    def kv_str(buf: bytearray, k: str, v: str) -> None:
        write_str(buf, k)
        buf += struct.pack("<I", 8)  # STRING
        write_str(buf, v)

    def kv_u32(buf: bytearray, k: str, v: int) -> None:
        write_str(buf, k)
        buf += struct.pack("<II", 4, v)  # UINT32

    def kv_f32(buf: bytearray, k: str, v: float) -> None:
        write_str(buf, k)
        buf += struct.pack("<If", 6, v)  # FLOAT32

    def kv_bool(buf: bytearray, k: str, v: bool) -> None:
        write_str(buf, k)
        buf += struct.pack("<IB", 7, 1 if v else 0)  # BOOL

    seed = 0x5EED_C0DE_1234_5678
    mask64 = (1 << 64) - 1

    def next_f32(scale: float) -> float:
        nonlocal seed
        seed = (seed ^ ((seed << 13) & mask64)) & mask64
        seed = (seed ^ (seed >> 7)) & mask64
        seed = (seed ^ ((seed << 17) & mask64)) & mask64
        u = (seed >> 40) / float(1 << 24)
        return (u * 2.0 - 1.0) * scale

    tensors: list[tuple[str, list[int], bytes, int]] = []

    def push_tensor(name: str, dims: list[int], is_norm: bool) -> None:
        nelems = math.prod(dims)
        vals = [
            (1.0 + next_f32(0.05)) if is_norm else next_f32(0.15)
            for _ in range(nelems)
        ]
        raw = struct.pack(f"<{nelems}f", *vals)
        tensors.append((name, dims, raw, 0))

    push_tensor("token_embd.weight", [n_embd, n_vocab], False)
    push_tensor("output_norm.weight", [n_embd], True)
    push_tensor("output.weight", [n_embd, n_vocab], False)

    for il in range(n_layers):
        push_tensor(f"blk.{il}.attn_norm.weight", [n_embd], True)
        push_tensor(f"blk.{il}.attn_q.weight", [n_embd, n_embd], False)
        push_tensor(f"blk.{il}.attn_k.weight", [n_embd, n_embd], False)
        push_tensor(f"blk.{il}.attn_v.weight", [n_embd, n_embd], False)
        push_tensor(f"blk.{il}.attn_output.weight", [n_embd, n_embd], False)
        push_tensor(f"blk.{il}.ffn_norm.weight", [n_embd], True)
        if il < n_dense_lead:
            push_tensor(f"blk.{il}.ffn_gate.weight", [n_embd, n_ff], False)
            push_tensor(f"blk.{il}.ffn_up.weight", [n_embd, n_ff], False)
            push_tensor(f"blk.{il}.ffn_down.weight", [n_ff, n_embd], False)
        else:
            push_tensor(f"blk.{il}.ffn_gate_inp.weight", [n_embd, n_experts], False)
            push_tensor(
                f"blk.{il}.ffn_gate_exps.weight", [n_embd, n_ff_exp, n_experts], False
            )
            push_tensor(
                f"blk.{il}.ffn_up_exps.weight", [n_embd, n_ff_exp, n_experts], False
            )
            push_tensor(
                f"blk.{il}.ffn_down_exps.weight", [n_ff_exp, n_embd, n_experts], False
            )

    positioned: list[tuple[str, list[int], bytes, int]] = []
    cur_off = 0
    for name, dims, raw, _ in tensors:
        cur_off = (cur_off + align - 1) & ~(align - 1)
        positioned.append((name, dims, raw, cur_off))
        cur_off += len(raw)

    kv_buf = bytearray()
    kv_count = 0
    arch = "ernie4_5-moe"
    for fn, key, val in [
        (kv_str, "general.architecture", arch),
        (kv_str, "general.name", "synthetic-mixed-dense-moe"),
        (kv_u32, f"{arch}.block_count", n_layers),
        (kv_u32, f"{arch}.context_length", 1024 if include_json_tokens else 256),
        (kv_u32, f"{arch}.embedding_length", n_embd),
        (kv_u32, f"{arch}.feed_forward_length", n_ff),
        (kv_u32, f"{arch}.attention.head_count", n_head),
        (kv_u32, f"{arch}.attention.head_count_kv", n_head_kv),
        (kv_f32, f"{arch}.attention.layer_norm_rms_epsilon", 1e-5),
        (kv_f32, f"{arch}.rope.freq_base", 10_000.0),
        (kv_u32, f"{arch}.expert_count", n_experts),
        (kv_u32, f"{arch}.expert_used_count", n_experts_used),
        (kv_u32, f"{arch}.expert_feed_forward_length", n_ff_exp),
        (kv_u32, f"{arch}.interleave_moe_layer_step", 1),
        (kv_u32, f"{arch}.leading_dense_block_count", n_dense_lead),
        (kv_str, "tokenizer.ggml.model", "gpt2"),
        (kv_str, "tokenizer.ggml.pre", "default"),
    ]:
        fn(kv_buf, key, val)
        kv_count += 1

    # Empty merges array
    write_str(kv_buf, "tokenizer.ggml.merges")
    kv_buf += struct.pack("<IIQ", 9, 8, 0)
    kv_count += 1

    # 64-token vocabulary
    json_extras = [
        "{",
        "}",
        "[",
        "]",
        ":",
        ",",
        '"',
        "-",
        ".",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "_",
        "\n",
    ]
    write_str(kv_buf, "tokenizer.ggml.tokens")
    kv_buf += struct.pack("<IIQ", 9, 8, n_vocab)
    for i in range(n_vocab):
        if i == 0:
            tok = "<bos>"
        elif i == 1:
            tok = "<eos>"
        elif i == 2:
            tok = " "
        elif i < 29:
            tok = chr(ord("a") + (i - 3))
        elif include_json_tokens and (i - 29) < len(json_extras):
            tok = json_extras[i - 29]
        else:
            ch = chr(ord("a") + ((i - 3) % 26))
            tok = f"{ch}{i}"
        write_str(kv_buf, tok)
    kv_count += 1

    write_str(kv_buf, "tokenizer.ggml.token_type")
    kv_buf += struct.pack("<IIQ", 9, 5, n_vocab)
    for i in range(n_vocab):
        kv_buf += struct.pack("<i", 3 if i < 2 else 1)
    kv_count += 1

    kv_u32(kv_buf, "tokenizer.ggml.bos_token_id", 0)
    kv_count += 1
    kv_u32(kv_buf, "tokenizer.ggml.eos_token_id", 1)
    kv_count += 1
    kv_bool(kv_buf, "tokenizer.ggml.add_bos_token", True)
    kv_count += 1

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(positioned), kv_count)
    header += kv_buf
    for name, dims, _, rel_off in positioned:
        write_str(header, name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", d)
        header += struct.pack("<IQ", 0, rel_off)  # GGML_TYPE_F32

    aligned_header = (len(header) + align - 1) & ~(align - 1)
    header += b"\x00" * (aligned_header - len(header))

    with dest.open("wb") as f:
        f.write(header)
        written = 0
        for _, _, raw, rel_off in positioned:
            pad = rel_off - written
            if pad > 0:
                f.write(b"\x00" * pad)
                written += pad
            f.write(raw)
            written += len(raw)

    return dest


@pytest.fixture(scope="session")
def synthetic_gguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gguf") / "synthetic-mixed-dense-moe.gguf"
    return write_synthetic_mixed_dense_moe_gguf(path)


@pytest.fixture(scope="session")
def incompatible_vocab_gguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gguf-mismatch") / "synthetic-vocab-512.gguf"
    return write_synthetic_mixed_dense_moe_gguf(path, n_vocab=512)


@pytest.fixture(scope="session")
def synthetic_json_gguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gguf-json") / "synthetic-json.gguf"
    return write_synthetic_mixed_dense_moe_gguf(path, include_json_tokens=True)
