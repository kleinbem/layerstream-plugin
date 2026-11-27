import fs from "node:fs";
import path from "node:path";

function writeSyntheticMixedDenseMoeGguf(
  destPath,
  { nVocab = 64, includeJsonTokens = false } = {},
) {
  const nEmbd = 64;
  const nHead = 2;
  const nHeadKv = 2;
  const nFf = 128;
  const nFfExp = 64;
  const nLayers = 6;
  const nDenseLead = 4;
  const nExperts = 4;
  const nExpertsUsed = 2;
  const align = 32;

  function makeStrBuf(str) {
    const raw = Buffer.from(str, "utf8");
    const lenBuf = Buffer.alloc(8);
    lenBuf.writeBigUInt64LE(BigInt(raw.length), 0);
    return Buffer.concat([lenBuf, raw]);
  }

  function kvStr(key, val) {
    const t = Buffer.alloc(4);
    t.writeUInt32LE(8, 0);
    return Buffer.concat([makeStrBuf(key), t, makeStrBuf(val)]);
  }

  function kvU32(key, val) {
    const b = Buffer.alloc(8);
    b.writeUInt32LE(4, 0);
    b.writeUInt32LE(val >>> 0, 4);
    return Buffer.concat([makeStrBuf(key), b]);
  }

  function kvF32(key, val) {
    const b = Buffer.alloc(8);
    b.writeUInt32LE(6, 0);
    b.writeFloatLE(val, 4);
    return Buffer.concat([makeStrBuf(key), b]);
  }

  function kvBool(key, val) {
    const b = Buffer.alloc(5);
    b.writeUInt32LE(7, 0);
    b.writeUInt8(val ? 1 : 0, 4);
    return Buffer.concat([makeStrBuf(key), b]);
  }

  let seed = 0x5eed_c0de_1234_5678n;
  const mask64 = (1n << 64n) - 1n;

  function nextF32(scale) {
    seed = (seed ^ ((seed << 13n) & mask64)) & mask64;
    seed = (seed ^ (seed >> 7n)) & mask64;
    seed = (seed ^ ((seed << 17n) & mask64)) & mask64;
    const u = Number(seed >> 40n) / Number(1 << 24);
    return (u * 2.0 - 1.0) * scale;
  }

  const jsonExtras = [
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
  ];

  const tensors = [];

  function pushTensor(name, dims, isNorm) {
    const nelems = dims.reduce((a, b) => a * b, 1);
    const vals = new Float32Array(nelems);
    for (let i = 0; i < nelems; i += 1) {
      vals[i] = isNorm ? 1.0 + nextF32(0.05) : nextF32(0.15);
    }
    if (includeJsonTokens) {
      if (name === "token_embd.weight") {
        for (let v = 0; v < nVocab; v += 1) {
          vals[v * nEmbd] = 2.0;
        }
      } else if (name === "output.weight") {
        const tokBias = new Map([
          [1, -0.5],
          [2, -1.5],
        ]);
        for (let v = 3; v < Math.min(29, nVocab); v += 1) {
          tokBias.set(v, 0.4);
        }
        const extraBias = {
          "}": 2.8,
          "]": 2.6,
          '"': 2.4,
          ",": 1.8,
          ".": 1.5,
          1: 1.2,
          "\n": -1.5,
        };
        for (let idxE = 0; idxE < jsonExtras.length; idxE += 1) {
          const tid = 29 + idxE;
          if (tid < nVocab) {
            const ch = jsonExtras[idxE];
            tokBias.set(tid, Object.hasOwn(extraBias, ch) ? extraBias[ch] : -1.2);
          }
        }
        for (const [tid, bval] of tokBias.entries()) {
          vals[tid * nEmbd] = bval;
        }
      } else if (!isNorm) {
        for (let i = 0; i < nelems; i += 1) {
          vals[i] *= 0.15;
        }
      }
    }
    const raw = Buffer.from(vals.buffer, vals.byteOffset, vals.byteLength);
    tensors.push({ name, dims, raw, relOff: 0 });
  }

  pushTensor("token_embd.weight", [nEmbd, nVocab], false);
  pushTensor("output_norm.weight", [nEmbd], true);
  pushTensor("output.weight", [nEmbd, nVocab], false);

  for (let il = 0; il < nLayers; il += 1) {
    pushTensor(`blk.${il}.attn_norm.weight`, [nEmbd], true);
    pushTensor(`blk.${il}.attn_q.weight`, [nEmbd, nEmbd], false);
    pushTensor(`blk.${il}.attn_k.weight`, [nEmbd, nEmbd], false);
    pushTensor(`blk.${il}.attn_v.weight`, [nEmbd, nEmbd], false);
    pushTensor(`blk.${il}.attn_output.weight`, [nEmbd, nEmbd], false);
    pushTensor(`blk.${il}.ffn_norm.weight`, [nEmbd], true);
    if (il < nDenseLead) {
      pushTensor(`blk.${il}.ffn_gate.weight`, [nEmbd, nFf], false);
      pushTensor(`blk.${il}.ffn_up.weight`, [nEmbd, nFf], false);
      pushTensor(`blk.${il}.ffn_down.weight`, [nFf, nEmbd], false);
    } else {
      pushTensor(`blk.${il}.ffn_gate_inp.weight`, [nEmbd, nExperts], false);
      pushTensor(`blk.${il}.ffn_gate_exps.weight`, [nEmbd, nFfExp, nExperts], false);
      pushTensor(`blk.${il}.ffn_up_exps.weight`, [nEmbd, nFfExp, nExperts], false);
      pushTensor(`blk.${il}.ffn_down_exps.weight`, [nFfExp, nEmbd, nExperts], false);
    }
  }

  let curOff = 0;
  for (const t of tensors) {
    curOff = (curOff + align - 1) & ~(align - 1);
    t.relOff = curOff;
    curOff += t.raw.length;
  }

  const arch = "ernie4_5-moe";
  const kvParts = [
    kvStr("general.architecture", arch),
    kvStr("general.name", "synthetic-mixed-dense-moe"),
    kvU32(`${arch}.block_count`, nLayers),
    kvU32(`${arch}.context_length`, includeJsonTokens ? 1024 : 256),
    kvU32(`${arch}.embedding_length`, nEmbd),
    kvU32(`${arch}.feed_forward_length`, nFf),
    kvU32(`${arch}.attention.head_count`, nHead),
    kvU32(`${arch}.attention.head_count_kv`, nHeadKv),
    kvF32(`${arch}.attention.layer_norm_rms_epsilon`, 1e-5),
    kvF32(`${arch}.rope.freq_base`, 10000.0),
    kvU32(`${arch}.expert_count`, nExperts),
    kvU32(`${arch}.expert_used_count`, nExpertsUsed),
    kvU32(`${arch}.expert_feed_forward_length`, nFfExp),
    kvU32(`${arch}.interleave_moe_layer_step`, 1),
    kvU32(`${arch}.leading_dense_block_count`, nDenseLead),
    kvStr("tokenizer.ggml.model", "gpt2"),
    kvStr("tokenizer.ggml.pre", "default"),
  ];

  // Empty merges array
  const mergesHdr = Buffer.alloc(16);
  mergesHdr.writeUInt32LE(9, 0);
  mergesHdr.writeUInt32LE(8, 4);
  mergesHdr.writeBigUInt64LE(0n, 8);
  kvParts.push(Buffer.concat([makeStrBuf("tokenizer.ggml.merges"), mergesHdr]));

  // Tokens array
  const tokArrHdr = Buffer.alloc(16);
  tokArrHdr.writeUInt32LE(9, 0);
  tokArrHdr.writeUInt32LE(8, 4);
  tokArrHdr.writeBigUInt64LE(BigInt(nVocab), 8);
  const tokParts = [makeStrBuf("tokenizer.ggml.tokens"), tokArrHdr];
  for (let i = 0; i < nVocab; i += 1) {
    let tok;
    if (i === 0) tok = "<bos>";
    else if (i === 1) tok = "<eos>";
    else if (i === 2) tok = " ";
    else if (i < 29) tok = String.fromCharCode("a".charCodeAt(0) + (i - 3));
    else if (includeJsonTokens && i - 29 < jsonExtras.length) tok = jsonExtras[i - 29];
    else {
      const ch = String.fromCharCode("a".charCodeAt(0) + ((i - 3) % 26));
      tok = `${ch}${i}`;
    }
    tokParts.push(makeStrBuf(tok));
  }
  kvParts.push(Buffer.concat(tokParts));

  // Token types array
  const typeHdr = Buffer.alloc(16);
  typeHdr.writeUInt32LE(9, 0);
  typeHdr.writeUInt32LE(5, 4);
  typeHdr.writeBigUInt64LE(BigInt(nVocab), 8);
  const typeBuf = Buffer.alloc(nVocab * 4);
  for (let i = 0; i < nVocab; i += 1) {
    typeBuf.writeInt32LE(i < 2 ? 3 : 1, i * 4);
  }
  kvParts.push(Buffer.concat([makeStrBuf("tokenizer.ggml.token_type"), typeHdr, typeBuf]));

  kvParts.push(kvU32("tokenizer.ggml.bos_token_id", 0));
  kvParts.push(kvU32("tokenizer.ggml.eos_token_id", 1));
  kvParts.push(kvBool("tokenizer.ggml.add_bos_token", true));

  const preamble = Buffer.alloc(24);
  preamble.write("GGUF", 0, 4, "ascii");
  preamble.writeUInt32LE(3, 4);
  preamble.writeBigUInt64LE(BigInt(tensors.length), 8);
  preamble.writeBigUInt64LE(BigInt(kvParts.length), 16);

  const tensorInfos = [];
  for (const t of tensors) {
    const info = Buffer.alloc(4 + t.dims.length * 8 + 4 + 8);
    info.writeUInt32LE(t.dims.length, 0);
    for (let d = 0; d < t.dims.length; d += 1) {
      info.writeBigUInt64LE(BigInt(t.dims[d]), 4 + d * 8);
    }
    const afterDims = 4 + t.dims.length * 8;
    info.writeUInt32LE(0, afterDims); // GGML_TYPE_F32
    info.writeBigUInt64LE(BigInt(t.relOff), afterDims + 4);
    tensorInfos.push(makeStrBuf(t.name), info);
  }

  const headerUnaligned = Buffer.concat([preamble, ...kvParts, ...tensorInfos]);
  const alignedHeaderLen = (headerUnaligned.length + align - 1) & ~(align - 1);
  const headerPad = Buffer.alloc(alignedHeaderLen - headerUnaligned.length);

  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const fd = fs.openSync(destPath, "w");
  try {
    fs.writeSync(fd, headerUnaligned);
    if (headerPad.length > 0) fs.writeSync(fd, headerPad);
    let written = 0;
    for (const t of tensors) {
      const pad = t.relOff - written;
      if (pad > 0) {
        fs.writeSync(fd, Buffer.alloc(pad));
        written += pad;
      }
      fs.writeSync(fd, t.raw);
      written += t.raw.length;
    }
  } finally {
    fs.closeSync(fd);
  }

  return destPath;
}

export { writeSyntheticMixedDenseMoeGguf };
