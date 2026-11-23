# layerstream (Node.js / TypeScript SDK)

Native Node-API (`napi-rs`) & TypeScript SDK for [`layerstream-core`](../../crates/layerstream-core): an out-of-core GGUF layer-streaming inference engine backed by unpatched `llama.cpp`, `O_DIRECT` / `F_NOCACHE` async ping-pong GPU streaming, auto-sized hybrid residency, sparse MoE TinyLFU expert caching, and speculative decoding.

Zero runtime npm dependencies, dual ESM + CommonJS exports, full `AbortSignal` mid-prefill/mid-decode cancellation, `Symbol.asyncDispose` (`await using`), and `isolatedWorker: true` for true 0 MB GPU / CUDA / Vulkan / Metal driver context handoff.

## Quickstart

```ts
import { AutoModel } from "layerstream";

await using model = await AutoModel.fromPretrained("Qwen/Qwen2.5-14B-Instruct", {
  draftModel: "Qwen/Qwen2.5-0.5B-Instruct",
  isolatedWorker: true,
});

for await (const chunk of await model.chat(
  [{ role: "user", content: "Explain NVMe O_DIRECT layer streaming." }],
  { stream: true, maxTokens: 128 }
)) {
  process.stdout.write(chunk);
}
// <- Worker reaped on scope exit; 0 B VRAM or GPU driver context remains in Node!
```

## Drop-In In-Process `OpenAI` Client & Structured Outputs

Use `OpenAI` from `"layerstream"` (or `"layerstream/openai"`) directly in-process with zero HTTP daemon or port setup, native GBNF / JSON Schema / Zod (`toJSONSchema()`) constrained decoding, `<think>` → `reasoning_content`, and `<tool_call>` → `tool_calls`:

```ts
import { OpenAI } from "layerstream";

const client = new OpenAI("Qwen/Qwen2.5-14B-Instruct", { isolatedWorker: true });
try {
  const resp = await client.chat.completions.parse({
    messages: [{ role: "user", content: "Recommend Q4_K_M for an 8 GB Orin Nano." }],
    response_format: {
      type: "json_schema",
      json_schema: {
        schema: {
          type: "object",
          properties: {
            quant: { type: "string" },
            fitsEdgeGpu: { type: "boolean" },
          },
          required: ["quant", "fitsEdgeGpu"],
          additionalProperties: false,
        },
      },
    },
  });
  console.log(resp.choices[0].message.parsed);
} finally {
  await client.close();
}
```

## Hardware Advisor, Preflight & Low-Level Controls

```ts
import { check, inspectGguf, recommend } from "layerstream";

// Rank GGUF quantizations in any HF repo (or base repo with -GGUF sibling) against GPU + NVMe:
const recs = await recommend("Qwen/Qwen3-30B-A3B-GGUF");

// Header-only (~ms, 0 MB VRAM) local GGUF metadata inspection:
const info = await inspectGguf("/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf");

// Preflight check (resolves/downloads uncached HF specs, probes GPU VRAM, and benchmarks storage I/O):
const report = await check(
  "hf://Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
);
```

`AutoModel` also exposes `.tokenize(text)`, `.detokenize(ids)`, `.applyChatTemplate(messages)`, `.kvClear()`, `.kvTrim(posMin)`, `grammar`, `jsonSchema`, `tools`, `parseOutput: true`, and `AbortController` `signal` cancellation.

## Building & Testing Locally

```bash
nix develop .#node
cd bindings/node
node scripts/build-native.mjs --locked
npm run typecheck
npm test
```

See the root [README.md](../../README.md) for bare-metal benchmarks and Mesh-LLM plugin usage.
