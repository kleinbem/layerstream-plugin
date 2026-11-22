"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const { after, before, describe, test } = require("node:test");

const {
  AutoModel,
  OpenAI,
  StreamTagParser,
  check,
  formatToolsSystemPrompt,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  recommend,
  version,
} = require("../index.js");
const { writeSyntheticMixedDenseMoeGguf } = require("./conftest.js");

let tmpDir;
let syntheticGguf;
let syntheticJsonGguf;
let incompatibleVocabGguf;

before(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "layerstream-node-test-"));
  syntheticGguf = writeSyntheticMixedDenseMoeGguf(path.join(tmpDir, "synthetic.gguf"));
  syntheticJsonGguf = writeSyntheticMixedDenseMoeGguf(
    path.join(tmpDir, "synthetic-json.gguf"),
    { includeJsonTokens: true }
  );
  incompatibleVocabGguf = writeSyntheticMixedDenseMoeGguf(
    path.join(tmpDir, "incompatible-512.gguf"),
    { nVocab: 512 }
  );
});

after(() => {
  if (tmpDir) {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

describe("layerstream Node-API SDK", () => {
  test("exports version and probes GPU memory safely", async () => {
    assert.equal(typeof version, "string");
    assert.ok(version.length > 0);
    const cpuMem = await probeGpuMemory(false);
    assert.equal(cpuMem, null);
  });

  test("loads, generates, streams, chats, tokenizes, trims KV, and unloads cleanly", async () => {
    const model = await AutoModel.fromPretrained(syntheticGguf, {
      useGpu: false,
      ctxSize: 256,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
    });
    try {
      assert.equal(model.isLoaded, true);
      assert.equal(model.isolatedWorker, false);
      assert.equal(model.modelPath, syntheticGguf);

      // Verify JS event loop ticks continuously while native generation runs
      let ticks = 0;
      const ticker = setInterval(() => {
        ticks += 1;
      }, 2);
      let out;
      try {
        out = await model.generate("a b c", { maxTokens: 12, temperature: 0.0 });
      } finally {
        clearInterval(ticker);
      }
      assert.equal(typeof out, "string");
      assert.ok(out.length > 0);
      assert.ok(ticks >= 1, "expected event loop timers to fire during native inference");
      assert.ok(model.lastSummary && model.lastSummary.completionTokens > 0);

      // AsyncIterable streaming parity
      const stream = await model.generate("a b c", {
        maxTokens: 12,
        temperature: 0.0,
        stream: true,
      });
      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      assert.equal(chunks.join(""), out);
      assert.ok(stream.summary && stream.summary.completionTokens > 0);

      // Tokenize / detokenize / applyChatTemplate / kvTrim / kvClear
      const ids = await model.tokenize("a b c");
      assert.ok(Array.isArray(ids) && ids.length > 0);
      const detok = await model.detokenize(ids);
      assert.equal(typeof detok, "string");

      const tmpl = await model.applyChatTemplate([
        { role: "user", content: "hello" },
        { role: "tool", name: "lookup", tool_call_id: "call_1", content: "42" },
      ]);
      assert.ok(tmpl.includes("tool"));
      assert.ok(tmpl.includes("42"));

      assert.equal(typeof (await model.kvTrim(1)), "boolean");
      await model.kvClear();

      const st = await model.stats();
      assert.equal(st.loaded, true);
      assert.equal(st.nLayers, 6);
      assert.equal(st.moeLayers, 2);
    } finally {
      await model.unload();
      // Idempotent second unload
      await model.unload();
      assert.equal(model.isLoaded, false);
    }
  });

  test("supports AbortSignal cancellation and early break in async streaming", async () => {
    const model = await AutoModel.fromPretrained(syntheticGguf, {
      useGpu: false,
      ctxSize: 256,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
    });
    try {
      // Early break in `for await` invokes `return()` and releases engine lock cleanly
      const s1 = await model.generate("a b c", { maxTokens: 32, stream: true });
      let seen = 0;
      for await (const _chunk of s1) {
        seen += 1;
        if (seen >= 2) break;
      }
      assert.equal(seen, 2);

      // Pre-aborted signal rejects immediately
      const preCtrl = new AbortController();
      preCtrl.abort();
      await assert.rejects(
        () => model.generate("a b c", { maxTokens: 16, signal: preCtrl.signal }),
        (err) => err.name === "AbortError"
      );

      // Engine remains usable immediately after early stream break + abort
      const followUp = await model.generate("a b", { maxTokens: 4 });
      assert.ok(followUp.length > 0);
    } finally {
      await model.unload();
    }
  });

  test("isolatedWorker=true matches in-process output and reaps child process on unload", async () => {
    const inproc = await AutoModel.fromPretrained(syntheticGguf, {
      useGpu: false,
      ctxSize: 256,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
      isolatedWorker: false,
    });
    let expected;
    let expectedDetok;
    try {
      expected = await inproc.generate("a b c", { maxTokens: 8, temperature: 0.0 });
      expectedDetok = await inproc.detokenize(await inproc.tokenize("a b c"));
    } finally {
      await inproc.unload();
    }

    const iso = await AutoModel.fromPretrained(syntheticGguf, {
      useGpu: false,
      ctxSize: 256,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
      isolatedWorker: true,
    });
    const pid = iso.workerPid;
    assert.equal(typeof pid, "number");
    assert.ok(pid > 0);

    try {
      const actual = await iso.generate("a b c", { maxTokens: 8, temperature: 0.0 });
      assert.equal(actual, expected);

      const streamed = [];
      for await (const ch of await iso.generate("a b c", {
        maxTokens: 8,
        temperature: 0.0,
        stream: true,
      })) {
        streamed.push(ch);
      }
      assert.equal(streamed.join(""), expected);

      const ids = await iso.tokenize("a b c");
      assert.equal(await iso.detokenize(ids), expectedDetok);

      const st = await iso.stats();
      assert.equal(st.isolatedWorker, true);
      assert.equal(st.workerPid, pid);
    } finally {
      await iso.unload();
    }

    assert.equal(iso.isLoaded, false);
    assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  });

  test("enforces GBNF & JSON Schema constrained decoding (greedy, sampled, and speculative)", async () => {
    const schema = {
      $defs: {
        Meta: {
          type: "object",
          properties: { ok: { type: "boolean" } },
          required: ["ok"],
          additionalProperties: false,
        },
      },
      type: "object",
      properties: {
        mode: { type: "string", enum: ["fast", "safe"] },
        n: { type: "integer" },
        meta: { $ref: "#/$defs/Meta" },
      },
      required: ["mode", "n", "meta"],
      additionalProperties: false,
    };

    assert.ok(jsonSchemaToGbnf(schema).startsWith("root ::= ws "));
    assert.ok(jsonObjectGbnf().startsWith("root ::= ws json-object"));

    const model = await AutoModel.fromPretrained(syntheticJsonGguf, {
      draftModel: syntheticJsonGguf,
      maxDraftTokens: 4,
      useGpu: false,
      ctxSize: 256,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
    });
    try {
      const rawGreedy = await model.generate("a b c", {
        maxTokens: 96,
        temperature: 0.0,
        jsonSchema: schema,
      });
      const parsed = JSON.parse(rawGreedy);
      assert.deepEqual(Object.keys(parsed).sort(), ["meta", "mode", "n"]);
      assert.ok(["fast", "safe"].includes(parsed.mode));
      assert.equal(typeof parsed.n, "number");
      assert.ok(Number.isInteger(parsed.n));
      assert.deepEqual(Object.keys(parsed.meta), ["ok"]);
      assert.equal(typeof parsed.meta.ok, "boolean");

      const parts = [];
      for await (const ch of await model.generate("a b c", {
        maxTokens: 96,
        temperature: 0.8,
        seed: 777,
        jsonSchema: schema,
        stream: true,
      })) {
        parts.push(ch);
      }
      const parsedSampled = JSON.parse(parts.join(""));
      assert.deepEqual(Object.keys(parsedSampled).sort(), ["meta", "mode", "n"]);
      assert.ok(["fast", "safe"].includes(parsedSampled.mode));
      assert.ok(Number.isInteger(parsedSampled.n));
      assert.equal(typeof parsedSampled.meta.ok, "boolean");

      // Invalid grammar and unsupported schema constraints fail closed
      await assert.rejects(
        () => model.generate("a b", { maxTokens: 8, grammar: "not valid gbnf ===" }),
        /grammar/i
      );
      assert.throws(() => jsonSchemaToGbnf({ type: "integer", minimum: 1 }), /minimum/);
    } finally {
      await model.unload();
    }
  });

  test("supports drop-in in-process OpenAI client, .parse() with Zod-like schema, and tag parsing", async () => {
    const parsedTags = parseAssistantOutput(
      '<think>step 1</think>Result <tool_call>{"name":"ping","arguments":{"x":1}}</tool_call>'
    );
    assert.equal(parsedTags.reasoningContent, "step 1");
    assert.equal(parsedTags.content, "Result");
    assert.equal(parsedTags.toolCalls.length, 1);
    assert.equal(parsedTags.toolCalls[0].function.name, "ping");

    const parser = new StreamTagParser();
    const deltas = [
      ...parser.push("<thi"),
      ...parser.push('nk>plan</think>ok <tool_call>{"name":"f","arguments":{}}</tool_call>'),
      ...parser.flush(),
    ];
    assert.equal(parser.hasToolCalls, true);
    assert.ok(deltas.some((e) => e.type === "reasoning" && e.text === "plan"));
    assert.ok(deltas.some((e) => e.type === "tool_call" && e.name === "f"));

    const toolsSpec = [
      {
        type: "function",
        function: { name: "lookup", parameters: { type: "object" } },
      },
    ];
    assert.ok(formatToolsSystemPrompt(toolsSpec).includes("lookup"));
    assert.throws(() => formatToolsSystemPrompt([1]), TypeError);

    const mockZodSchema = {
      toJSONSchema() {
        return {
          type: "object",
          properties: {
            mode: { type: "string", enum: ["alpha", "beta"] },
            flag: { type: "boolean" },
          },
          required: ["mode", "flag"],
          additionalProperties: false,
        };
      },
      parse(data) {
        assert.ok(["alpha", "beta"].includes(data.mode));
        assert.equal(typeof data.flag, "boolean");
        return { validated: true, ...data };
      },
    };

    const client = new OpenAI(syntheticJsonGguf, {
      useGpu: false,
      ctxSize: 1024,
      batchSize: 64,
      vramBudgetMb: 0,
      preferDirectIo: false,
    });
    try {
      const modelsPage = await client.models.list();
      assert.equal(modelsPage.data.length, 1);

      const resp = await client.chat.completions.create({
        messages: [{ role: "user", content: "a b c" }],
        tools: toolsSpec,
        max_tokens: 8,
      });
      assert.equal(resp.object, "chat.completion");
      assert.ok(resp.usage && resp.usage.total_tokens > 0);

      const parsedResp = await client.chat.completions.parse({
        messages: [{ role: "user", content: "a b" }],
        response_format: mockZodSchema,
        max_tokens: 96,
      });
      assert.equal(parsedResp.choices[0].message.parsed.validated, true);
      assert.ok(["alpha", "beta"].includes(parsedResp.choices[0].message.parsed.mode));

      const chatStream = await client.chat.completions.create({
        messages: [{ role: "user", content: "a b" }],
        max_tokens: 6,
        stream: true,
      });
      const streamChunks = [];
      for await (const ch of chatStream) {
        streamChunks.push(ch);
      }
      assert.equal(streamChunks[0].choices[0].delta.role, "assistant");
      assert.ok(
        ["stop", "length", "tool_calls"].includes(
          streamChunks[streamChunks.length - 1].choices[0].finish_reason
        )
      );

      const compStream = await client.completions.create({
        prompt: "a b",
        max_tokens: 4,
        stream: true,
      });
      const compChunks = [];
      for await (const item of compStream) {
        compChunks.push(item.choices[0].text);
      }
      assert.ok(compChunks.join("").length > 0);
    } finally {
      await client.close();
    }
  });

  test("supports inspectGguf, check, recommend, and pull", async () => {
    const info = await inspectGguf(syntheticGguf);
    assert.equal(info.architecture, "ernie4_5-moe");
    assert.equal(info.nLayers, 6);
    assert.equal(info.shardCount, 1);
    assert.ok(info.tensorCount > 0);

    const report = await check(syntheticGguf, {
      draftModel: syntheticGguf,
      useGpu: false,
    });
    assert.equal(report.ok, true);
    assert.ok(["GOOD", "OK"].includes(report.verdict));

    await assert.rejects(
      () =>
        check(syntheticGguf, {
          draftModel: incompatibleVocabGguf,
          useGpu: false,
        }),
      /incompatible draftModel tokenizer/
    );

    const payload = fs.readFileSync(syntheticGguf);
    const sha = crypto.createHash("sha256").update(payload).digest("hex");
    const gib = 1024 * 1024 * 1024;
    const treeBase = JSON.stringify([{ type: "file", path: "model.safetensors", size: 64 }]);
    const treeGguf = JSON.stringify([
      { type: "file", path: "Model-Q4_K_M.gguf", size: 8 * gib, lfs: { size: 8 * gib } },
      { type: "file", path: "Model-Q8_0.gguf", size: 16 * gib, lfs: { size: 16 * gib } },
      { type: "file", path: "Tiny.gguf", size: payload.length, lfs: { size: payload.length } },
    ]);

    const server = http.createServer((req, res) => {
      if (req.url === "/api/models/AdvisorOrg/Model/tree/main?recursive=true") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(treeBase);
      } else if (req.url === "/api/models/AdvisorOrg/Model-GGUF/tree/main?recursive=true") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(treeGguf);
      } else if (req.url === "/AdvisorOrg/Model-GGUF/resolve/main/Tiny.gguf") {
        res.writeHead(200, { ETag: `"${sha}"`, "Content-Length": String(payload.length) });
        res.end(payload);
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = server.address().port;
    const prevEndpoint = process.env.HF_ENDPOINT;
    process.env.HF_ENDPOINT = `http://127.0.0.1:${port}`;
    try {
      const cacheDir = path.join(tmpDir, "hf-cache");
      const recs = await recommend("AdvisorOrg/Model", {
        freeRamGb: 5.2,
        nvmeMibS: 2800,
        hasDraft: true,
        cacheDir,
        useGpu: false,
      });
      assert.equal(recs.length, 3);
      assert.equal(recs[0].resolvedRepo, "AdvisorOrg/Model-GGUF");
      assert.ok(recs.some((r) => r.recommended));

      const pulled = await pull("hf://AdvisorOrg/Model-GGUF:Tiny.gguf", { cacheDir });
      assert.deepEqual(fs.readFileSync(pulled), payload);
    } finally {
      if (prevEndpoint === undefined) delete process.env.HF_ENDPOINT;
      else process.env.HF_ENDPOINT = prevEndpoint;
      await new Promise((resolve) => server.close(resolve));
    }
  });
});
