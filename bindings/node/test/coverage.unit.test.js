import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { createAbortError, isPlainObject, throwIfAborted } from "../_shared.js";
import {
  AutoModel,
  check,
  formatToolsSystemPrompt,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  OpenAI,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  recommend,
  TokenStream,
} from "../index.js";
import { ChatCompletionStream, CompletionStream } from "../openai.js";
import { runWorkerEntrypoint } from "../worker.js";

function makeFakeNativeStream(chunks, { summary = null, throwOnNext = null } = {}) {
  let idx = 0;
  let closed = false;
  let cancelled = false;
  return {
    get closed() {
      return closed;
    },
    get cancelled() {
      return cancelled;
    },
    summary,
    cancel() {
      cancelled = true;
    },
    async close() {
      closed = true;
    },
    async nextChunk() {
      if (throwOnNext) {
        const e = throwOnNext;
        throwOnNext = null;
        throw e;
      }
      if (closed || idx >= chunks.length) return null;
      return chunks[idx++];
    },
  };
}

function makeFakeBackend({
  failWith = null,
  chatText = "hello",
  chatParsed = { content: "hello", reasoningContent: null, toolCalls: [] },
  genText = "world",
  summary = { promptTokens: 3, completionTokens: 2, finishReason: "stop" },
  streamChunks = ["a", "b"],
} = {}) {
  let loaded = true;
  const maybeFail = () => {
    if (failWith) throw failWith;
  };
  return {
    get isLoaded() {
      return loaded;
    },
    workerPid: 9876,
    modelPath: "/fake/m.gguf",
    lastSummary: summary,
    generateStream() {
      maybeFail();
      return makeFakeNativeStream(streamChunks, { summary });
    },
    async generateWithSummary() {
      maybeFail();
      return { text: genText, summary };
    },
    chatStream() {
      maybeFail();
      return Promise.resolve(makeFakeNativeStream(streamChunks, { summary }));
    },
    async chatWithSummary() {
      maybeFail();
      return { text: chatText, parsed: chatParsed, summary };
    },
    async tokenize() {
      maybeFail();
      return [1, 2];
    },
    async detokenize() {
      maybeFail();
      return "ab";
    },
    async applyChatTemplate() {
      maybeFail();
      return "tmpl";
    },
    async kvClear() {
      maybeFail();
    },
    async kvTrim() {
      maybeFail();
      return true;
    },
    async saveState() {
      maybeFail();
      return 7;
    },
    async loadState() {
      maybeFail();
      return 7;
    },
    async stats() {
      maybeFail();
      return { loaded };
    },
    async unload() {
      loaded = false;
      maybeFail();
    },
  };
}

test("_shared.js helpers cover abort error variants and plain object checks", () => {
  assert.equal(isPlainObject({}), true);
  assert.equal(isPlainObject(null), false);
  assert.equal(isPlainObject([]), false);

  const customErr = new Error("custom abort");
  assert.equal(createAbortError({ reason: customErr }), customErr);
  const strErr = createAbortError({ reason: "stopped by user" });
  assert.equal(strErr.name, "AbortError");
  assert.equal(strErr.code, "ABORT_ERR");
  assert.equal(strErr.message, "stopped by user");
  const defErr = createAbortError({});
  assert.equal(defErr.message, "The operation was aborted");

  assert.doesNotThrow(() => throwIfAborted(undefined));
  const ac = new AbortController();
  ac.abort("bye");
  assert.throws(() => throwIfAborted(ac.signal), /bye/);
});

test("worker.js entrypoint handles missing port, success, and fatal errors", () => {
  assert.equal(runWorkerEntrypoint(["node", "worker.js"]), 2);
  assert.equal(runWorkerEntrypoint(["node", "worker.js", "--worker-port"]), 2);
  let seenPort = 0;
  assert.equal(
    runWorkerEntrypoint(["node", "worker.js", "--worker-port", "41234"], (p) => {
      seenPort = p;
    }),
    0,
  );
  assert.equal(seenPort, 41234);

  const origWrite = process.stderr.write;
  let captured = "";
  process.stderr.write = (chunk) => {
    captured += String(chunk);
    return true;
  };
  try {
    assert.equal(
      runWorkerEntrypoint(["node", "worker.js", "--worker-port", "41234"], () => {
        throw new Error("boom");
      }),
      1,
    );
  } finally {
    process.stderr.write = origWrite;
  }
  assert.match(captured, /layerstream worker fatal/);
});

test("native.js handles LAYERSTREAM_NATIVE_PATH and missing addon errors", () => {
  const fakePlat = process.platform === "win32" ? "darwin" : "win32";
  const fakeArch = process.platform === "win32" ? "arm64" : "x64";
  const missing = () =>
    execFileSync(
      process.execPath,
      [
        "--input-type=module",
        "-e",
        `Object.defineProperty(process, "arch", { value: "${fakeArch}" }); ` +
          `Object.defineProperty(process, "platform", { value: "${fakePlat}" }); ` +
          'await import("./native.js");',
      ],
      {
        cwd: new URL("..", import.meta.url),
        env: { ...process.env, LAYERSTREAM_NATIVE_PATH: "/nonexistent/layerstream.node" },
        stdio: "pipe",
      },
    );
  assert.throws(missing, /Could not find native layerstream Node-API binary/);
});

test("index.js standalone helpers and validation paths", async () => {
  assert.ok(jsonObjectGbnf().startsWith("root ::= ws "));
  assert.ok(jsonSchemaToGbnf(true).startsWith("root ::= ws "));
  assert.ok(jsonSchemaToGbnf('{"type":"string"}').startsWith("root ::= ws "));
  assert.ok(
    jsonSchemaToGbnf({ toJSONSchema: () => ({ type: "integer" }) }).startsWith("root ::= ws "),
  );
  assert.throws(() => jsonSchemaToGbnf("{bad json"), TypeError);
  assert.throws(() => jsonSchemaToGbnf([]), TypeError);
  assert.throws(() => jsonSchemaToGbnf(123), TypeError);

  assert.throws(() => parseAssistantOutput(123), TypeError);
  assert.equal(typeof formatToolsSystemPrompt(null), "string");
  assert.throws(() => formatToolsSystemPrompt("bad"), TypeError);
  assert.throws(() => formatToolsSystemPrompt(["bad"]), TypeError);
  assert.equal(
    typeof formatToolsSystemPrompt([{ type: "function", function: { name: "f" } }]),
    "string",
  );

  await assert.rejects(() => AutoModel.fromPretrained(""), TypeError);
  await assert.rejects(() => AutoModel.fromPretrained("m.gguf", "bad"), TypeError);
  await assert.rejects(() => AutoModel.fromPretrained("m.gguf", { vramBudgetMb: true }), TypeError);
  await assert.rejects(() => AutoModel.fromPretrained("m.gguf", { vramBudgetMb: {} }), TypeError);
  await assert.rejects(() => AutoModel.fromPretrained("m.gguf", { vramBudgetMb: -1 }), RangeError);
  await assert.rejects(() => AutoModel.fromPretrained("m.gguf", { kvCacheType: 123 }), RangeError);
  await assert.rejects(
    () => AutoModel.fromPretrained("m.gguf", { kvCacheType: "invalid" }),
    RangeError,
  );
  await assert.rejects(
    () => AutoModel.fromPretrained("/definitely/missing/model.gguf", { useGpu: false }),
    (err) => err.code === "ENOENT",
  );

  await assert.doesNotReject(() => probeGpuMemory(false));
  await assert.rejects(() => inspectGguf(""), TypeError);
  await assert.rejects(() => inspectGguf("/nonexistent/file.gguf"), Error);
  await assert.rejects(() => pull(""), TypeError);
  await assert.rejects(() => pull("invalid-spec"), Error);
  await assert.rejects(() => recommend(""), TypeError);
  await assert.rejects(() => recommend("bad-repo-no-slash"), Error);
  await assert.rejects(() => check(""), TypeError);
  await assert.rejects(() => check("/nonexistent/file.gguf", { useGpu: false }), Error);
});

test("TokenStream covers abort, cancel, close idempotency, asyncDispose, and iterator return/throw", async () => {
  const acPre = new AbortController();
  acPre.abort();
  const nsPre = makeFakeNativeStream(["x"]);
  const tsPre = new TokenStream(nsPre, acPre.signal);
  assert.equal(nsPre.cancelled, true);
  await assert.rejects(async () => {
    for await (const _ of tsPre) {
      // unreachable
    }
  }, /aborted/);
  await tsPre.close(); // idempotent

  const acMid = new AbortController();
  const nsMid = makeFakeNativeStream(["first", "second"]);
  const tsMid = new TokenStream(nsMid, acMid.signal);
  tsMid.cancel();
  assert.equal(nsMid.cancelled, true);
  const it = tsMid[Symbol.asyncIterator]();
  const first = await it.next();
  assert.equal(first.value, "first");
  acMid.abort("mid-abort");
  await assert.rejects(() => it.next(), /mid-abort/);

  const nsErr = makeFakeNativeStream([], { throwOnNext: new Error("ABORT_ERR: cancelled") });
  const tsErr = new TokenStream(nsErr);
  await assert.rejects(async () => {
    for await (const _ of tsErr) {
      // unreachable
    }
  }, /aborted/);

  const nsThrow = makeFakeNativeStream(["a"]);
  const tsThrow = new TokenStream(nsThrow);
  const it2 = tsThrow[Symbol.asyncIterator]();
  await assert.rejects(() => it2.throw(new Error("boom")), /boom/);
  assert.equal(nsThrow.closed, true);

  const nsDisp = makeFakeNativeStream(["a"]);
  {
    await using _s = new TokenStream(nsDisp);
  }
  assert.equal(nsDisp.closed, true);

  // Abort firing while awaiting nextChunk()
  const acDuring = new AbortController();
  const nsDuring = makeFakeNativeStream(["x"]);
  nsDuring.nextChunk = async () => {
    acDuring.abort("during-chunk");
    return "x";
  };
  const tsDuring = new TokenStream(nsDuring, acDuring.signal);
  await assert.rejects(async () => {
    for await (const _ of tsDuring) {
      // unreachable
    }
  }, /during-chunk/);
});

test("AutoModel validation, isolated/inproc error mapping, abort paths, and asyncDispose", async () => {
  const fake = makeFakeBackend();
  const m = new AutoModel(fake, false);
  assert.equal(m.isLoaded, true);
  assert.equal(m.isolatedWorker, false);
  assert.equal(m.workerPid, null);
  assert.equal(m.modelPath, "/fake/m.gguf");
  assert.equal(m.lastSummary.finishReason, "stop");

  // _buildGenWire validation
  assert.throws(() => m._buildGenWire("bad"), TypeError);
  assert.throws(
    () => m._buildGenWire({ grammar: 'root ::= "a"', jsonSchema: { type: "string" } }),
    RangeError,
  );
  assert.throws(() => m._buildGenWire({ grammar: 123 }), TypeError);
  assert.throws(() => m._buildGenWire({ grammar: "   " }), RangeError);
  assert.throws(() => m._buildGenWire({ stop: 123 }), TypeError);
  assert.deepEqual(m._buildGenWire({ stop: "END" }).stop, ["END"]);
  assert.deepEqual(m._buildGenWire({ stop: ["A", "B"] }).stop, ["A", "B"]);
  assert.throws(() => m._buildGenWire({ temperature: -1 }), RangeError);
  assert.throws(() => m._buildGenWire({ topP: 0 }), RangeError);
  assert.throws(() => m._buildGenWire({ minP: 2 }), RangeError);
  assert.throws(() => m._buildGenWire({ repeatPenalty: 0 }), RangeError);
  assert.throws(() => m._buildGenWire({ frequencyPenalty: Number.POSITIVE_INFINITY }), RangeError);
  assert.throws(() => m._buildGenWire({ presencePenalty: Number.NaN }), RangeError);
  assert.throws(() => m._buildGenWire({ penaltyLastN: -2 }), RangeError);

  // generate / _generateWithSummary guards
  await assert.rejects(() => m.generate(123), TypeError);
  await assert.rejects(() => m.generate("hi", "bad"), TypeError);
  await assert.rejects(() => m._generateWithSummary(123), TypeError);
  await assert.rejects(() => m._generateWithSummary("hi", "bad"), TypeError);
  const preAc = new AbortController();
  preAc.abort();
  await assert.rejects(() => m.generate("hi", { signal: preAc.signal }), /aborted/);
  await assert.rejects(() => m._generateWithSummary("hi", { signal: preAc.signal }), /aborted/);

  // chat / _chatWithSummary / message validation
  await assert.rejects(() => m.chat("not-array"), TypeError);
  await assert.rejects(() => m.chat([]), RangeError);
  await assert.rejects(() => m.chat(["bad"]), TypeError);
  await assert.rejects(() => m.chat([{ role: "", content: "hi" }]), TypeError);
  await assert.rejects(() => m.chat([{ role: "user" }]), TypeError);
  await assert.rejects(() => m.chat([{ role: "assistant", tool_calls: "not-array" }]), TypeError);
  await assert.rejects(() => m.chat([{ role: "user", content: "hi" }], "bad"), TypeError);
  await assert.rejects(
    () => m._chatWithSummary([{ role: "user", content: "hi" }], "bad"),
    TypeError,
  );
  await assert.rejects(
    () => m.chat([{ role: "user", content: "hi" }], { signal: preAc.signal }),
    /aborted/,
  );
  await assert.rejects(
    () => m._chatWithSummary([{ role: "user", content: "hi" }], { signal: preAc.signal }),
    /aborted/,
  );
  await assert.rejects(
    () => m.chat([{ role: "user", content: "hi" }], { stream: true, parseOutput: true }),
    RangeError,
  );

  // Message with tool_calls, name, tool_call_id
  const chatOut = await m.chat(
    [{ role: "assistant", tool_calls: [{ id: "1" }], name: "fn", tool_call_id: "1" }],
    { parseOutput: true },
  );
  assert.equal(chatOut.content, "hello");

  // Chat stream abort race (slow chatStream)
  const slowChatBackend = makeFakeBackend();
  let resolveSlow;
  slowChatBackend.chatStream = () =>
    new Promise((res) => {
      resolveSlow = res;
    });
  const slowModel = new AutoModel(slowChatBackend, false);
  const raceAc = new AbortController();
  const raceP = slowModel.chat([{ role: "user", content: "hi" }], {
    stream: true,
    signal: raceAc.signal,
  });
  raceAc.abort("race-abort");
  await assert.rejects(() => raceP, /race-abort/);
  const lateStream = makeFakeNativeStream(["x"]);
  resolveSlow(lateStream);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(lateStream.closed, true);

  // Isolated mode paths (stream + non-stream + error mapping)
  const isoFake = makeFakeBackend();
  isoFake.chatStream = () => makeFakeNativeStream(["a", "b"]);
  const isoModel = new AutoModel(isoFake, true);
  assert.equal(isoModel.workerPid, 9876);
  assert.equal(await isoModel.generate("hi"), "world");
  const isoChatStream = await isoModel.chat([{ role: "user", content: "hi" }], { stream: true });
  await isoChatStream.close();
  assert.equal(
    (await isoModel.chat([{ role: "user", content: "hi" }], { parseOutput: true })).content,
    "hello",
  );

  // Isolated + inproc error mapping across all methods
  const errModel = new AutoModel(
    makeFakeBackend({ failWith: new Error("ENOENT: missing") }),
    false,
  );
  const isoErrModel = new AutoModel(makeFakeBackend({ failWith: new Error("ABORT_ERR: x") }), true);
  await assert.rejects(
    () => errModel.generate("hi", { stream: true }),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(
    () => errModel.generate("hi"),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => isoErrModel.generate("hi"), /aborted/);
  await assert.rejects(
    () => errModel.chat([{ role: "user", content: "hi" }], { stream: true }),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(
    () => isoErrModel.chat([{ role: "user", content: "hi" }], { stream: true }),
    /aborted/,
  );
  await assert.rejects(
    () => errModel.chat([{ role: "user", content: "hi" }]),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => isoErrModel.chat([{ role: "user", content: "hi" }]), /aborted/);
  await assert.rejects(() => errModel.tokenize(123), TypeError);
  await assert.rejects(
    () => errModel.tokenize("hi"),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => errModel.detokenize(["bad"]), TypeError);
  await assert.rejects(
    () => errModel.detokenize([1]),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(
    () => errModel.applyChatTemplate([{ role: "user", content: "hi" }]),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(
    () => errModel.kvClear(),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => errModel.kvTrim(-1), RangeError);
  await assert.rejects(
    () => errModel.kvTrim(1),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => errModel.saveState(""), TypeError);
  await assert.rejects(
    () => errModel.saveState("/tmp/s"),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(() => errModel.loadState(""), TypeError);
  await assert.rejects(
    () => errModel.loadState("/tmp/s"),
    (e) => e.code === "ENOENT",
  );
  await assert.rejects(
    () => errModel.stats(),
    (e) => e.code === "ENOENT",
  );

  // Abort firing during in-process generateWithSummary
  const acGen = new AbortController();
  const genAbortBackend = makeFakeBackend();
  genAbortBackend.generateWithSummary = async () => {
    acGen.abort("gen-mid-abort");
    return { text: "late", summary: null };
  };
  const genAbortModel = new AutoModel(genAbortBackend, false);
  await assert.rejects(
    () => genAbortModel.generate("hi", { signal: acGen.signal }),
    /gen-mid-abort/,
  );

  {
    await using _disp = new AutoModel(makeFakeBackend(), false);
  }
});

test("openai.js covers constructor forms, response formats, streams, fallback engines, and unloadAll", async () => {
  assert.throws(() => new OpenAI(12345), TypeError);
  const emptyClient = new OpenAI();
  await assert.rejects(() => emptyClient.completions.create({ prompt: "hi" }), RangeError);
  await emptyClient.close();

  // Plain engine without _chatWithSummary / _generateWithSummary
  const makePlainEngine = (
    chatOut,
    genOut,
    summary = { finishReason: "stop", promptTokens: 1, completionTokens: 2 },
  ) => ({
    isLoaded: true,
    modelPath: "/plain/m.gguf",
    lastSummary: summary,
    async chat(_msgs, opts) {
      if (opts.stream) {
        return new TokenStream(
          makeFakeNativeStream(Array.isArray(chatOut) ? chatOut : [chatOut], { summary }),
        );
      }
      return chatOut;
    },
    async generate(_prompt, opts) {
      if (opts.stream) {
        return new TokenStream(
          makeFakeNativeStream(Array.isArray(genOut) ? genOut : [genOut], { summary }),
        );
      }
      return genOut;
    },
    async unload() {
      this.isLoaded = false;
    },
  });

  const plain = makePlainEngine(
    '<think>hmm</think><tool_call>{"name":"ping","arguments":{"x":1}}</tool_call>',
    "plain-text",
  );
  const client = new OpenAI(plain, { apiKey: "ignored", baseURL: "ignored" });
  assert.equal((await client.models.list()).data[0].id, "/plain/m.gguf");

  // Plain engine chat (tool_calls) + completions + default alias resolution
  const rChat = await client.chat.completions.create({
    model: "/plain/m.gguf",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.equal(rChat.choices[0].finish_reason, "tool_calls");
  assert.equal(rChat.choices[0].message.reasoning_content, "hmm");

  const rComp = await client.completions.create({ prompt: "hi" });
  assert.equal(rComp.choices[0].text, "plain-text");

  // Length finish_reason
  const lenClient = new OpenAI(makePlainEngine("cut", "cut", { finishReason: "length" }));
  const rLen = await lenClient.chat.completions.create({
    messages: [{ role: "user", content: "hi" }],
  });
  assert.equal(rLen.choices[0].finish_reason, "length");

  // response_format variants in parse()
  const jsonClient = new OpenAI(makePlainEngine('{"ok":true}', "x"));
  const pBool = await jsonClient.chat.completions.parse({
    messages: [{ role: "user", content: "hi" }],
    response_format: true,
  });
  assert.deepEqual(pBool.choices[0].message.parsed, { ok: true });

  const pNoParser = await jsonClient.chat.completions.parse({
    messages: [{ role: "user", content: "hi" }],
    response_format: { toJSONSchema: () => ({ type: "object" }) },
  });
  assert.deepEqual(pNoParser.choices[0].message.parsed, { ok: true });

  const pJsonSchema = await jsonClient.chat.completions.parse({
    messages: [{ role: "user", content: "hi" }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "t",
        schema: { type: "object" },
        parse: (d) => ({ wrapped: d }),
      },
    },
  });
  assert.deepEqual(pJsonSchema.choices[0].message.parsed, { wrapped: { ok: true } });

  const pTopParse = await jsonClient.chat.completions.parse({
    messages: [{ role: "user", content: "hi" }],
    response_format: {
      type: "json_schema",
      schema: { type: "object" },
      parse: (d) => ({ top: d }),
    },
  });
  assert.deepEqual(pTopParse.choices[0].message.parsed, { top: { ok: true } });

  const pText = await jsonClient.chat.completions.parse({
    messages: [{ role: "user", content: "hi" }],
    response_format: { type: "text" },
  });
  assert.equal(pText.choices[0].message.parsed, null);

  await assert.rejects(
    () =>
      jsonClient.chat.completions.parse({
        messages: [{ role: "user", content: "hi" }],
        response_format: { type: "json_schema", json_schema: "bad" },
      }),
    TypeError,
  );
  await assert.rejects(
    () =>
      jsonClient.chat.completions.parse({
        messages: [{ role: "user", content: "hi" }],
        response_format: { type: "bogus" },
      }),
    RangeError,
  );
  await assert.rejects(
    () =>
      jsonClient.chat.completions.parse({
        messages: [{ role: "user", content: "hi" }],
        response_format: 123,
      }),
    TypeError,
  );

  // ChatCompletionStream: reasoning + tool_call + structuredJson + length + return/throw/error/asyncDispose
  const ccs = new ChatCompletionStream(
    new TokenStream(
      makeFakeNativeStream(
        ['<think>plan</think><tool_call>{"name":"f","arguments":{}}</tool_call>'],
        { summary: { finishReason: "stop", promptTokens: 1, completionTokens: 2 } },
      ),
    ),
    "id1",
    100,
    "m",
    false,
  );
  const ccsChunks = [];
  for await (const c of ccs) ccsChunks.push(c);
  assert.ok(ccsChunks.some((c) => c.choices[0].delta.reasoning_content === "plan"));
  assert.ok(ccsChunks.some((c) => c.choices[0].delta.tool_calls));
  assert.equal(ccsChunks.at(-1).choices[0].finish_reason, "tool_calls");

  const ccsJson = new ChatCompletionStream(
    new TokenStream(makeFakeNativeStream(['{"a":1}'], { summary: { finishReason: "length" } })),
    "id2",
    100,
    "m",
    true,
  );
  const jsonChunks = [];
  for await (const c of ccsJson) jsonChunks.push(c);
  assert.equal(jsonChunks.at(-1).choices[0].finish_reason, "length");

  const ccsErr = new ChatCompletionStream(
    new TokenStream(makeFakeNativeStream([], { throwOnNext: new Error("stream boom") })),
    "id3",
    100,
    "m",
    true,
  );
  await assert.rejects(async () => {
    for await (const _ of ccsErr) {
      // first chunk is assistant role, second throws
    }
  }, /stream boom/);

  const ccsThrow = new ChatCompletionStream(
    new TokenStream(makeFakeNativeStream(["x"])),
    "id4",
    100,
    "m",
    true,
  );
  const ccsIt = ccsThrow[Symbol.asyncIterator]();
  await assert.rejects(() => ccsIt.throw(new Error("manual")), /manual/);
  assert.equal((await ccsIt.next()).done, true);

  {
    await using _ccsDisp = new ChatCompletionStream(
      new TokenStream(makeFakeNativeStream(["x"])),
      "id5",
      100,
      "m",
      true,
    );
  }

  // ChatCompletionStream early break (iterator return()) + close during pending innerIter.next()
  const ccsBreak = new ChatCompletionStream(
    new TokenStream(makeFakeNativeStream(["a", "b"])),
    "id6",
    100,
    "m",
    true,
  );
  for await (const _ of ccsBreak) {
    break;
  }

  let resolvePendingChat;
  const pendingChatNs = makeFakeNativeStream([]);
  pendingChatNs.nextChunk = () =>
    new Promise((res) => {
      resolvePendingChat = res;
    });
  const ccsPending = new ChatCompletionStream(
    new TokenStream(pendingChatNs),
    "id7",
    100,
    "m",
    true,
  );
  const ccsPendingIt = ccsPending[Symbol.asyncIterator]();
  await ccsPendingIt.next(); // consume initial assistant role chunk
  const ccsNextP = ccsPendingIt.next();
  await ccsPending.close();
  resolvePendingChat("late");
  assert.equal((await ccsNextP).done, true);

  // CompletionStream close during pending innerIter.next()
  let resolvePendingComp;
  const pendingCompNs = makeFakeNativeStream([]);
  pendingCompNs.nextChunk = () =>
    new Promise((res) => {
      resolvePendingComp = res;
    });
  const csPending = new CompletionStream(new TokenStream(pendingCompNs), "c5", 100, "m");
  const csPendingIt = csPending[Symbol.asyncIterator]();
  const csNextP = csPendingIt.next();
  await csPending.close();
  resolvePendingComp("late");
  assert.equal((await csNextP).done, true);

  // CompletionStream: return, throw, error, asyncDispose
  const csErr = new CompletionStream(
    new TokenStream(makeFakeNativeStream([], { throwOnNext: new Error("cs boom") })),
    "c1",
    100,
    "m",
  );
  await assert.rejects(async () => {
    for await (const _ of csErr) {
      // throws immediately
    }
  }, /cs boom/);

  const csThrow = new CompletionStream(
    new TokenStream(makeFakeNativeStream(["a", "b"])),
    "c2",
    100,
    "m",
  );
  const csIt = csThrow[Symbol.asyncIterator]();
  assert.equal((await csIt.next()).value.choices[0].text, "a");
  await assert.rejects(() => csIt.throw(new Error("cs-manual")), /cs-manual/);
  assert.equal((await csIt.next()).done, true);

  const csBreak = new CompletionStream(
    new TokenStream(makeFakeNativeStream(["a", "b"])),
    "c3",
    100,
    "m",
  );
  for await (const _ of csBreak) {
    break;
  }

  {
    await using _csDisp = new CompletionStream(
      new TokenStream(makeFakeNativeStream(["a"])),
      "c4",
      100,
      "m",
    );
  }

  // ModelManager: defaultId alias resolution + unloadAll error propagation
  const origFromPretrained = AutoModel.fromPretrained;
  const failingUnloadBackend = makeFakeBackend();
  failingUnloadBackend.unload = async () => {
    throw new Error("unload fail");
  };
  const failingUnloadModel = new AutoModel(failingUnloadBackend, false);
  const okLazyModel = new AutoModel(makeFakeBackend(), false);
  let callCount = 0;
  AutoModel.fromPretrained = async () => (++callCount === 1 ? okLazyModel : failingUnloadModel);
  try {
    const lazyClient = new OpenAI({ model: "lazy-spec" });
    const res1 = await lazyClient.completions.create({ model: "/fake/m.gguf", prompt: "hi" });
    assert.equal(res1.model, "lazy-spec");
    await lazyClient.completions.create({ model: "second-spec", prompt: "hi" });
    await assert.rejects(() => lazyClient.unload(), /unload fail/);

    await using _dispClient = new OpenAI("third-spec");

    // unloadAll() drains in-flight model load before unloading
    let resolveInFlight;
    AutoModel.fromPretrained = () =>
      new Promise((res) => {
        resolveInFlight = res;
      });
    const inFlightClient = new OpenAI({ model: "in-flight-spec" });
    const reqP = inFlightClient.completions.create({ prompt: "hi" });
    const unloadP = inFlightClient.unload();
    resolveInFlight(new AutoModel(makeFakeBackend(), false));
    await Promise.allSettled([reqP, unloadP]);
  } finally {
    AutoModel.fromPretrained = origFromPretrained;
  }
});
