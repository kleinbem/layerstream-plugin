const path = require("node:path");
const native = require("./native.js");
const { IsolatedWorkerClient } = require("./worker.js");
const { OpenAI } = require("./openai.js");

const ASYNC_DISPOSE = Symbol.asyncDispose;
const MAX_I32 = 2147483647;

function createAbortError(signal) {
  if (signal && signal.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error("The operation was aborted");
  err.name = "AbortError";
  err.code = "ABORT_ERR";
  return err;
}

function mapNativeError(err, signal = undefined) {
  if (signal && signal.aborted) {
    return createAbortError(signal);
  }
  if (err instanceof Error && typeof err.message === "string") {
    if (err.message.startsWith("ABORT_ERR:")) {
      return createAbortError(signal);
    }
    if (err.message.startsWith("ENOENT: ")) {
      const e = new Error(err.message.slice("ENOENT: ".length));
      e.code = "ENOENT";
      return e;
    }
  }
  return err;
}

function isPlainObject(val) {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

function extractJsonSchemaString(schema) {
  if (typeof schema === "string") {
    try {
      JSON.parse(schema);
    } catch (exc) {
      throw new TypeError(`Invalid JSON schema string: ${exc.message}`);
    }
    return schema;
  }
  if (schema && typeof schema === "object") {
    if (typeof schema.toJSONSchema === "function") {
      return JSON.stringify(schema.toJSONSchema());
    }
    if (!Array.isArray(schema)) {
      return JSON.stringify(schema);
    }
  }
  throw new TypeError(
    "jsonSchema must be a JSON Schema object, JSON string, boolean schema, or schema with toJSONSchema()",
  );
}

function jsonSchemaToGbnf(schema) {
  if (typeof schema === "boolean") {
    return native.jsonSchemaToGbnf(JSON.stringify(schema));
  }
  const raw = extractJsonSchemaString(schema);
  return native.jsonSchemaToGbnf(raw);
}

function jsonObjectGbnf() {
  return native.jsonObjectGbnf();
}

function parseAssistantOutput(raw) {
  if (typeof raw !== "string") {
    throw new TypeError("raw assistant text must be a string");
  }
  return native.parseAssistantOutput(raw);
}

function validateTools(tools) {
  if (tools === undefined || tools === null) {
    return undefined;
  }
  if (!Array.isArray(tools)) {
    throw new TypeError("tools must be an array of tool definition objects");
  }
  for (const item of tools) {
    if (!isPlainObject(item)) {
      throw new TypeError("each entry in 'tools' must be a plain object");
    }
  }
  return tools;
}

function formatToolsSystemPrompt(tools) {
  const valid = validateTools(tools);
  return native.formatToolsSystemPrompt(JSON.stringify(valid ?? []));
}

function validateStop(stop) {
  if (stop === undefined || stop === null) {
    return undefined;
  }
  if (typeof stop === "string") {
    return [stop];
  }
  if (Array.isArray(stop) && stop.every((s) => typeof s === "string")) {
    return stop;
  }
  throw new TypeError("stop must be undefined, null, a string, or an array of strings");
}

function validateMessages(messages) {
  if (!Array.isArray(messages)) {
    throw new TypeError("messages must be an array of { role, content } objects");
  }
  if (messages.length === 0) {
    throw new RangeError("messages must not be empty");
  }
  const out = [];
  for (const item of messages) {
    if (!isPlainObject(item)) {
      throw new TypeError("each message must be an object");
    }
    if (typeof item.role !== "string" || item.role.length === 0) {
      throw new TypeError("each message must contain a non-empty 'role' string");
    }
    if (!("content" in item) && !("tool_calls" in item)) {
      throw new TypeError("each message must contain 'content' or 'tool_calls'");
    }
    const clean = {
      role: item.role,
      content: item.content ?? null,
    };
    if (item.tool_calls !== undefined && item.tool_calls !== null) {
      if (!Array.isArray(item.tool_calls)) {
        throw new TypeError("message 'tool_calls' must be an array");
      }
      clean.tool_calls = item.tool_calls;
    }
    if (item.name !== undefined && item.name !== null) {
      clean.name = String(item.name);
    }
    if (item.tool_call_id !== undefined && item.tool_call_id !== null) {
      clean.tool_call_id = String(item.tool_call_id);
    }
    out.push(clean);
  }
  return out;
}

function resolveGrammarOption(grammar, jsonSchema) {
  if (
    grammar !== undefined &&
    grammar !== null &&
    jsonSchema !== undefined &&
    jsonSchema !== null
  ) {
    throw new RangeError("pass at most one of 'grammar' or 'jsonSchema', not both");
  }
  if (grammar !== undefined && grammar !== null) {
    if (typeof grammar !== "string") {
      throw new TypeError("grammar must be a string");
    }
    if (grammar.trim().length === 0) {
      throw new RangeError("grammar must not be empty");
    }
    return grammar;
  }
  if (jsonSchema !== undefined && jsonSchema !== null) {
    return jsonSchemaToGbnf(jsonSchema);
  }
  return undefined;
}

function normalizeOpenOptions(opts = {}) {
  if (!isPlainObject(opts)) {
    throw new TypeError("options must be an object");
  }
  const vramRaw = opts.vramBudgetMb !== undefined ? opts.vramBudgetMb : opts.vram_budget_mb;
  let vramBudgetAuto = true;
  let vramBudgetBytes = 0;
  if (vramRaw !== undefined) {
    if (typeof vramRaw === "boolean") {
      throw new TypeError(
        "vramBudgetMb must be 'auto', null, or a non-negative integer MiB (not boolean)",
      );
    }
    if (vramRaw === null) {
      vramBudgetAuto = false;
      vramBudgetBytes = 0;
    } else if (typeof vramRaw === "string") {
      const trimmed = vramRaw.trim();
      if (trimmed.toLowerCase() === "auto") {
        vramBudgetAuto = true;
        vramBudgetBytes = 0;
      } else {
        const parsed = Number(trimmed);
        if (!Number.isInteger(parsed) || parsed < 0) {
          throw new RangeError(
            `Invalid vramBudgetMb '${trimmed}' (expected 'auto' or non-negative integer MiB)`,
          );
        }
        vramBudgetAuto = false;
        vramBudgetBytes = parsed * 1024 * 1024;
      }
    } else if (typeof vramRaw === "number") {
      if (!Number.isFinite(vramRaw) || vramRaw < 0) {
        throw new RangeError("vramBudgetMb must be non-negative");
      }
      vramBudgetAuto = false;
      vramBudgetBytes = Math.floor(vramRaw * 1024 * 1024);
    } else {
      throw new TypeError("vramBudgetMb must be 'auto', null, or a non-negative number MiB");
    }
  }

  const kvRaw = opts.kvCacheType ?? opts.kv_cache_type ?? "q8_0";
  if (typeof kvRaw !== "string" || !["f16", "q8_0", "q4_0"].includes(kvRaw.trim().toLowerCase())) {
    throw new RangeError(`Invalid kvCacheType '${kvRaw}' (expected 'f16', 'q8_0', or 'q4_0')`);
  }

  const isolatedWorker = Boolean(opts.isolatedWorker ?? opts.isolated_worker ?? false);

  return {
    isolatedWorker,
    wire: {
      draftModel: opts.draftModel ?? opts.draft_model ?? undefined,
      quant: opts.quant ?? undefined,
      revision: opts.revision ?? undefined,
      cacheDir: opts.cacheDir ?? opts.cache_dir ?? undefined,
      vramBudgetAuto,
      vramBudgetBytes,
      residentLayers: opts.residentLayers ?? opts.resident_layers ?? undefined,
      ctxSize: opts.ctxSize ?? opts.ctx_size ?? 4096,
      batchSize: opts.batchSize ?? opts.batch_size ?? 512,
      threads: opts.threads ?? undefined,
      kvCacheType: kvRaw.trim().toLowerCase(),
      useGpu: opts.useGpu ?? opts.use_gpu ?? true,
      preferDirectIo: opts.preferDirectIo ?? opts.prefer_direct_io ?? true,
      cpuEmbeddings: opts.cpuEmbeddings ?? opts.cpu_embeddings ?? false,
      cpuHead: opts.cpuHead ?? opts.cpu_head ?? false,
      streamHead: opts.streamHead ?? opts.stream_head ?? false,
      moeActiveStreaming: opts.moeActiveStreaming ?? opts.moe_active_streaming ?? true,
      maxDraftTokens: opts.maxDraftTokens ?? opts.max_draft_tokens ?? 5,
      verbose: opts.verbose ?? false,
    },
  };
}

class TokenStream {
  constructor(inner, signal = undefined) {
    this._inner = inner;
    this._signal = signal;
    this._closed = false;
    this._onAbort = null;
    if (signal) {
      if (signal.aborted) {
        this._inner.cancel();
      } else {
        this._onAbort = () => {
          this._inner.cancel();
        };
        signal.addEventListener("abort", this._onAbort, { once: true });
      }
    }
  }

  get summary() {
    return this._inner.summary ?? null;
  }

  _detachAbort() {
    if (this._signal && this._onAbort) {
      this._signal.removeEventListener("abort", this._onAbort);
      this._onAbort = null;
    }
  }

  cancel() {
    this._inner.cancel();
  }

  async close() {
    this._detachAbort();
    if (this._closed) {
      return;
    }
    this._closed = true;
    await this._inner.close();
  }

  async [ASYNC_DISPOSE]() {
    await this.close();
  }

  [Symbol.asyncIterator]() {
    return {
      next: async () => {
        if (this._signal && this._signal.aborted) {
          await this.close();
          throw createAbortError(this._signal);
        }
        try {
          const chunk = await this._inner.nextChunk();
          if (this._signal && this._signal.aborted) {
            await this.close();
            throw createAbortError(this._signal);
          }
          if (chunk === null || chunk === undefined) {
            this._detachAbort();
            this._closed = true;
            return { done: true, value: undefined };
          }
          return { done: false, value: chunk };
        } catch (err) {
          await this.close();
          throw mapNativeError(err, this._signal);
        }
      },
      return: async () => {
        await this.close();
        return { done: true, value: undefined };
      },
      throw: async (err) => {
        await this.close();
        throw err;
      },
    };
  }
}

class AutoModel {
  constructor(backend, isolatedWorker) {
    this._backend = backend;
    this._isolated = Boolean(isolatedWorker);
  }

  static async fromPretrained(model, opts = {}) {
    if (typeof model !== "string" || model.trim().length === 0) {
      throw new TypeError("model must be a non-empty string");
    }
    const { isolatedWorker, wire } = normalizeOpenOptions(opts);
    try {
      if (!isolatedWorker) {
        const backend = await native.NativeAutoModel.open(model, JSON.stringify(wire));
        return new AutoModel(backend, false);
      }
      const worker = await IsolatedWorkerClient.create(model, wire);
      return new AutoModel(worker, true);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  _buildGenWire(opts = {}, allowChatExtras = false) {
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const grammar = resolveGrammarOption(
      opts.grammar,
      opts.jsonSchema !== undefined ? opts.jsonSchema : opts.json_schema,
    );
    const stop = validateStop(opts.stop);
    const wire = {
      maxTokens: opts.maxTokens ?? opts.max_tokens ?? 256,
      temperature: opts.temperature ?? 0.0,
      topP: opts.topP ?? opts.top_p ?? 0.95,
      minP: opts.minP ?? opts.min_p ?? 0.0,
      topK: opts.topK ?? opts.top_k ?? 40,
      repeatPenalty: opts.repeatPenalty ?? opts.repeat_penalty ?? 1.0,
      frequencyPenalty: opts.frequencyPenalty ?? opts.frequency_penalty ?? 0.0,
      presencePenalty: opts.presencePenalty ?? opts.presence_penalty ?? 0.0,
      penaltyLastN: opts.penaltyLastN ?? opts.penalty_last_n ?? 64,
      seed: opts.seed ?? 42,
      stop,
      ngramSpeculation: opts.ngramSpeculation ?? opts.ngram_speculation ?? true,
      maxDraftTokens: opts.maxDraftTokens ?? opts.max_draft_tokens ?? undefined,
      grammar,
    };
    if (
      typeof wire.temperature !== "number" ||
      !Number.isFinite(wire.temperature) ||
      wire.temperature < 0
    ) {
      throw new RangeError("temperature must be a finite non-negative number");
    }
    if (
      typeof wire.topP !== "number" ||
      !Number.isFinite(wire.topP) ||
      wire.topP <= 0 ||
      wire.topP > 1
    ) {
      throw new RangeError("top_p must be in (0.0, 1.0]");
    }
    if (
      typeof wire.minP !== "number" ||
      !Number.isFinite(wire.minP) ||
      wire.minP < 0 ||
      wire.minP > 1
    ) {
      throw new RangeError("min_p must be in [0.0, 1.0]");
    }
    if (
      typeof wire.repeatPenalty !== "number" ||
      !Number.isFinite(wire.repeatPenalty) ||
      wire.repeatPenalty <= 0
    ) {
      throw new RangeError("repeat_penalty must be a finite positive number");
    }
    if (typeof wire.frequencyPenalty !== "number" || !Number.isFinite(wire.frequencyPenalty)) {
      throw new RangeError("frequency_penalty must be finite");
    }
    if (typeof wire.presencePenalty !== "number" || !Number.isFinite(wire.presencePenalty)) {
      throw new RangeError("presence_penalty must be finite");
    }
    if (
      typeof wire.penaltyLastN !== "number" ||
      !Number.isInteger(wire.penaltyLastN) ||
      wire.penaltyLastN < -1
    ) {
      throw new RangeError("penalty_last_n must be >= -1");
    }
    if (allowChatExtras) {
      wire.addGenerationPrompt = opts.addGenerationPrompt ?? opts.add_generation_prompt ?? true;
      wire.tools = validateTools(opts.tools);
      wire.parseOutput = Boolean(opts.parseOutput ?? opts.parse_output ?? false);
    }
    return wire;
  }

  async generate(prompt, opts = {}) {
    if (typeof prompt !== "string") {
      throw new TypeError("prompt must be a string");
    }
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }
    const stream = Boolean(opts.stream ?? false);

    if (stream) {
      const wire = this._buildGenWire(opts, false);
      const wireJson = JSON.stringify(wire);
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      try {
        const inner = this._isolated
          ? this._backend.startStream("generate_stream", { prompt, wire })
          : this._backend.generateStream(prompt, wireJson);
        return new TokenStream(inner, signal);
      } catch (err) {
        throw mapNativeError(err, signal);
      }
    }

    const { text } = await this._generateWithSummary(prompt, opts);
    return text;
  }

  async _generateWithSummary(prompt, opts = {}) {
    if (typeof prompt !== "string") {
      throw new TypeError("prompt must be a string");
    }
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }
    const wire = this._buildGenWire(opts, false);
    const wireJson = JSON.stringify(wire);
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }

    if (this._isolated) {
      const { result, summary } = await this._backend.rpcWithSummary(
        "generate",
        { prompt, wire },
        signal,
      );
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      return { text: String(result.text ?? ""), summary };
    }

    const cancelTok = signal ? new native.NativeCancelToken() : undefined;
    const onAbort = cancelTok ? () => cancelTok.cancel() : null;
    if (signal && onAbort) {
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) {
        signal.removeEventListener("abort", onAbort);
        cancelTok.cancel();
        throw createAbortError(signal);
      }
    }
    try {
      const out = await this._backend.generateWithSummary(prompt, wireJson, cancelTok);
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      return { text: out.text, summary: out.summary };
    } catch (err) {
      throw mapNativeError(err, signal);
    } finally {
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
    }
  }

  async chat(messages, opts = {}) {
    const cleanMessages = validateMessages(messages);
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }
    const stream = Boolean(opts.stream ?? false);

    if (stream) {
      const wire = this._buildGenWire(opts, true);
      if (wire.parseOutput) {
        throw new RangeError(
          "parseOutput=true cannot be combined with stream=true (use StreamTagParser or OpenAI)",
        );
      }
      const messagesJson = JSON.stringify(cleanMessages);
      const wireJson = JSON.stringify(wire);
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }

      if (this._isolated) {
        try {
          const inner = this._backend.startStream("chat_stream", {
            messages: cleanMessages,
            wire,
          });
          return new TokenStream(inner, signal);
        } catch (err) {
          throw mapNativeError(err, signal);
        }
      }

      const cancelTok = signal ? new native.NativeCancelToken() : undefined;
      let abortReject = null;
      const onAbort = signal
        ? () => {
            if (cancelTok) cancelTok.cancel();
            if (abortReject) abortReject(createAbortError(signal));
          }
        : null;
      const abortPromise =
        signal && onAbort
          ? new Promise((_resolve, reject) => {
              abortReject = reject;
              signal.addEventListener("abort", onAbort, { once: true });
              if (signal.aborted) {
                if (cancelTok) cancelTok.cancel();
                reject(createAbortError(signal));
              }
            })
          : null;

      try {
        const startPromise = this._backend.chatStream(messagesJson, wireJson, cancelTok);
        let inner;
        if (abortPromise) {
          startPromise.then(
            (lateStream) => {
              if (signal.aborted && lateStream) {
                void lateStream.close().catch(() => {});
              }
            },
            () => {},
          );
          inner = await Promise.race([startPromise, abortPromise]);
        } else {
          inner = await startPromise;
        }
        if (signal && signal.aborted) {
          await inner.close();
          throw createAbortError(signal);
        }
        return new TokenStream(inner, signal);
      } catch (err) {
        throw mapNativeError(err, signal);
      } finally {
        if (signal && onAbort) {
          signal.removeEventListener("abort", onAbort);
        }
      }
    }

    const { payload } = await this._chatWithSummary(cleanMessages, opts, true);
    return payload;
  }

  async _chatWithSummary(messages, opts = {}, alreadyValidated = false) {
    const cleanMessages = alreadyValidated ? messages : validateMessages(messages);
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }
    const wire = this._buildGenWire(opts, true);
    const messagesJson = JSON.stringify(cleanMessages);
    const wireJson = JSON.stringify(wire);
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }

    if (this._isolated) {
      const { result, summary } = await this._backend.rpcWithSummary(
        "chat",
        { messages: cleanMessages, wire },
        signal,
      );
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      const payload = wire.parseOutput ? result.parsed : String(result.text ?? "");
      return { payload, summary };
    }

    const cancelTok = signal ? new native.NativeCancelToken() : undefined;
    const onAbort = cancelTok ? () => cancelTok.cancel() : null;
    if (signal && onAbort) {
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) {
        signal.removeEventListener("abort", onAbort);
        cancelTok.cancel();
        throw createAbortError(signal);
      }
    }
    try {
      const out = await this._backend.chatWithSummary(messagesJson, wireJson, cancelTok);
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      const payload = wire.parseOutput ? out.parsed : out.text;
      return { payload, summary: out.summary };
    } catch (err) {
      throw mapNativeError(err, signal);
    } finally {
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
    }
  }

  async tokenize(text, opts = {}) {
    if (typeof text !== "string") {
      throw new TypeError("text must be a string");
    }
    const addSpecial = Boolean(opts.addSpecial ?? opts.add_special ?? false);
    const parseSpecial = Boolean(opts.parseSpecial ?? opts.parse_special ?? false);
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("tokenize", {
          text,
          addSpecial,
          parseSpecial,
        });
        return res.tokens;
      }
      return await this._backend.tokenize(text, addSpecial, parseSpecial);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async detokenize(tokens) {
    if (!Array.isArray(tokens) || !tokens.every((t) => Number.isInteger(t))) {
      throw new TypeError("tokens must be an array of integer token IDs");
    }
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("detokenize", { tokens });
        return res.text;
      }
      return await this._backend.detokenize(tokens);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async applyChatTemplate(messages, opts = {}) {
    const cleanMessages = validateMessages(messages);
    const addGen = Boolean(opts.addGenerationPrompt ?? opts.add_generation_prompt ?? true);
    const tools = validateTools(opts.tools);
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("apply_chat_template", {
          messages: cleanMessages,
          addGenerationPrompt: addGen,
          tools: tools ?? null,
        });
        return res.prompt;
      }
      return await this._backend.applyChatTemplate(
        JSON.stringify(cleanMessages),
        addGen,
        tools ? JSON.stringify(tools) : undefined,
      );
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async kvClear() {
    try {
      if (this._isolated) {
        await this._backend.rpc("kv_clear", {});
        return;
      }
      await this._backend.kvClear();
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async kvTrim(posMin) {
    if (!Number.isInteger(posMin) || posMin < 0 || posMin > MAX_I32) {
      throw new RangeError("posMin must be a non-negative 32-bit integer");
    }
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("kv_trim", { posMin });
        return Boolean(res.trimmed);
      }
      return await this._backend.kvTrim(posMin);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async saveState(statePath) {
    if (typeof statePath !== "string" || statePath.trim().length === 0) {
      throw new TypeError("path must be a non-empty string");
    }
    const resolvedPath = path.resolve(statePath);
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("save_state", { path: resolvedPath });
        return Number(res.tokens ?? 0);
      }
      return await this._backend.saveState(resolvedPath);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async loadState(statePath) {
    if (typeof statePath !== "string" || statePath.trim().length === 0) {
      throw new TypeError("path must be a non-empty string");
    }
    const resolvedPath = path.resolve(statePath);
    try {
      if (this._isolated) {
        const res = await this._backend.rpc("load_state", { path: resolvedPath });
        return Number(res.tokens ?? 0);
      }
      return await this._backend.loadState(resolvedPath);
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  async stats() {
    try {
      return await this._backend.stats();
    } catch (err) {
      throw mapNativeError(err);
    }
  }

  get isLoaded() {
    return this._backend.isLoaded;
  }

  get isolatedWorker() {
    return this._isolated;
  }

  get workerPid() {
    return this._isolated ? this._backend.workerPid : null;
  }

  get modelPath() {
    return this._backend.modelPath ?? null;
  }

  get lastSummary() {
    return this._backend.lastSummary ?? null;
  }

  async unload() {
    await this._backend.unload();
  }

  async [ASYNC_DISPOSE]() {
    await this.unload();
  }
}

async function probeGpuMemory(useGpu = true) {
  return native.probeGpuMemory(Boolean(useGpu));
}

async function inspectGguf(modelPath) {
  if (typeof modelPath !== "string" || modelPath.length === 0) {
    throw new TypeError("path must be a non-empty string");
  }
  try {
    return await native.inspectGguf(modelPath);
  } catch (err) {
    throw mapNativeError(err);
  }
}

async function pull(spec, opts = {}) {
  if (typeof spec !== "string" || spec.length === 0) {
    throw new TypeError("spec must be a non-empty string");
  }
  const cacheDir = opts?.cacheDir ?? opts?.cache_dir ?? undefined;
  try {
    return await native.pull(spec, cacheDir);
  } catch (err) {
    throw mapNativeError(err);
  }
}

async function recommend(repo, opts = {}) {
  if (typeof repo !== "string" || repo.length === 0) {
    throw new TypeError("repo must be a non-empty string");
  }
  try {
    return await native.recommend(repo, {
      revision: opts.revision,
      freeRamGb: opts.freeRamGb ?? opts.free_ram_gb,
      nvmeMibS: opts.nvmeMibS ?? opts.nvme_mib_s,
      hasDraft: opts.hasDraft ?? opts.has_draft,
      cacheDir: opts.cacheDir ?? opts.cache_dir,
      useGpu: opts.useGpu ?? opts.use_gpu,
    });
  } catch (err) {
    throw mapNativeError(err);
  }
}

async function check(model, opts = {}) {
  if (typeof model !== "string" || model.length === 0) {
    throw new TypeError("model must be a non-empty string");
  }
  try {
    return await native.check(model, {
      draftModel: opts.draftModel ?? opts.draft_model,
      quant: opts.quant,
      revision: opts.revision,
      cacheDir: opts.cacheDir ?? opts.cache_dir,
      useGpu: opts.useGpu ?? opts.use_gpu,
      allowSlow: opts.allowSlow ?? opts.allow_slow,
    });
  } catch (err) {
    throw mapNativeError(err);
  }
}

const version = native.version();

module.exports = {
  AutoModel,
  OpenAI,
  StreamTagParser: native.StreamTagParser,
  TokenStream,
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
};
