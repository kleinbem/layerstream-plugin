import path from "node:path";
import {
  ASYNC_DISPOSE,
  createAbortError,
  isPlainObject,
  stringifyOptions,
  throwBridgeError,
  throwIfAborted,
} from "./_shared.js";
import native from "./native.js";
import { spawnWorker } from "./worker.js";

const MAX_I32 = 2147483647;

function mapNativeError(err, signal = undefined) {
  if (signal?.aborted) {
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

function validateMessages(messages) {
  if (!Array.isArray(messages)) {
    throw new TypeError("messages must be an array of { role, content } objects");
  }
  try {
    return JSON.parse(native.normalizeChatMessagesJson(JSON.stringify(messages)));
  } catch (err) {
    throwBridgeError(err);
  }
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
  try {
    return JSON.parse(native.normalizeOpenOptionsJson(stringifyOptions(opts)));
  } catch (err) {
    throwBridgeError(err);
  }
}

async function withCancelToken(signal, fn) {
  const cancelTok = signal ? new native.NativeCancelToken() : undefined;
  const onAbort = cancelTok ? () => cancelTok.cancel() : null;
  if (signal && onAbort) {
    signal.addEventListener("abort", onAbort, { once: true });
  }
  try {
    const out = await fn(cancelTok);
    throwIfAborted(signal);
    return out;
  } catch (err) {
    throw mapNativeError(err, signal);
  } finally {
    if (signal && onAbort) {
      signal.removeEventListener("abort", onAbort);
    }
  }
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
        if (this._signal?.aborted) {
          await this.close();
          throw createAbortError(this._signal);
        }
        try {
          const chunk = await this._inner.nextChunk();
          if (this._signal?.aborted) {
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
      const worker = await spawnWorker(model, wire);
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
    try {
      return JSON.parse(
        native.normalizeGenOptionsJson(stringifyOptions(opts), grammar, allowChatExtras),
      );
    } catch (err) {
      throwBridgeError(err);
    }
  }

  async generate(prompt, opts = {}) {
    if (typeof prompt !== "string") {
      throw new TypeError("prompt must be a string");
    }
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    if (signal?.aborted) {
      throw createAbortError(signal);
    }
    const stream = Boolean(opts.stream ?? false);

    if (stream) {
      const wire = this._buildGenWire(opts, false);
      const wireJson = JSON.stringify(wire);
      throwIfAborted(signal);
      try {
        const inner = this._backend.generateStream(prompt, wireJson);
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
    throwIfAborted(signal);
    const wire = this._buildGenWire(opts, false);
    const wireJson = JSON.stringify(wire);
    throwIfAborted(signal);

    if (this._isolated) {
      try {
        const out = await this._backend.generateWithSummary(prompt, wireJson);
        throwIfAborted(signal);
        return { text: String(out.text ?? ""), summary: out.summary ?? null };
      } catch (err) {
        throw mapNativeError(err, signal);
      }
    }

    const out = await withCancelToken(signal, (cancelTok) =>
      this._backend.generateWithSummary(prompt, wireJson, cancelTok),
    );
    return { text: out.text, summary: out.summary };
  }

  async chat(messages, opts = {}) {
    const cleanMessages = validateMessages(messages);
    if (!isPlainObject(opts)) {
      throw new TypeError("generation options must be an object");
    }
    const signal = opts.signal;
    throwIfAborted(signal);
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

      if (this._isolated) {
        try {
          const inner = this._backend.chatStream(messagesJson, wireJson);
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
    throwIfAborted(signal);
    const wire = this._buildGenWire(opts, true);
    const messagesJson = JSON.stringify(cleanMessages);
    const wireJson = JSON.stringify(wire);

    if (this._isolated) {
      try {
        const out = await this._backend.chatWithSummary(messagesJson, wireJson);
        throwIfAborted(signal);
        const payload = wire.parseOutput ? (out.parsed ?? null) : String(out.text ?? "");
        return { payload, summary: out.summary ?? null };
      } catch (err) {
        throw mapNativeError(err, signal);
      }
    }

    const out = await withCancelToken(signal, (cancelTok) =>
      this._backend.chatWithSummary(messagesJson, wireJson, cancelTok),
    );
    const payload = wire.parseOutput ? out.parsed : out.text;
    return { payload, summary: out.summary };
  }

  async tokenize(text, opts = {}) {
    if (typeof text !== "string") {
      throw new TypeError("text must be a string");
    }
    const addSpecial = Boolean(opts.addSpecial ?? opts.add_special ?? false);
    const parseSpecial = Boolean(opts.parseSpecial ?? opts.parse_special ?? false);
    try {
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

const StreamTagParser = native.StreamTagParser;
const version = native.version();

export {
  AutoModel,
  check,
  formatToolsSystemPrompt,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  recommend,
  StreamTagParser,
  TokenStream,
  version,
};
