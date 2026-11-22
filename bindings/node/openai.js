"use strict";

const crypto = require("node:crypto");
const native = require("./native.js");

const ASYNC_DISPOSE = Symbol.asyncDispose || Symbol.for("nodejs.dispose.async");

function isPlainObject(val) {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

function usageFromSummary(summary) {
  if (!summary) return null;
  const p = Number(summary.promptTokens ?? summary.prompt_tokens ?? 0);
  const c = Number(summary.completionTokens ?? summary.completion_tokens ?? 0);
  return {
    prompt_tokens: p,
    completion_tokens: c,
    total_tokens: p + c,
  };
}

function compileResponseFormat(responseFormat) {
  if (responseFormat === undefined || responseFormat === null) {
    return { grammar: undefined, parser: null };
  }

  // Zod v4 (`toJSONSchema()`) or custom schema object with `.parse(data)` + `.toJSONSchema()`
  if (typeof responseFormat === "object" && typeof responseFormat.toJSONSchema === "function") {
    const schemaObj = responseFormat.toJSONSchema();
    const grammar = native.jsonSchemaToGbnf(JSON.stringify(schemaObj));
    const parser =
      typeof responseFormat.parse === "function"
        ? (raw) => responseFormat.parse(JSON.parse(raw))
        : (raw) => JSON.parse(raw);
    return { grammar, parser };
  }

  if (isPlainObject(responseFormat)) {
    const rfType = responseFormat.type;
    if (rfType === "text") {
      return { grammar: undefined, parser: null };
    }
    if (rfType === "json_object") {
      return { grammar: native.jsonObjectGbnf(), parser: (raw) => JSON.parse(raw) };
    }
    if (rfType === "json_schema") {
      const inner = isPlainObject(responseFormat.json_schema)
        ? responseFormat.json_schema
        : responseFormat;
      const schema = inner.schema !== undefined ? inner.schema : inner;
      if (schema === undefined || schema === null) {
        throw new RangeError("response_format 'json_schema' requires a 'schema' object");
      }
      const validator =
        typeof inner.parse === "function"
          ? (raw) => inner.parse(JSON.parse(raw))
          : (raw) => JSON.parse(raw);
      return {
        grammar: native.jsonSchemaToGbnf(JSON.stringify(schema)),
        parser: validator,
      };
    }
    if (rfType === "gbnf") {
      const g = responseFormat.grammar;
      if (typeof g !== "string" || g.trim().length === 0) {
        throw new RangeError("response_format 'gbnf' requires a non-empty 'grammar' string");
      }
      return { grammar: g, parser: null };
    }
    // Direct JSON Schema object (`{ type: "object", properties: ... }`) passed to `.parse(...)`
    if (
      rfType === "object" ||
      rfType === "array" ||
      "properties" in responseFormat ||
      "anyOf" in responseFormat ||
      "oneOf" in responseFormat ||
      "allOf" in responseFormat ||
      "$ref" in responseFormat
    ) {
      return {
        grammar: native.jsonSchemaToGbnf(JSON.stringify(responseFormat)),
        parser: (raw) => JSON.parse(raw),
      };
    }
    throw new RangeError(
      `Unsupported response_format type '${rfType}' (expected 'text', 'json_object', 'json_schema', or 'gbnf')`
    );
  }

  throw new TypeError(
    "response_format must be undefined, a response_format object, a JSON Schema, or a schema with toJSONSchema()"
  );
}

class ChatCompletionStream {
  constructor(tokenStream, reqId, created, modelId, structuredJson) {
    this._stream = tokenStream;
    this._id = reqId;
    this._created = created;
    this._modelId = modelId;
    this._structuredJson = Boolean(structuredJson);
    this._tagParser = this._structuredJson ? null : new native.StreamTagParser();
    this._pending = [
      {
        id: reqId,
        object: "chat.completion.chunk",
        created,
        model: modelId,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null, logprobs: null }],
        usage: null,
      },
    ];
    this._finished = false;
  }

  async close() {
    this._finished = true;
    await this._stream.close();
  }

  async [ASYNC_DISPOSE]() {
    await this.close();
  }

  _enqueueDeltas(deltas) {
    for (const ev of deltas) {
      if (ev.type === "content") {
        this._pending.push({
          id: this._id,
          object: "chat.completion.chunk",
          created: this._created,
          model: this._modelId,
          choices: [
            { index: 0, delta: { content: ev.text ?? "" }, finish_reason: null, logprobs: null },
          ],
          usage: null,
        });
      } else if (ev.type === "reasoning") {
        this._pending.push({
          id: this._id,
          object: "chat.completion.chunk",
          created: this._created,
          model: this._modelId,
          choices: [
            {
              index: 0,
              delta: { reasoning_content: ev.text ?? "" },
              finish_reason: null,
              logprobs: null,
            },
          ],
          usage: null,
        });
      } else if (ev.type === "tool_call") {
        this._pending.push({
          id: this._id,
          object: "chat.completion.chunk",
          created: this._created,
          model: this._modelId,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: ev.index ?? 0,
                    id: ev.id ?? "call_ls_1",
                    type: "function",
                    function: {
                      name: ev.name ?? "",
                      arguments: ev.arguments ?? "{}",
                    },
                  },
                ],
              },
              finish_reason: null,
              logprobs: null,
            },
          ],
          usage: null,
        });
      }
    }
  }

  [Symbol.asyncIterator]() {
    const innerIter = this._stream[Symbol.asyncIterator]();
    return {
      next: async () => {
        try {
          while (this._pending.length === 0) {
            if (this._finished) {
              return { done: true, value: undefined };
            }
            const step = await innerIter.next();
            if (step.done) {
              this._finished = true;
              if (this._tagParser) {
                this._enqueueDeltas(this._tagParser.flush());
              }
              const summary = this._stream.summary;
              const rawReason = summary?.finishReason ?? "stop";
              let finishReason = "stop";
              if (this._tagParser && this._tagParser.hasToolCalls) {
                finishReason = "tool_calls";
              } else if (rawReason === "length") {
                finishReason = "length";
              }
              this._pending.push({
                id: this._id,
                object: "chat.completion.chunk",
                created: this._created,
                model: this._modelId,
                choices: [{ index: 0, delta: {}, finish_reason: finishReason, logprobs: null }],
                usage: usageFromSummary(summary),
              });
              break;
            }
            const piece = step.value;
            if (!this._tagParser) {
              if (piece) {
                this._pending.push({
                  id: this._id,
                  object: "chat.completion.chunk",
                  created: this._created,
                  model: this._modelId,
                  choices: [{ index: 0, delta: { content: piece }, finish_reason: null, logprobs: null }],
                  usage: null,
                });
              }
            } else {
              this._enqueueDeltas(this._tagParser.push(piece));
            }
          }
          if (this._pending.length === 0) {
            return { done: true, value: undefined };
          }
          return { done: false, value: this._pending.shift() };
        } catch (err) {
          await this.close();
          throw err;
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

class CompletionStream {
  constructor(tokenStream, reqId, created, modelId) {
    this._stream = tokenStream;
    this._id = reqId;
    this._created = created;
    this._modelId = modelId;
    this._finished = false;
  }

  async close() {
    this._finished = true;
    await this._stream.close();
  }

  async [ASYNC_DISPOSE]() {
    await this.close();
  }

  [Symbol.asyncIterator]() {
    const innerIter = this._stream[Symbol.asyncIterator]();
    return {
      next: async () => {
        if (this._finished) {
          return { done: true, value: undefined };
        }
        try {
          const step = await innerIter.next();
          if (step.done) {
            this._finished = true;
            const summary = this._stream.summary;
            const reason = summary?.finishReason ?? "stop";
            return {
              done: false,
              value: {
                id: this._id,
                object: "text_completion",
                created: this._created,
                model: this._modelId,
                choices: [{ text: "", index: 0, finish_reason: reason, logprobs: null }],
                usage: usageFromSummary(summary),
              },
            };
          }
          return {
            done: false,
            value: {
              id: this._id,
              object: "text_completion",
              created: this._created,
              model: this._modelId,
              choices: [{ text: step.value, index: 0, finish_reason: null, logprobs: null }],
              usage: null,
            },
          };
        } catch (err) {
          await this.close();
          throw err;
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

class ModelManager {
  constructor(initialModel, openOptions) {
    this._initialModel = initialModel;
    this._openOptions = openOptions;
    this._models = new Map();
    this._owned = new Set();
    this._defaultId = null;
    this._initPromise = null;

    if (
      initialModel &&
      typeof initialModel === "object" &&
      typeof initialModel.generate === "function" &&
      typeof initialModel.chat === "function"
    ) {
      const id = initialModel.modelPath || "default";
      this._models.set(id, initialModel);
      this._defaultId = id;
      this._initialModel = null;
    }
  }

  async _ensureInitialLoaded() {
    if (!this._initialModel) return;
    if (!this._initPromise) {
      const spec = this._initialModel;
      this._initPromise = (async () => {
        const { AutoModel } = require("./index.js");
        const loaded = await AutoModel.fromPretrained(spec, this._openOptions);
        this._models.set(spec, loaded);
        this._owned.add(spec);
        this._defaultId = spec;
        this._initialModel = null;
      })();
    }
    await this._initPromise;
  }

  async resolve(requestedModel) {
    await this._ensureInitialLoaded();
    const key = requestedModel || this._defaultId;
    if (!key) {
      throw new RangeError(
        "No model specified; pass `model` to new OpenAI({ model }) or `create({ model })`"
      );
    }
    const existing = this._models.get(key);
    if (existing && existing.isLoaded) {
      return { modelId: key, engine: existing };
    }
    if (!existing && this._models.size === 1 && this._defaultId) {
      const def = this._models.get(this._defaultId);
      if (def && def.isLoaded && (!requestedModel || requestedModel === def.modelPath)) {
        return { modelId: this._defaultId, engine: def };
      }
    }
    const { AutoModel } = require("./index.js");
    const loaded = await AutoModel.fromPretrained(key, this._openOptions);
    this._models.set(key, loaded);
    this._owned.add(key);
    if (!this._defaultId) {
      this._defaultId = key;
    }
    return { modelId: key, engine: loaded };
  }

  async listModels() {
    await this._ensureInitialLoaded();
    const data = [];
    for (const [id, inst] of this._models.entries()) {
      if (inst.isLoaded) {
        data.push({ id, object: "model", created: 0, owned_by: "layerstream" });
      }
    }
    return { object: "list", data };
  }

  async unloadAll() {
    if (this._initPromise) {
      try {
        await this._initPromise;
      } catch {}
    }
    const ownedInstances = [];
    for (const key of this._owned) {
      const inst = this._models.get(key);
      if (inst) ownedInstances.push(inst);
      this._models.delete(key);
    }
    this._owned.clear();
    for (const inst of ownedInstances) {
      await inst.unload();
    }
  }
}

class ChatCompletionsResource {
  constructor(mgr) {
    this._mgr = mgr;
  }

  async create(params = {}, callOpts = {}) {
    const { modelId, engine } = await this._mgr.resolve(params.model);
    const maxTokens =
      params.max_completion_tokens ?? params.maxCompletionTokens ?? params.max_tokens ?? params.maxTokens ?? 512;
    const { grammar } = compileResponseFormat(params.response_format ?? params.responseFormat);
    const signal = params.signal ?? callOpts?.signal;
    const reqId = `chatcmpl-ls-${crypto.randomBytes(8).toString("hex")}`;
    const created = Math.floor(Date.now() / 1000);

    const genOpts = {
      maxTokens,
      temperature: params.temperature ?? 0.0,
      topP: params.top_p ?? params.topP ?? 0.95,
      topK: params.top_k ?? params.topK ?? 40,
      seed: params.seed ?? 42,
      stop: params.stop,
      tools: params.tools,
      grammar,
      ngramSpeculation: params.ngram_speculation ?? params.ngramSpeculation ?? true,
      maxDraftTokens: params.max_draft_tokens ?? params.maxDraftTokens,
      signal,
    };

    if (params.stream) {
      const tokStream = await engine.chat(params.messages, { ...genOpts, stream: true });
      return new ChatCompletionStream(
        tokStream,
        reqId,
        created,
        modelId,
        grammar !== undefined
      );
    }

    let raw;
    let summary;
    if (typeof engine._chatWithSummary === "function") {
      const out = await engine._chatWithSummary(params.messages, genOpts);
      raw = out.payload;
      summary = out.summary;
    } else {
      raw = await engine.chat(params.messages, { ...genOpts, stream: false });
      summary = engine.lastSummary ?? null;
    }

    const rawReason = summary?.finishReason ?? "stop";
    let message;
    let finishReason = "stop";

    if (grammar !== undefined) {
      message = {
        role: "assistant",
        content: String(raw),
        reasoning_content: null,
        tool_calls: null,
        parsed: null,
      };
      finishReason = rawReason === "length" ? "length" : "stop";
    } else {
      const parsedOut = native.parseAssistantOutput(String(raw));
      const tcs = (parsedOut.toolCalls || []).map((tc) => ({
        id: tc.id,
        type: tc.type,
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments,
        },
      }));
      message = {
        role: "assistant",
        content: parsedOut.content || tcs.length === 0 ? parsedOut.content : null,
        reasoning_content: parsedOut.reasoningContent ?? null,
        tool_calls: tcs.length > 0 ? tcs : null,
        parsed: null,
      };
      if (tcs.length > 0) {
        finishReason = "tool_calls";
      } else if (rawReason === "length") {
        finishReason = "length";
      }
    }

    return {
      id: reqId,
      object: "chat.completion",
      created,
      model: modelId,
      choices: [{ index: 0, message, finish_reason: finishReason, logprobs: null }],
      usage: usageFromSummary(summary),
    };
  }

  async parse(params = {}, callOpts = {}) {
    const rf = params.response_format ?? params.responseFormat;
    const { parser } = compileResponseFormat(rf);
    const resp = await this.create({ ...params, stream: false }, callOpts);
    const rawContent = resp.choices[0].message.content ?? "";
    resp.choices[0].message.parsed = parser ? parser(rawContent) : null;
    return resp;
  }
}

class CompletionsResource {
  constructor(mgr) {
    this._mgr = mgr;
  }

  async create(params = {}, callOpts = {}) {
    const { modelId, engine } = await this._mgr.resolve(params.model);
    const signal = params.signal ?? callOpts?.signal;
    const reqId = `cmpl-ls-${crypto.randomBytes(8).toString("hex")}`;
    const created = Math.floor(Date.now() / 1000);

    const genOpts = {
      maxTokens: params.max_tokens ?? params.maxTokens ?? 256,
      temperature: params.temperature ?? 0.0,
      topP: params.top_p ?? params.topP ?? 0.95,
      topK: params.top_k ?? params.topK ?? 40,
      seed: params.seed ?? 42,
      stop: params.stop,
      grammar: params.grammar,
      jsonSchema: params.json_schema ?? params.jsonSchema,
      ngramSpeculation: params.ngram_speculation ?? params.ngramSpeculation ?? true,
      maxDraftTokens: params.max_draft_tokens ?? params.maxDraftTokens,
      signal,
    };

    if (params.stream) {
      const tokStream = await engine.generate(params.prompt, { ...genOpts, stream: true });
      return new CompletionStream(tokStream, reqId, created, modelId);
    }

    let text;
    let summary;
    if (typeof engine._generateWithSummary === "function") {
      const out = await engine._generateWithSummary(params.prompt, genOpts);
      text = out.text;
      summary = out.summary;
    } else {
      text = await engine.generate(params.prompt, { ...genOpts, stream: false });
      summary = engine.lastSummary ?? null;
    }

    const reason = summary?.finishReason ?? "stop";
    return {
      id: reqId,
      object: "text_completion",
      created,
      model: modelId,
      choices: [{ text, index: 0, finish_reason: reason, logprobs: null }],
      usage: usageFromSummary(summary),
    };
  }
}

class OpenAI {
  constructor(modelOrOptions = undefined, extraOptions = undefined) {
    let initialModel = null;
    let rawOpts = {};
    if (typeof modelOrOptions === "string") {
      initialModel = modelOrOptions;
      rawOpts = isPlainObject(extraOptions) ? { ...extraOptions } : {};
    } else if (
      modelOrOptions &&
      typeof modelOrOptions === "object" &&
      typeof modelOrOptions.generate === "function" &&
      typeof modelOrOptions.chat === "function"
    ) {
      initialModel = modelOrOptions;
      rawOpts = isPlainObject(extraOptions) ? { ...extraOptions } : {};
    } else if (isPlainObject(modelOrOptions)) {
      rawOpts = { ...modelOrOptions };
      initialModel = rawOpts.model ?? null;
      delete rawOpts.model;
    } else if (modelOrOptions !== undefined && modelOrOptions !== null) {
      throw new TypeError(
        "OpenAI constructor expects an options object, model spec string, or AutoModel instance"
      );
    }

    delete rawOpts.apiKey;
    delete rawOpts.api_key;
    delete rawOpts.baseURL;
    delete rawOpts.baseUrl;
    delete rawOpts.base_url;

    this._mgr = new ModelManager(initialModel, rawOpts);
    this.chat = { completions: new ChatCompletionsResource(this._mgr) };
    this.completions = new CompletionsResource(this._mgr);
    this.models = { list: () => this._mgr.listModels() };
  }

  async unload() {
    await this._mgr.unloadAll();
  }

  async close() {
    await this.unload();
  }

  async [ASYNC_DISPOSE]() {
    await this.unload();
  }
}

module.exports = {
  ChatCompletionStream,
  CompletionStream,
  OpenAI,
};
