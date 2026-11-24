const crypto = require("node:crypto");
const native = require("./native.js");

const ASYNC_DISPOSE = Symbol.asyncDispose || Symbol.for("nodejs.dispose.async");

function isPlainObject(val) {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

function createAbortError(signal) {
  if (signal && signal.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error("The operation was aborted");
  err.name = "AbortError";
  err.code = "ABORT_ERR";
  return err;
}

function throwIfAborted(signal) {
  if (signal && signal.aborted) {
    throw createAbortError(signal);
  }
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

  if (typeof responseFormat === "boolean") {
    return {
      grammar: native.jsonSchemaToGbnf(JSON.stringify(responseFormat)),
      parser: (raw) => JSON.parse(raw),
    };
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
    let validator = (raw) => JSON.parse(raw);
    if (responseFormat.type === "json_schema") {
      if (responseFormat.json_schema !== undefined) {
        if (!isPlainObject(responseFormat.json_schema)) {
          throw new TypeError("response_format.json_schema must be an object");
        }
        if (typeof responseFormat.json_schema.parse === "function") {
          const fn = responseFormat.json_schema.parse.bind(responseFormat.json_schema);
          validator = (raw) => fn(JSON.parse(raw));
        }
      } else if (typeof responseFormat.parse === "function") {
        const fn = responseFormat.parse.bind(responseFormat);
        validator = (raw) => fn(JSON.parse(raw));
      }
    }
    let compiled;
    try {
      compiled = native.compileResponseFormat(JSON.stringify(responseFormat), true);
    } catch (err) {
      throw new RangeError(err?.message || String(err));
    }
    return {
      grammar: compiled.grammar ?? undefined,
      parser: compiled.expectsJson ? validator : null,
    };
  }

  throw new TypeError(
    "response_format must be undefined, a response_format object, a JSON Schema, or a schema with toJSONSchema()",
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
    this._closed = false;
  }

  async close() {
    this._closed = true;
    this._finished = true;
    this._pending.length = 0;
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
        if (this._closed) {
          return { done: true, value: undefined };
        }
        try {
          while (this._pending.length === 0) {
            if (this._closed || this._finished) {
              return { done: true, value: undefined };
            }
            const step = await innerIter.next();
            if (this._closed) {
              this._pending.length = 0;
              return { done: true, value: undefined };
            }
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
                  choices: [
                    { index: 0, delta: { content: piece }, finish_reason: null, logprobs: null },
                  ],
                  usage: null,
                });
              }
            } else {
              this._enqueueDeltas(this._tagParser.push(piece));
            }
          }
          if (this._closed || this._pending.length === 0) {
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
    this._closed = false;
  }

  async close() {
    this._closed = true;
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
        if (this._closed || this._finished) {
          return { done: true, value: undefined };
        }
        try {
          const step = await innerIter.next();
          if (this._closed) {
            return { done: true, value: undefined };
          }
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
    this._loadingModels = new Map();
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
      const p = (async () => {
        try {
          const { AutoModel } = require("./index.js");
          const loaded = await AutoModel.fromPretrained(spec, this._openOptions);
          this._models.set(spec, loaded);
          this._owned.add(spec);
          if (!this._defaultId) {
            this._defaultId = spec;
          }
          this._initialModel = null;
        } finally {
          if (this._initPromise === p) {
            this._initPromise = null;
          }
        }
      })();
      this._initPromise = p;
    }
    await this._initPromise;
  }

  async resolve(requestedModel) {
    await this._ensureInitialLoaded();
    const key = requestedModel || this._defaultId;
    if (!key) {
      throw new RangeError(
        "No model specified; pass `model` to new OpenAI({ model }) or `create({ model })`",
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

    let inFlight = this._loadingModels.get(key);
    if (!inFlight) {
      const p = (async () => {
        try {
          const { AutoModel } = require("./index.js");
          const loaded = await AutoModel.fromPretrained(key, this._openOptions);
          this._models.set(key, loaded);
          this._owned.add(key);
          if (!this._defaultId) {
            this._defaultId = key;
          }
          return loaded;
        } finally {
          if (this._loadingModels.get(key) === p) {
            this._loadingModels.delete(key);
          }
        }
      })();
      this._loadingModels.set(key, p);
      inFlight = p;
    }

    const loaded = await inFlight;
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
    while (this._initPromise || this._loadingModels.size > 0) {
      const pending = [this._initPromise, ...this._loadingModels.values()].filter(Boolean);
      await Promise.allSettled(pending);
    }

    let firstError = null;
    for (const key of [...this._owned]) {
      try {
        const inst = this._models.get(key);
        if (inst) {
          await inst.unload();
        }
        this._models.delete(key);
        this._owned.delete(key);
        if (this._defaultId === key) {
          this._defaultId = null;
        }
      } catch (err) {
        firstError ??= err;
      }
    }
    if (firstError) {
      throw firstError;
    }
  }
}

class ChatCompletionsResource {
  constructor(mgr) {
    this._mgr = mgr;
  }

  async create(params = {}, callOpts = {}) {
    return this._createInternal(params, callOpts, null);
  }

  async _createInternal(params = {}, callOpts = {}, precompiledFormat = null) {
    const signal = params.signal ?? callOpts?.signal;
    throwIfAborted(signal);

    const { grammar } =
      precompiledFormat ?? compileResponseFormat(params.response_format ?? params.responseFormat);
    throwIfAborted(signal);

    const { modelId, engine } = await this._mgr.resolve(params.model);
    throwIfAborted(signal);

    const maxTokens =
      params.max_completion_tokens ??
      params.maxCompletionTokens ??
      params.max_tokens ??
      params.maxTokens ??
      512;
    const reqId = `chatcmpl-ls-${crypto.randomBytes(8).toString("hex")}`;
    const created = Math.floor(Date.now() / 1000);

    const genOpts = {
      maxTokens,
      temperature: params.temperature ?? 0.0,
      topP: params.top_p ?? params.topP ?? 0.95,
      minP: params.min_p ?? params.minP ?? 0.0,
      topK: params.top_k ?? params.topK ?? 40,
      repeatPenalty: params.repeat_penalty ?? params.repeatPenalty ?? 1.0,
      frequencyPenalty: params.frequency_penalty ?? params.frequencyPenalty ?? 0.0,
      presencePenalty: params.presence_penalty ?? params.presencePenalty ?? 0.0,
      penaltyLastN: params.penalty_last_n ?? params.penaltyLastN ?? 64,
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
      return new ChatCompletionStream(tokStream, reqId, created, modelId, grammar !== undefined);
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
    const signal = params.signal ?? callOpts?.signal;
    throwIfAborted(signal);
    const rf = params.response_format ?? params.responseFormat;
    const compiled = compileResponseFormat(rf);
    throwIfAborted(signal);
    const resp = await this._createInternal({ ...params, stream: false }, callOpts, compiled);
    const rawContent = resp.choices[0].message.content ?? "";
    resp.choices[0].message.parsed = compiled.parser ? compiled.parser(rawContent) : null;
    return resp;
  }
}

class CompletionsResource {
  constructor(mgr) {
    this._mgr = mgr;
  }

  async create(params = {}, callOpts = {}) {
    const signal = params.signal ?? callOpts?.signal;
    throwIfAborted(signal);

    const { modelId, engine } = await this._mgr.resolve(params.model);
    throwIfAborted(signal);

    const reqId = `cmpl-ls-${crypto.randomBytes(8).toString("hex")}`;
    const created = Math.floor(Date.now() / 1000);

    const genOpts = {
      maxTokens: params.max_tokens ?? params.maxTokens ?? 256,
      temperature: params.temperature ?? 0.0,
      topP: params.top_p ?? params.topP ?? 0.95,
      minP: params.min_p ?? params.minP ?? 0.0,
      topK: params.top_k ?? params.topK ?? 40,
      repeatPenalty: params.repeat_penalty ?? params.repeatPenalty ?? 1.0,
      frequencyPenalty: params.frequency_penalty ?? params.frequencyPenalty ?? 0.0,
      presencePenalty: params.presence_penalty ?? params.presencePenalty ?? 0.0,
      penaltyLastN: params.penalty_last_n ?? params.penaltyLastN ?? 64,
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
        "OpenAI constructor expects an options object, model spec string, or AutoModel instance",
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
