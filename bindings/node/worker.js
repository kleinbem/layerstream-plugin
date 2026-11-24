// Thin Node.js adapter around the unified Rust isolated-worker IPC client & server
// (`layerstream-core::worker_ipc`).

const native = require("./native.js");

function createAbortError(signal) {
  if (signal?.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error(
    typeof signal?.reason === "string" ? signal.reason : "The operation was aborted",
  );
  err.name = "AbortError";
  return err;
}

class IsolatedWorkerClient {
  constructor(nativeClient) {
    this._native = nativeClient;
  }

  static async create(model, openWire) {
    const client = await native.NativeWorkerClient.spawn(
      process.execPath,
      [__filename],
      model,
      JSON.stringify(openWire || {}),
    );
    return new IsolatedWorkerClient(client);
  }

  async rpc(op, params = {}, signal = undefined) {
    const { result } = await this.rpcWithSummary(op, params, signal);
    return result;
  }

  async rpcWithSummary(op, params = {}, signal = undefined) {
    if (signal?.aborted) {
      throw createAbortError(signal);
    }
    switch (op) {
      case "generate": {
        const out = await this._native.generateWithSummary(
          params.prompt,
          JSON.stringify(params.wire || {}),
        );
        return { result: { text: out.text }, summary: out.summary ?? null };
      }
      case "chat": {
        const out = await this._native.chatWithSummary(
          JSON.stringify(params.messages),
          JSON.stringify(params.wire || {}),
        );
        return {
          result: { text: out.text, parsed: out.parsed ?? null },
          summary: out.summary ?? null,
        };
      }
      case "tokenize": {
        const tokens = await this._native.tokenize(
          params.text,
          Boolean(params.addSpecial),
          Boolean(params.parseSpecial),
        );
        return { result: { tokens }, summary: null };
      }
      case "detokenize": {
        const text = await this._native.detokenize(params.tokens);
        return { result: { text }, summary: null };
      }
      case "apply_chat_template": {
        const prompt = await this._native.applyChatTemplate(
          JSON.stringify(params.messages),
          Boolean(params.addGenerationPrompt),
          params.tools ? JSON.stringify(params.tools) : undefined,
        );
        return { result: { prompt }, summary: null };
      }
      case "kv_clear": {
        await this._native.kvClear();
        return { result: {}, summary: null };
      }
      case "kv_trim": {
        const trimmed = await this._native.kvTrim(Number(params.posMin));
        return { result: { trimmed }, summary: null };
      }
      case "save_state": {
        const tokens = await this._native.saveState(String(params.path));
        return { result: { tokens }, summary: null };
      }
      case "load_state": {
        const tokens = await this._native.loadState(String(params.path));
        return { result: { tokens }, summary: null };
      }
      default:
        throw new RangeError(`unknown worker op '${op}'`);
    }
  }

  startStream(op, params = {}) {
    if (op === "generate_stream") {
      return this._native.generateStream(params.prompt, JSON.stringify(params.wire || {}));
    }
    if (op === "chat_stream") {
      return this._native.chatStream(
        JSON.stringify(params.messages),
        JSON.stringify(params.wire || {}),
      );
    }
    throw new RangeError(`unknown worker stream op '${op}'`);
  }

  get isLoaded() {
    return this._native.isLoaded;
  }

  get workerPid() {
    return this._native.workerPid ?? null;
  }

  get modelPath() {
    return this._native.modelPath ?? null;
  }

  get lastSummary() {
    return this._native.lastSummary ?? null;
  }

  async stats() {
    return await this._native.stats();
  }

  async unload() {
    await this._native.unload();
  }
}

if (require.main === module) {
  const idx = process.argv.indexOf("--worker-port");
  if (idx === -1 || idx + 1 >= process.argv.length) {
    process.exit(2);
  }
  const port = Number(process.argv[idx + 1]);
  try {
    native.runWorkerServer(port);
  } catch (err) {
    process.stderr.write(`layerstream worker fatal: ${err?.stack || err}\n`);
    process.exit(1);
  }
}

module.exports = {
  IsolatedWorkerClient,
};
