"use strict";

const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const net = require("node:net");
const path = require("node:path");
const readline = require("node:readline");

function sendJsonLine(sock, obj) {
  if (!sock || sock.destroyed) {
    throw new Error("model has been unloaded");
  }
  sock.write(JSON.stringify(obj) + "\n");
}

function reviveWorkerError(errType, message) {
  if (typeof message === "string" && message.startsWith("ENOENT: ")) {
    const e = new Error(message.slice("ENOENT: ".length));
    e.code = "ENOENT";
    return e;
  }
  if (errType === "TypeError") return new TypeError(message);
  if (errType === "RangeError") return new RangeError(message);
  return new Error(message);
}

class WorkerState {
  constructor(proc, sock) {
    this.proc = proc;
    this.sock = sock;
    this.nextId = 1;
    this.pendingRpc = new Map();
    this.activeStreams = new Map();
    this.loaded = false;
    this.modelPath = null;
    this.lastSummary = null;
    this.shuttingDown = false;
  }
}

function failAllPending(state, message) {
  const errPayload = {
    ok: false,
    event: "error",
    errType: "Error",
    message,
  };
  for (const resolver of state.pendingRpc.values()) {
    resolver(errPayload);
  }
  state.pendingRpc.clear();
  for (const streamQueue of state.activeStreams.values()) {
    streamQueue.push(errPayload);
  }
  state.activeStreams.clear();
}

async function shutdownWorkerState(state) {
  if (state.shuttingDown) {
    if (state._exitPromise) {
      await state._exitPromise;
    }
    return;
  }
  state.shuttingDown = true;
  state.loaded = false;
  state.modelPath = null;

  const sock = state.sock;
  state.sock = null;
  const proc = state.proc;
  state.proc = null;

  if (sock && !sock.destroyed) {
    try {
      sock.write(JSON.stringify({ id: 0, op: "shutdown", params: {} }) + "\n");
    } catch {}
    try {
      sock.destroy();
    } catch {}
  }

  failAllPending(state, "isolated worker exited");

  if (proc) {
    try {
      if (proc.stdin && !proc.stdin.destroyed) {
        proc.stdin.end();
      }
    } catch {}

    state._exitPromise = new Promise((resolve) => {
      if (proc.exitCode !== null || proc.signalCode !== null) {
        resolve();
        return;
      }
      const killTimer = setTimeout(() => {
        try {
          proc.kill("SIGTERM");
        } catch {}
        setTimeout(() => {
          try {
            if (proc.exitCode === null && proc.signalCode === null) {
              proc.kill("SIGKILL");
            }
          } catch {}
          resolve();
        }, 1000).unref();
      }, 2000);
      killTimer.unref();
      proc.once("exit", () => {
        clearTimeout(killTimer);
        resolve();
      });
    });
    await state._exitPromise;
  }
}

const workerFinalizer = new FinalizationRegistry((state) => {
  void shutdownWorkerState(state);
});

class EventQueue {
  constructor() {
    this._items = [];
    this._waiters = [];
  }

  push(item) {
    const waiter = this._waiters.shift();
    if (waiter) {
      waiter(item);
    } else {
      this._items.push(item);
    }
  }

  shift() {
    if (this._items.length > 0) {
      return Promise.resolve(this._items.shift());
    }
    return new Promise((resolve) => {
      this._waiters.push(resolve);
    });
  }
}

class WorkerTokenStream {
  constructor(client, reqId, evQueue) {
    this._client = client;
    this._reqId = reqId;
    this._queue = evQueue;
    this._done = false;
    this._summary = null;
  }

  get summary() {
    return this._summary ? { ...this._summary } : null;
  }

  cancel() {
    const client = this._client;
    if (!this._done && client) {
      client._sendCancelFireAndForget(this._reqId);
    }
  }

  async close() {
    if (this._done) {
      this._client = null;
      return;
    }
    this._done = true;
    const client = this._client;
    this._client = null;
    if (client) {
      const captured = await client._cancelStreamAndDrain(this._reqId, this._queue);
      if (captured && !this._summary) {
        this._summary = captured;
      }
    }
  }

  async nextChunk() {
    if (this._done || !this._client) {
      return null;
    }
    const client = this._client;
    const msg = await this._queue.shift();
    if (msg.event === "chunk") {
      return String(msg.text ?? "");
    }
    if (msg.event === "done") {
      this._done = true;
      this._summary = msg.summary ?? null;
      this._client = null;
      client._finishStream(this._reqId, msg.summary ?? null);
      return null;
    }
    if (msg.event === "error") {
      this._done = true;
      this._client = null;
      client._finishStream(this._reqId, null);
      throw reviveWorkerError(msg.errType, msg.message || "worker stream failed");
    }
    return null;
  }
}

function attachSocketReader(state, rl) {
  rl.on("line", (line) => {
    if (!line) return;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    const reqId = Number(msg.id);
    const streamQ = state.activeStreams.get(reqId);
    if (streamQ) {
      streamQ.push(msg);
      return;
    }
    const rpcResolve = state.pendingRpc.get(reqId);
    if (rpcResolve) {
      state.pendingRpc.delete(reqId);
      rpcResolve(msg);
    }
  });

  const onClose = () => {
    state.loaded = false;
    failAllPending(state, "isolated worker exited");
  };
  rl.on("close", onClose);
}

class IsolatedWorkerClient {
  constructor(state) {
    this._state = state;
    workerFinalizer.register(this, state, this);
  }

  static async create(model, openWire) {
    const token = crypto.randomBytes(32).toString("hex");
    const server = net.createServer();
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });

    const address = server.address();
    const port = typeof address === "object" && address ? address.port : 0;

    let proc = null;
    let conn = null;
    try {
      const connPromise = new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new Error("timed out waiting for isolated worker connection"));
        }, 30000);
        timer.unref();
        server.once("connection", (socket) => {
          clearTimeout(timer);
          resolve(socket);
        });
        server.once("error", (err) => {
          clearTimeout(timer);
          reject(err);
        });
      });

      proc = spawn(process.execPath, [__filename, "--worker-port", String(port)], {
        stdio: ["pipe", "ignore", "inherit"],
        env: { ...process.env },
      });
      proc.stdin.write(token + "\n");

      conn = await connPromise;
      conn.setNoDelay(true);
    } catch (err) {
      if (conn) conn.destroy();
      if (proc) {
        try {
          proc.kill("SIGKILL");
        } catch {}
      }
      throw new Error(`failed to start isolated layerstream worker: ${err.message}`);
    } finally {
      server.close();
    }

    const state = new WorkerState(proc, conn);
    const rl = readline.createInterface({ input: conn, crlfDelay: Infinity });

    const firstLine = await new Promise((resolve, reject) => {
      const onLine = (l) => {
        rl.removeListener("close", onClose);
        resolve(l);
      };
      const onClose = () => {
        rl.removeListener("line", onLine);
        reject(new Error("isolated worker exited before handshake"));
      };
      rl.once("line", onLine);
      rl.once("close", onClose);
    });

    let hello;
    try {
      hello = JSON.parse(firstLine);
    } catch {
      await shutdownWorkerState(state);
      throw new Error("invalid handshake from isolated worker");
    }
    if (!hello || hello.token !== token) {
      await shutdownWorkerState(state);
      throw new Error("isolated layerstream worker failed authentication handshake");
    }

    attachSocketReader(state, rl);
    const client = new IsolatedWorkerClient(state);

    try {
      const { result } = await client.rpcWithSummary("open", { model, wire: openWire });
      state.loaded = true;
      state.modelPath = result.modelPath ?? null;
      return client;
    } catch (err) {
      await client.unload();
      throw err;
    }
  }

  _allocId() {
    const id = this._state.nextId;
    this._state.nextId += 1;
    return id;
  }

  _sendCancelFireAndForget(targetId) {
    const sock = this._state.sock;
    if (!sock || sock.destroyed) return;
    try {
      sendJsonLine(sock, {
        id: this._allocId(),
        op: "cancel",
        params: { targetId },
      });
    } catch {}
  }

  async rpc(op, params, signal = undefined) {
    const { result } = await this.rpcWithSummary(op, params, signal);
    return result;
  }

  async rpcWithSummary(op, params, signal = undefined) {
    const state = this._state;
    if (!state.sock || state.sock.destroyed || (op !== "open" && !state.loaded)) {
      throw new Error("model has been unloaded");
    }
    const reqId = this._allocId();
    const respPromise = new Promise((resolve) => {
      state.pendingRpc.set(reqId, resolve);
    });

    const onAbort = signal
      ? () => {
          this._sendCancelFireAndForget(reqId);
        }
      : null;
    if (signal && onAbort) {
      if (signal.aborted) {
        state.pendingRpc.delete(reqId);
        const err = new Error("The operation was aborted");
        err.name = "AbortError";
        throw err;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }

    try {
      sendJsonLine(state.sock, { id: reqId, op, params });
      const resp = await respPromise;
      if (!resp.ok) {
        throw reviveWorkerError(resp.errType, resp.message || "worker request failed");
      }
      const summary = resp.summary ? { ...resp.summary } : null;
      if (summary) {
        state.lastSummary = { ...summary };
      }
      return { result: resp.result, summary };
    } finally {
      state.pendingRpc.delete(reqId);
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
    }
  }

  startStream(op, params) {
    const state = this._state;
    if (!state.sock || state.sock.destroyed || !state.loaded) {
      throw new Error("model has been unloaded");
    }
    const reqId = this._allocId();
    const evQ = new EventQueue();
    state.activeStreams.set(reqId, evQ);
    try {
      sendJsonLine(state.sock, { id: reqId, op, params });
    } catch (err) {
      state.activeStreams.delete(reqId);
      throw err;
    }
    return new WorkerTokenStream(this, reqId, evQ);
  }

  async _cancelStreamAndDrain(reqId, evQ) {
    const state = this._state;
    if (!state.activeStreams.has(reqId) || !state.sock || state.sock.destroyed) {
      state.activeStreams.delete(reqId);
      return null;
    }
    this._sendCancelFireAndForget(reqId);
    let captured = null;
    const timeout = new Promise((resolve) => setTimeout(() => resolve(null), 2000).unref());
    try {
      while (state.activeStreams.has(reqId)) {
        const msg = await Promise.race([evQ.shift(), timeout]);
        if (!msg) break;
        if (msg.event === "done" || msg.event === "error") {
          captured = msg.summary ? { ...msg.summary } : null;
          this._finishStream(reqId, captured);
          break;
        }
      }
    } finally {
      state.activeStreams.delete(reqId);
    }
    return captured;
  }

  _finishStream(reqId, summary) {
    this._state.activeStreams.delete(reqId);
    if (summary) {
      this._state.lastSummary = { ...summary };
    }
  }

  get isLoaded() {
    const proc = this._state.proc;
    return Boolean(
      this._state.loaded && proc && proc.exitCode === null && proc.signalCode === null
    );
  }

  get workerPid() {
    const proc = this._state.proc;
    if (proc && proc.exitCode === null && proc.signalCode === null) {
      return proc.pid ?? null;
    }
    return null;
  }

  get modelPath() {
    return this._state.loaded ? this._state.modelPath : null;
  }

  get lastSummary() {
    return this._state.lastSummary ? { ...this._state.lastSummary } : null;
  }

  async stats() {
    if (!this.isLoaded) {
      return { loaded: false, isolatedWorker: true, workerPid: null };
    }
    const st = await this.rpc("stats", {});
    return {
      ...st,
      isolatedWorker: true,
      workerPid: this.workerPid,
    };
  }

  async unload() {
    workerFinalizer.unregister(this);
    await shutdownWorkerState(this._state);
  }
}

async function runChildWorkerMain(port) {
  const native = require("./native.js");

  const stdinRl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  const token = await new Promise((resolve) => {
    stdinRl.once("line", (l) => resolve(l.trim()));
    stdinRl.once("close", () => resolve(""));
  });
  if (!token) {
    process.exit(1);
  }

  process.stdin.resume();
  process.stdin.on("end", () => process.exit(0));
  process.stdin.on("error", () => process.exit(0));

  const sock = net.createConnection({ host: "127.0.0.1", port });
  await new Promise((resolve, reject) => {
    sock.once("connect", resolve);
    sock.once("error", reject);
  });
  sock.setNoDelay(true);
  sendJsonLine(sock, { token });

  const cancelledIds = new Set();
  const activeCancels = new Map();
  const workQueue = new EventQueue();

  let model = null;

  const engineLoop = (async () => {
    while (true) {
      const cmd = await workQueue.shift();
      if (cmd === null) {
        if (model) {
          try {
            await model.unload();
          } catch {}
          model = null;
        }
        break;
      }

      const reqId = Number(cmd.id);
      const op = String(cmd.op);
      const params = cmd.params || {};

      if (cancelledIds.has(reqId)) {
        cancelledIds.delete(reqId);
        if (op === "generate_stream" || op === "chat_stream") {
          sendJsonLine(sock, { id: reqId, event: "done", summary: null });
        } else {
          sendJsonLine(sock, { id: reqId, ok: true, result: {}, summary: null });
        }
        continue;
      }

      try {
        if (op === "open") {
          model = await native.NativeAutoModel.open(
            params.model,
            JSON.stringify(params.wire || {})
          );
          sendJsonLine(sock, {
            id: reqId,
            ok: true,
            result: { modelPath: model.modelPath },
          });
        } else if (!model) {
          throw new Error("model has been unloaded");
        } else if (op === "stats") {
          const st = await model.stats();
          sendJsonLine(sock, { id: reqId, ok: true, result: st });
        } else if (op === "tokenize") {
          const tokens = await model.tokenize(
            params.text,
            Boolean(params.addSpecial),
            Boolean(params.parseSpecial)
          );
          sendJsonLine(sock, { id: reqId, ok: true, result: { tokens } });
        } else if (op === "detokenize") {
          const text = await model.detokenize(params.tokens);
          sendJsonLine(sock, { id: reqId, ok: true, result: { text } });
        } else if (op === "apply_chat_template") {
          const prompt = await model.applyChatTemplate(
            JSON.stringify(params.messages),
            Boolean(params.addGenerationPrompt),
            params.tools ? JSON.stringify(params.tools) : undefined
          );
          sendJsonLine(sock, { id: reqId, ok: true, result: { prompt } });
        } else if (op === "kv_clear") {
          await model.kvClear();
          sendJsonLine(sock, { id: reqId, ok: true, result: {} });
        } else if (op === "kv_trim") {
          const trimmed = await model.kvTrim(Number(params.posMin));
          sendJsonLine(sock, { id: reqId, ok: true, result: { trimmed } });
        } else if (op === "generate" || op === "chat") {
          const cancelTok = new native.NativeCancelToken();
          activeCancels.set(reqId, () => cancelTok.cancel());
          if (cancelledIds.has(reqId)) {
            cancelTok.cancel();
          }
          try {
            if (op === "generate") {
              const out = await model.generateWithSummary(
                params.prompt,
                JSON.stringify(params.wire || {}),
                cancelTok
              );
              sendJsonLine(sock, {
                id: reqId,
                ok: true,
                result: { text: out.text },
                summary: out.summary,
              });
            } else {
              const out = await model.chatWithSummary(
                JSON.stringify(params.messages),
                JSON.stringify(params.wire || {}),
                cancelTok
              );
              sendJsonLine(sock, {
                id: reqId,
                ok: true,
                result: { text: out.text, parsed: out.parsed ?? null },
                summary: out.summary,
              });
            }
          } finally {
            activeCancels.delete(reqId);
            cancelledIds.delete(reqId);
          }
        } else if (op === "generate_stream" || op === "chat_stream") {
          const stream =
            op === "generate_stream"
              ? model.generateStream(params.prompt, JSON.stringify(params.wire || {}))
              : await model.chatStream(
                  JSON.stringify(params.messages),
                  JSON.stringify(params.wire || {})
                );
          activeCancels.set(reqId, () => stream.cancel());
          if (cancelledIds.has(reqId)) {
            stream.cancel();
          }
          try {
            while (true) {
              const piece = await stream.nextChunk();
              if (piece === null || piece === undefined) {
                break;
              }
              if (cancelledIds.has(reqId)) {
                stream.cancel();
                break;
              }
              sendJsonLine(sock, { id: reqId, event: "chunk", text: piece });
            }
          } finally {
            activeCancels.delete(reqId);
            cancelledIds.delete(reqId);
            await stream.close();
          }
          sendJsonLine(sock, {
            id: reqId,
            event: "done",
            summary: stream.summary ?? model.lastSummary ?? null,
          });
        } else {
          throw new RangeError(`unknown worker op '${op}'`);
        }
      } catch (err) {
        const errType = err?.constructor?.name || "Error";
        const message = err?.message || String(err);
        if (op === "generate_stream" || op === "chat_stream") {
          try {
            sendJsonLine(sock, { id: reqId, event: "error", errType, message });
          } catch {}
        } else {
          try {
            sendJsonLine(sock, { id: reqId, ok: false, errType, message });
          } catch {}
        }
      }
    }
  })();

  const sockRl = readline.createInterface({ input: sock, crlfDelay: Infinity });
  for await (const line of sockRl) {
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }
    if (msg.op === "shutdown") {
      break;
    }
    if (msg.op === "cancel") {
      const targetId = Number(msg.params?.targetId);
      cancelledIds.add(targetId);
      const cancelFn = activeCancels.get(targetId);
      if (cancelFn) {
        try {
          cancelFn();
        } catch {}
      }
      continue;
    }
    workQueue.push(msg);
  }

  for (const cancelFn of activeCancels.values()) {
    try {
      cancelFn();
    } catch {}
  }
  workQueue.push(null);
  await engineLoop;
  try {
    sock.destroy();
  } catch {}
  process.exit(0);
}

if (require.main === module) {
  const idx = process.argv.indexOf("--worker-port");
  if (idx === -1 || idx + 1 >= process.argv.length) {
    process.exit(2);
  }
  const port = Number(process.argv[idx + 1]);
  runChildWorkerMain(port).catch((err) => {
    process.stderr.write(`layerstream worker fatal: ${err?.stack || err}\n`);
    process.exit(1);
  });
}

module.exports = {
  IsolatedWorkerClient,
  WorkerTokenStream,
};
