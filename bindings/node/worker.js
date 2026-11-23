"use strict";

const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const net = require("node:net");
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
    this._exitPromise = null;
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
    streamQueue.terminate(errPayload);
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

    state._exitPromise = new Promise((resolve, reject) => {
      if (proc.exitCode !== null || proc.signalCode !== null) {
        resolve();
        return;
      }

      let termTimer = null;
      let killTimer = null;
      let reapDeadline = null;

      const clearTimers = () => {
        if (termTimer) clearTimeout(termTimer);
        if (killTimer) clearTimeout(killTimer);
        if (reapDeadline) clearTimeout(reapDeadline);
      };

      const onExit = () => {
        clearTimers();
        proc.removeListener("error", onError);
        resolve();
      };

      const onError = (err) => {
        clearTimers();
        proc.removeListener("exit", onExit);
        if (proc.exitCode !== null || proc.signalCode !== null) {
          resolve();
        } else {
          reject(new Error(`failed to shut down isolated worker: ${err.message}`));
        }
      };

      proc.once("exit", onExit);
      proc.once("error", onError);

      if (proc.exitCode !== null || proc.signalCode !== null) {
        onExit();
        return;
      }

      termTimer = setTimeout(() => {
        try {
          if (proc.exitCode === null && proc.signalCode === null) {
            proc.kill("SIGTERM");
          }
        } catch {}

        killTimer = setTimeout(() => {
          try {
            if (proc.exitCode === null && proc.signalCode === null) {
              proc.kill("SIGKILL");
            }
          } catch {}

          reapDeadline = setTimeout(() => {
            proc.removeListener("exit", onExit);
            proc.removeListener("error", onError);
            if (proc.exitCode !== null || proc.signalCode !== null) {
              resolve();
            } else {
              reject(new Error("isolated worker did not exit after SIGKILL"));
            }
          }, 5000);
          reapDeadline.unref();
        }, 1000);
        killTimer.unref();
      }, 2000);
      termTimer.unref();
    });

    await state._exitPromise;
  }
}

const workerFinalizer = new FinalizationRegistry((state) => {
  void shutdownWorkerState(state).catch(() => {});
});

class EventQueue {
  constructor() {
    this._items = [];
    this._waiters = [];
    this._terminal = null;
  }

  push(item) {
    if (this._terminal) {
      return;
    }
    const isTerminal =
      item !== null &&
      typeof item === "object" &&
      (item.event === "done" || item.event === "error");
    if (isTerminal) {
      this._terminal = item;
    }
    const waiter = this._waiters.shift();
    if (waiter) {
      waiter(item);
      if (isTerminal) {
        while (this._waiters.length > 0) {
          this._waiters.shift()(item);
        }
      }
    } else {
      this._items.push(item);
    }
  }

  terminate(item) {
    if (this._terminal) {
      return;
    }
    this._terminal = item;
    while (this._waiters.length > 0) {
      this._waiters.shift()(item);
    }
  }

  shift() {
    if (this._items.length > 0) {
      return Promise.resolve(this._items.shift());
    }
    if (this._terminal) {
      return Promise.resolve(this._terminal);
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
      client._finishStream(this._reqId);
      return null;
    }
    if (msg.event === "error") {
      this._done = true;
      this._client = null;
      client._finishStream(this._reqId);
      throw reviveWorkerError(msg.errType, msg.message || "worker stream failed");
    }
    return null;
  }
}

function attachSocketReader(state, rl) {
  let tornDown = false;
  const onTransportDown = (err) => {
    if (tornDown) return;
    tornDown = true;
    state.loaded = false;
    const reason =
      err instanceof Error && err.message
        ? `isolated worker transport error: ${err.message}`
        : "isolated worker exited";
    failAllPending(state, reason);
    try {
      rl.close();
    } catch {}
    void shutdownWorkerState(state).catch(() => {});
  };

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
      if (msg.event === "done") {
        state.activeStreams.delete(reqId);
        if (msg.summary) {
          state.lastSummary = { ...msg.summary };
        }
      } else if (msg.event === "error") {
        state.activeStreams.delete(reqId);
      }
      streamQ.push(msg);
      return;
    }
    const rpcResolve = state.pendingRpc.get(reqId);
    if (rpcResolve) {
      state.pendingRpc.delete(reqId);
      rpcResolve(msg);
    }
  });

  rl.on("error", onTransportDown);
  rl.on("close", () => onTransportDown(null));
  if (state.sock) {
    state.sock.on("error", onTransportDown);
  }
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
    let rl = null;
    let startupTimer = null;

    const cleanupStartup = async (stateOrNull) => {
      if (startupTimer) {
        clearTimeout(startupTimer);
        startupTimer = null;
      }
      try {
        server.close();
      } catch {}
      if (rl) {
        try {
          rl.close();
        } catch {}
      }
      if (stateOrNull) {
        await shutdownWorkerState(stateOrNull).catch(() => {});
      } else {
        const tempState = new WorkerState(proc, conn);
        await shutdownWorkerState(tempState).catch(() => {});
      }
    };

    let state = null;
    try {
      await new Promise((resolve, reject) => {
        let settled = false;
        const fail = (err) => {
          if (settled) return;
          settled = true;
          reject(err);
        };

        startupTimer = setTimeout(() => {
          fail(new Error("timed out waiting for isolated worker handshake"));
        }, 30000);
        startupTimer.unref();

        server.once("error", (err) => {
          fail(err);
        });

        server.once("connection", (socket) => {
          if (settled) {
            socket.destroy();
            return;
          }
          conn = socket;
          conn.setNoDelay(true);
          conn.on("error", (err) => fail(err));
          try {
            server.close();
          } catch {}

          rl = readline.createInterface({ input: conn, crlfDelay: Infinity });
          const onHandshakeErr = (err) => fail(err);
          const onHandshakeClose = () => {
            fail(new Error("isolated worker exited before handshake"));
          };
          const onHandshakeLine = (firstLine) => {
            rl.removeListener("error", onHandshakeErr);
            rl.removeListener("close", onHandshakeClose);
            let hello;
            try {
              hello = JSON.parse(firstLine);
            } catch {
              fail(new Error("invalid handshake from isolated worker"));
              return;
            }
            if (!hello || hello.token !== token) {
              fail(new Error("isolated layerstream worker failed authentication handshake"));
              return;
            }
            if (!settled) {
              settled = true;
              resolve();
            }
          };

          rl.once("error", onHandshakeErr);
          rl.once("close", onHandshakeClose);
          rl.once("line", onHandshakeLine);
        });

        proc = spawn(process.execPath, [__filename, "--worker-port", String(port)], {
          stdio: ["pipe", "ignore", "inherit"],
          env: { ...process.env },
        });

        const onEarlyProcError = (err) => {
          fail(err);
        };
        const onEarlyProcExit = (code, sig) => {
          fail(
            new Error(
              `isolated worker exited before handshake (code=${code ?? "null"}, signal=${sig ?? "null"})`
            )
          );
        };

        proc.once("error", onEarlyProcError);
        proc.once("exit", onEarlyProcExit);
        if (proc.stdin) {
          proc.stdin.on("error", () => {});
          proc.stdin.write(token + "\n");
        }
      });

      clearTimeout(startupTimer);
      startupTimer = null;

      // Remove startup-only listeners on proc/conn before attaching steady-state reader.
      proc.removeAllListeners("error");
      proc.removeAllListeners("exit");
      proc.on("error", () => {});
      conn.removeAllListeners("error");

      state = new WorkerState(proc, conn);
      attachSocketReader(state, rl);
    } catch (err) {
      await cleanupStartup(state);
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes("handshake") ||
        msg.startsWith("failed to start isolated layerstream worker:")
      ) {
        throw new Error(msg);
      }
      throw new Error(`failed to start isolated layerstream worker: ${msg}`);
    }

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
      evQ.terminate({ id: reqId, event: "done", summary: null });
      state.activeStreams.delete(reqId);
      return null;
    }
    this._sendCancelFireAndForget(reqId);
    let captured = null;
    let timer = null;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve(null), 2000);
      timer.unref();
    });
    try {
      while (true) {
        const msg = await Promise.race([evQ.shift(), timeout]);
        if (!msg) break;
        if (msg.event === "done" || msg.event === "error") {
          captured = msg.summary ? { ...msg.summary } : null;
          this._finishStream(reqId);
          break;
        }
      }
    } finally {
      if (timer) clearTimeout(timer);
      evQ.terminate({ id: reqId, event: "done", summary: captured });
      state.activeStreams.delete(reqId);
    }
    return captured;
  }

  _finishStream(reqId) {
    this._state.activeStreams.delete(reqId);
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
  stdinRl.on("error", () => {});
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
  sock.on("error", () => process.exit(0));
  sendJsonLine(sock, { token });

  const outstandingIds = new Set();
  const cancelledIds = new Set();
  const activeCancels = new Map();
  const workQueue = new EventQueue();

  let model = null;

  const engineLoop = (async () => {
    while (true) {
      const cmd = await workQueue.shift();
      if (cmd === null || (cmd && cmd.event === "done")) {
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

      try {
        if (cancelledIds.has(reqId)) {
          if (op === "generate_stream" || op === "chat_stream") {
            sendJsonLine(sock, { id: reqId, event: "done", summary: null });
          } else {
            sendJsonLine(sock, { id: reqId, ok: true, result: {}, summary: null });
          }
          continue;
        }

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
        } else if (op === "generate_stream" || op === "chat_stream") {
          const cancelTok = new native.NativeCancelToken();
          activeCancels.set(reqId, () => cancelTok.cancel());
          if (cancelledIds.has(reqId)) {
            cancelTok.cancel();
          }
          const stream =
            op === "generate_stream"
              ? model.generateStream(params.prompt, JSON.stringify(params.wire || {}))
              : await model.chatStream(
                  JSON.stringify(params.messages),
                  JSON.stringify(params.wire || {}),
                  cancelTok
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
      } finally {
        activeCancels.delete(reqId);
        cancelledIds.delete(reqId);
        outstandingIds.delete(reqId);
      }
    }
  })();

  const sockRl = readline.createInterface({ input: sock, crlfDelay: Infinity });
  sockRl.on("error", () => {});
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
      if (outstandingIds.has(targetId)) {
        cancelledIds.add(targetId);
        const cancelFn = activeCancels.get(targetId);
        if (cancelFn) {
          try {
            cancelFn();
          } catch {}
        }
      }
      continue;
    }
    outstandingIds.add(Number(msg.id));
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
