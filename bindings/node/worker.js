// Child-process entrypoint and spawner for `isolatedWorker: true`.
// All IPC framing, loopback TCP auth, cancellation, and OS process reaping live in Rust
// (`layerstream-core::worker_ipc` + `bindings/node/src/worker.rs`).

import path from "node:path";
import native from "./native.js";

async function spawnWorker(model, openWire) {
  return await native.NativeWorkerClient.spawn(
    process.execPath,
    [import.meta.filename],
    model,
    JSON.stringify(openWire || {}),
  );
}

function runWorkerEntrypoint(argv = process.argv, runServer = native.runWorkerServer) {
  const idx = argv.indexOf("--worker-port");
  if (idx === -1 || idx + 1 >= argv.length) {
    return 2;
  }
  const port = Number(argv[idx + 1]);
  try {
    runServer(port);
    return 0;
  } catch (err) {
    process.stderr.write(`layerstream worker fatal: ${err?.stack || err}\n`);
    return 1;
  }
}

// coverage: top-level child-process main guard; tested via `runWorkerEntrypoint()` and live isolatedWorker spawn.
/* node:coverage ignore next 4 */
if (process.argv[1] && path.resolve(process.argv[1]) === import.meta.filename) {
  const code = runWorkerEntrypoint();
  if (code !== 0) process.exit(code);
}

export { runWorkerEntrypoint, spawnWorker };
