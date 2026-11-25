const ASYNC_DISPOSE = Symbol.asyncDispose;

function isPlainObject(val) {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

function createAbortError(signal) {
  if (signal && signal.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error(
    typeof signal?.reason === "string" ? signal.reason : "The operation was aborted",
  );
  err.name = "AbortError";
  err.code = "ABORT_ERR";
  return err;
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw createAbortError(signal);
  }
}

function throwBridgeError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  if (msg.startsWith("RANGE_ERR:")) {
    throw new RangeError(msg.slice("RANGE_ERR:".length));
  }
  throw new TypeError(msg.startsWith("TYPE_ERR:") ? msg.slice("TYPE_ERR:".length) : msg);
}

function stringifyOptions(opts) {
  return JSON.stringify(opts, (key, val) => {
    if (key === "signal" || key === "jsonSchema" || key === "json_schema" || key === "grammar") {
      return undefined;
    }
    if (typeof val === "number" && !Number.isFinite(val)) {
      return "__NON_FINITE__";
    }
    return val;
  });
}

export {
  ASYNC_DISPOSE,
  createAbortError,
  isPlainObject,
  stringifyOptions,
  throwBridgeError,
  throwIfAborted,
};
