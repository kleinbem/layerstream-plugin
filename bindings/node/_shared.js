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

export { ASYNC_DISPOSE, createAbortError, isPlainObject, throwIfAborted };
