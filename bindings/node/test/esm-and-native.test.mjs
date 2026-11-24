import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import * as rootEsm from "../index.mjs";
import * as openaiEsm from "../openai.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, "..");

test("ESM entrypoints export core SDK and OpenAI surfaces", () => {
  for (const name of [
    "AutoModel",
    "OpenAI",
    "probeGpuMemory",
    "inspectGguf",
    "check",
    "recommend",
    "pull",
    "jsonSchemaToGbnf",
    "jsonObjectGbnf",
    "parseAssistantOutput",
    "formatToolsSystemPrompt",
  ]) {
    assert.equal(typeof rootEsm[name], "function", `missing ESM export: ${name}`);
  }
  assert.equal(typeof openaiEsm.OpenAI, "function");
});

test("native.js rejects unsupported darwin/x64 platform cleanly", () => {
  const script = `
    Object.defineProperty(process, "platform", { value: "darwin" });
    Object.defineProperty(process, "arch", { value: "x64" });
    require(${JSON.stringify(path.join(pkgRoot, "native.js"))});
  `;
  assert.throws(
    () => execFileSync(process.execPath, ["-e", script], { stdio: "pipe" }),
    /Unsupported platform\/arch for layerstream native addon: darwin\/x64/
  );
});
