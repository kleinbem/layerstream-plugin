import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  AutoModel,
  check,
  formatToolsSystemPrompt,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  OpenAI,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  recommend,
} from "../index.js";
import { OpenAI as SubpathOpenAI } from "../openai.js";

const pkgRoot = path.resolve(import.meta.dirname, "..");

test("ESM entrypoints export core SDK and OpenAI surfaces", () => {
  for (const fn of [
    AutoModel,
    OpenAI,
    probeGpuMemory,
    inspectGguf,
    check,
    recommend,
    pull,
    jsonSchemaToGbnf,
    jsonObjectGbnf,
    parseAssistantOutput,
    formatToolsSystemPrompt,
    SubpathOpenAI,
  ]) {
    assert.equal(typeof fn, "function");
  }
});

test("native.js rejects unsupported darwin/x64 platform cleanly", () => {
  const nativeUrl = pathToFileURL(path.join(pkgRoot, "native.js")).href;
  const script = `
    Object.defineProperty(process, "platform", { value: "darwin" });
    Object.defineProperty(process, "arch", { value: "x64" });
    await import(${JSON.stringify(nativeUrl)});
  `;
  assert.throws(
    () => execFileSync(process.execPath, ["--input-type=module", "-e", script], { stdio: "pipe" }),
    /Unsupported platform\/arch for layerstream native addon: darwin\/x64/,
  );
});
