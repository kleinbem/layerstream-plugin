#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, "..");
const manifestPath = path.join(pkgRoot, "Cargo.toml");

function detectDefaultTriple() {
  const { platform, arch } = process;
  if (platform === "linux" && arch === "x64") return "linux-x64-gnu";
  if (platform === "linux" && arch === "arm64") return "linux-arm64-gnu";
  if (platform === "darwin" && arch === "arm64") return "darwin-arm64";
  if (platform === "darwin" && arch === "x64") return "darwin-x64";
  if (platform === "win32" && arch === "x64") return "win32-x64-msvc";
  throw new Error(`Unsupported platform/arch combination: ${platform}/${arch}`);
}

const rawArgs = process.argv.slice(2);
let triple = detectDefaultTriple();
const cargoArgs = ["build", "--message-format=json-render-diagnostics"];

for (let i = 0; i < rawArgs.length; i += 1) {
  const arg = rawArgs[i];
  if (arg === "--triple") {
    if (i + 1 >= rawArgs.length) {
      throw new Error("--triple requires a value");
    }
    triple = rawArgs[i + 1];
    i += 1;
  } else if (arg === "--message-format" || arg.startsWith("--message-format=")) {
    throw new Error("--message-format is managed by build-native.mjs");
  } else {
    cargoArgs.push(arg);
  }
}

const res = spawnSync("cargo", cargoArgs, {
  cwd: pkgRoot,
  stdio: ["inherit", "pipe", "inherit"],
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
  env: process.env,
});
if (res.error) {
  throw res.error;
}
if (res.status !== 0) {
  process.exit(res.status ?? 1);
}

let builtPath = null;
for (const line of (res.stdout || "").split("\n")) {
  const trimmed = line.trim();
  if (!trimmed) continue;
  let rec;
  try {
    rec = JSON.parse(trimmed);
  } catch {
    continue;
  }
  if (
    rec &&
    rec.reason === "compiler-artifact" &&
    (rec.manifest_path === manifestPath || rec.target?.name === "layerstream_node") &&
    Array.isArray(rec.target?.crate_types) &&
    rec.target.crate_types.includes("cdylib") &&
    Array.isArray(rec.filenames)
  ) {
    const cdylib = rec.filenames.find(
      (f) =>
        typeof f === "string" && (f.endsWith(".so") || f.endsWith(".dylib") || f.endsWith(".dll")),
    );
    if (cdylib) {
      builtPath = cdylib;
    }
  }
}

if (!builtPath || !fs.existsSync(builtPath)) {
  throw new Error("Cargo succeeded without reporting a built layerstream_node cdylib artifact");
}

const destPath = path.join(pkgRoot, `layerstream.${triple}.node`);
fs.copyFileSync(builtPath, destPath);
console.log(`Copied ${builtPath} -> ${destPath}`);
