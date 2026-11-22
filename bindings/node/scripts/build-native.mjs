#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, "..");

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
let profile = "debug";
const cargoArgs = ["build"];

for (let i = 0; i < rawArgs.length; i += 1) {
  const arg = rawArgs[i];
  if (arg === "--triple") {
    if (i + 1 >= rawArgs.length) {
      throw new Error("--triple requires a value");
    }
    triple = rawArgs[i + 1];
    i += 1;
  } else if (arg === "--release") {
    profile = "release";
    cargoArgs.push("--release");
  } else {
    cargoArgs.push(arg);
  }
}

const res = spawnSync("cargo", cargoArgs, {
  cwd: pkgRoot,
  stdio: "inherit",
  env: process.env,
});
if (res.status !== 0) {
  process.exit(res.status ?? 1);
}

const targetDir = process.env.CARGO_TARGET_DIR
  ? path.resolve(pkgRoot, process.env.CARGO_TARGET_DIR)
  : path.join(pkgRoot, "target");

const libFile =
  process.platform === "win32"
    ? "layerstream_node.dll"
    : process.platform === "darwin"
      ? "liblayerstream_node.dylib"
      : "liblayerstream_node.so";

const builtPath = path.join(targetDir, profile, libFile);
if (!fs.existsSync(builtPath)) {
  throw new Error(`Expected built cdylib at '${builtPath}', but file was not found`);
}

const destPath = path.join(pkgRoot, `layerstream.${triple}.node`);
fs.copyFileSync(builtPath, destPath);
console.log(`Copied ${builtPath} -> ${destPath}`);
