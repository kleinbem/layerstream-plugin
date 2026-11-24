const fs = require("node:fs");
const path = require("node:path");

function resolveNativeTriple() {
  const { platform, arch } = process;
  if (platform === "linux" && arch === "x64") return "linux-x64-gnu";
  if (platform === "linux" && arch === "arm64") return "linux-arm64-gnu";
  if (platform === "darwin" && arch === "arm64") return "darwin-arm64";
  if (platform === "win32" && arch === "x64") return "win32-x64-msvc";
  throw new Error(`Unsupported platform/arch for layerstream native addon: ${platform}/${arch}`);
}

function loadNativeAddon() {
  const triple = resolveNativeTriple();
  const candidates = [];
  if (process.env.LAYERSTREAM_NATIVE_PATH) {
    candidates.push(path.resolve(process.env.LAYERSTREAM_NATIVE_PATH));
  }
  candidates.push(path.join(__dirname, `layerstream.${triple}.node`));

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return require(candidate);
    }
  }

  throw new Error(
    `Could not find native layerstream Node-API binary ('layerstream.${triple}.node'). ` +
      `Run 'npm run build' in bindings/node or set LAYERSTREAM_NATIVE_PATH.`,
  );
}

module.exports = loadNativeAddon();
