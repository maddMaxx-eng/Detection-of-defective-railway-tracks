// Copies the ONNX Runtime Web WASM runtime from node_modules to public/ort so it is served from
// the same origin (no CDN dependency). Runs automatically before `dev` and `build`.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "onnxruntime-web", "dist");
const to = join(root, "public", "ort");
const files = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

if (!existsSync(from)) {
  console.error("onnxruntime-web is not installed. Run `npm install` first.");
  process.exit(1);
}
mkdirSync(to, { recursive: true });
for (const f of files) {
  if (!existsSync(join(from, f))) {
    console.error(`Missing ${f} in onnxruntime-web/dist - the package layout changed.`);
    process.exit(1);
  }
  copyFileSync(join(from, f), join(to, f));
}
console.log(`ONNX Runtime Web: copied ${files.length} files to public/ort`);
