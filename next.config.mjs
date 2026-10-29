/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [
      // Model files have content-hashed names (see scripts/export_onnx.py) so they can be cached forever.
      { source: "/models/:file(.+\\.onnx)", headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }] },
      { source: "/ort/:file*", headers: [{ key: "Cache-Control", value: "public, max-age=86400" }] },
    ];
  },
};
export default nextConfig;
