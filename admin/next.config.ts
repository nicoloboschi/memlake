import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The repo root has no lockfile of its own, and there may be lockfiles above
  // it; pin the workspace root to admin/ so tracing stays inside this app.
  turbopack: { root: path.resolve(process.cwd()) },

  /*
   * Keep the server-only heavyweights out of the bundler entirely.
   *
   *  - @grpc/grpc-js and @grpc/proto-loader do runtime `require`s and read the
   *    .proto off disk; bundling them breaks both.
   *  - @huggingface/transformers pulls in onnxruntime-node, which ships native
   *    .node binaries.
   *
   * They are resolved from node_modules at runtime instead. None of these are
   * ever imported from a "use client" component, so nothing reaches the browser
   * bundle either way.
   */
  serverExternalPackages: [
    "@grpc/grpc-js",
    "@grpc/proto-loader",
    "@huggingface/transformers",
    "onnxruntime-node",
    "sharp",
  ],
};

export default nextConfig;
