/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  // the public site lives under a sub-path on GitHub Pages (VITE_BASE=/QuantOS/); dev and preview stay at the root
  base: process.env.VITE_BASE ?? "/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    port: 5173,
    // dev: the browser talks to the Node gateway only; it never reaches the Python service
    proxy: { "/api": "http://127.0.0.1:8787" },
  },
  test: { environment: "jsdom", globals: false },
});
