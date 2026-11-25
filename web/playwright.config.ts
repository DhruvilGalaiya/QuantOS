import { defineConfig } from "@playwright/test";

/**
 * Smoke tests against the built public site (static JSON, no servers): every page renders its headline, shows no
 * error box and logs no console errors, and a few key numbers match the exported data.
 *
 *   npm run build -w @quantos/web && npm run e2e -w @quantos/web
 *
 * BASE must match the build's VITE_BASE ("/" locally, "/QuantOS/" for GitHub Pages). Uses the installed Chrome.
 */
const BASE = process.env.BASE ?? "/";
const PORT = 4174;

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { baseURL: `http://127.0.0.1:${PORT}${BASE}`, channel: "chrome", viewport: { width: 1366, height: 860 } },
  projects: [
    { name: "desktop" },
    { name: "phone", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }, grep: /@phone/ },
  ],
  webServer: {
    command: `npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort --base ${BASE}`,
    url: `http://127.0.0.1:${PORT}${BASE}`,
    reuseExistingServer: !process.env.CI,
  },
});
