import { afterEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp } from "../src/app.ts";

const TODAY = JSON.stringify({ as_of: "2026-11-25", indices: [] });

/** A fake quant service: answers with an ETag and honours If-None-Match; counts calls. */
function fakeUpstream(opts: { status?: number; fail?: boolean } = {}) {
  const calls: { url: string; inm: string | null }[] = [];
  const fetchImpl = (async (input: URL | RequestInfo, init?: RequestInit) => {
    const url = String(input);
    const inm = new Headers(init?.headers).get("if-none-match");
    calls.push({ url, inm });
    if (opts.fail) throw new Error("ECONNREFUSED");
    if (opts.status) return new Response("{}", { status: opts.status });
    if (inm === 'W/"v1"') return new Response(null, { status: 304, headers: { etag: 'W/"v1"' } });
    return new Response(TODAY, { status: 200, headers: { etag: 'W/"v1"', "content-type": "application/json" } });
  }) as typeof fetch;
  return { calls, fetchImpl };
}

let app: FastifyInstance;
afterEach(async () => app?.close());

async function make(up = fakeUpstream(), config = {}) {
  app = await buildApp({ fetch: up.fetchImpl, config: { LOG_LEVEL: "silent", CACHE_FRESH_MS: 0, ...config } });
  return up;
}

describe("gateway", () => {
  it("health check", async () => {
    await make();
    const r = await app.inject("/healthz");
    expect(r.json()).toEqual({ ok: true });
  });

  it("proxies a page route to the internal API", async () => {
    const up = await make();
    const r = await app.inject("/api/v1/today");
    expect(r.statusCode).toBe(200);
    expect(r.json()).toEqual(JSON.parse(TODAY));
    expect(up.calls[0].url).toBe("http://127.0.0.1:8001/internal/v1/today");
    expect(r.headers.etag).toBe('W/"v1"');
  });

  it("answers 304 when the browser already has the current version", async () => {
    await make();
    const r = await app.inject({ url: "/api/v1/today", headers: { "if-none-match": 'W/"v1"' } });
    expect(r.statusCode).toBe(304);
    expect(r.body).toBe("");
  });

  it("revalidates its cache with the upstream ETag", async () => {
    const up = await make();
    await app.inject("/api/v1/today");
    const r = await app.inject("/api/v1/today");
    expect(r.json()).toEqual(JSON.parse(TODAY));
    expect(up.calls.map((c) => c.inm)).toEqual([null, 'W/"v1"']);
  });

  it("serves from cache without calling upstream while fresh", async () => {
    const up = await make(fakeUpstream(), { CACHE_FRESH_MS: 60_000 });
    await Promise.all([app.inject("/api/v1/today"), app.inject("/api/v1/today"), app.inject("/api/v1/today")]);
    expect(up.calls).toHaveLength(1);
  });

  it("returns 503 when the quant service is down and nothing is cached", async () => {
    await make(fakeUpstream({ fail: true }));
    const r = await app.inject("/api/v1/today");
    expect(r.statusCode).toBe(503);
    expect(r.json().error).toBe("upstream_unavailable");
  });

  it("maps upstream server errors to 502", async () => {
    await make(fakeUpstream({ status: 500 }));
    expect((await app.inject("/api/v1/meta")).statusCode).toBe(502);
  });

  it("unknown routes are 404 and never reach the quant service", async () => {
    const up = await make();
    const r = await app.inject("/api/v1/../internal/v1/today");
    expect([404]).toContain(r.statusCode);
    expect((await app.inject("/api/v1/nope")).statusCode).toBe(404);
    expect(up.calls).toHaveLength(0);
  });

  it("is read-only", async () => {
    await make();
    expect((await app.inject({ method: "POST", url: "/api/v1/today", payload: {} })).statusCode).toBe(404);
  });

  it("sets security headers", async () => {
    await make();
    const h = (await app.inject("/api/v1/today")).headers;
    expect(h["content-security-policy"]).toContain("default-src 'none'");
    expect(h["x-content-type-options"]).toBe("nosniff");
    expect(h["strict-transport-security"]).toBeDefined();
    expect(h["x-frame-options"]).toBe("SAMEORIGIN");
  });

  it("allows only configured CORS origins", async () => {
    await make();
    const ok = await app.inject({ url: "/api/v1/today", headers: { origin: "http://localhost:5173" } });
    expect(ok.headers["access-control-allow-origin"]).toBe("http://localhost:5173");
    const bad = await app.inject({ url: "/api/v1/today", headers: { origin: "https://evil.example" } });
    expect(bad.headers["access-control-allow-origin"]).toBeUndefined();
  });

  it("rate limits each client", async () => {
    await make(fakeUpstream(), { RATE_LIMIT_PER_MIN: 2 });
    await app.inject("/healthz");
    await app.inject("/healthz");
    const r = await app.inject("/healthz");
    expect(r.statusCode).toBe(429);
    expect(r.json().error).toBe("rate_limited");
  });
});

describe("parameter routes", () => {
  it("passes a valid symbol through and upper-cases tickers", async () => {
    const up = await make();
    expect((await app.inject("/api/v1/regimes/NIFTY_50")).statusCode).toBe(200);
    expect((await app.inject("/api/v1/us/aapl")).statusCode).toBe(200);
    expect(up.calls.map((c) => c.url)).toEqual([
      "http://127.0.0.1:8001/internal/v1/regimes/NIFTY_50",
      "http://127.0.0.1:8001/internal/v1/us/AAPL",
    ]);
  });

  it("rejects malformed ids before the quant service sees them", async () => {
    const up = await make();
    for (const bad of ["/api/v1/regimes/nifty;drop", "/api/v1/volatility/%2e%2e", "/api/v1/us/1ABC", "/api/v1/us/AAAAAAAAAAAA"]) {
      expect([400, 404]).toContain((await app.inject(bad)).statusCode);
    }
    expect(up.calls).toHaveLength(0);
  });

  it("passes an upstream 404 through as not_found", async () => {
    await make(fakeUpstream({ status: 404 }));
    const r = await app.inject("/api/v1/regimes/NOPE");
    expect(r.statusCode).toBe(404);
    expect(r.json().error).toBe("not_found");
  });
});
