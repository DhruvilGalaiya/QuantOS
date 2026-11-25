import type { FastifyInstance } from "fastify";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { buildApp } from "../src/app.ts";
import { hashPassword, issueSession, readCookie, verifyPassword, verifySession } from "../src/auth.ts";
import { TickHub } from "../src/ticks.ts";

const SECRET = "s".repeat(40);
const TICKS = JSON.stringify({ ts: "t1", market_open: true, quotes: [{ symbol: "NIFTY_50", name: "NIFTY 50", ltp: 25000 }] });
let HASH = "";
beforeAll(async () => {
  HASH = await hashPassword("correct horse battery");
});

let app: FastifyInstance;
afterEach(async () => app?.close());

function fakeQuant() {
  const calls: string[] = [];
  const fetchImpl = (async (input: URL | RequestInfo) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith("/internal/v1/ticks")) return new Response(TICKS, { status: 200 });
    return new Response("{}", { status: 200, headers: { etag: 'W/"x"' } });
  }) as typeof fetch;
  return { calls, fetchImpl };
}

async function make(config: Record<string, unknown> = { OWNER_PASSWORD_HASH: HASH, SESSION_SECRET: SECRET }) {
  const q = fakeQuant();
  app = await buildApp({ fetch: q.fetchImpl, config: { LOG_LEVEL: "silent", ...config } });
  return q;
}

const login = (password: string) =>
  app.inject({ method: "POST", url: "/api/v1/auth/login", payload: { password }, headers: { "content-type": "application/json" } });

describe("auth primitives", () => {
  it("hashes and verifies passwords", async () => {
    expect(await verifyPassword("correct horse battery", HASH)).toBe(true);
    expect(await verifyPassword("wrong", HASH)).toBe(false);
    expect(await verifyPassword("x", "plain-text")).toBe(false);
  });

  it("signs sessions that expire and cannot be forged", () => {
    const t = issueSession(SECRET, 1000, 0);
    expect(verifySession(SECRET, t, 500)).toBe(true);
    expect(verifySession(SECRET, t, 1500)).toBe(false); // expired
    expect(verifySession("o".repeat(40), t, 500)).toBe(false); // other secret
    const [, sig] = t.split(".");
    expect(verifySession(SECRET, `99999999999999.${sig}`, 500)).toBe(false); // extended expiry, old signature
    expect(verifySession(SECRET, undefined)).toBe(false);
  });

  it("reads one cookie from a header", () => {
    expect(readCookie("a=1; quantos_owner=abc.def; b=2", "quantos_owner")).toBe("abc.def");
    expect(readCookie(undefined, "x")).toBeUndefined();
  });
});

describe("owner sign-in", () => {
  it("is off when not configured", async () => {
    await make({});
    expect((await app.inject("/api/v1/auth/me")).json()).toEqual({ enabled: false, owner: false });
    expect((await login("anything")).statusCode).toBe(503);
  });

  it("rejects a wrong password and sets no cookie", async () => {
    await make();
    const r = await login("nope");
    expect(r.statusCode).toBe(401);
    expect(r.headers["set-cookie"]).toBeUndefined();
  });

  it("signs in with an HttpOnly, SameSite=Strict cookie and recognises it", async () => {
    await make();
    const r = await login("correct horse battery");
    expect(r.statusCode).toBe(200);
    const cookie = String(r.headers["set-cookie"]);
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("SameSite=Strict");
    expect(cookie).toContain("Path=/api");
    expect(cookie).not.toContain("Secure"); // local HTTP by default
    const me = await app.inject({ url: "/api/v1/auth/me", headers: { cookie: cookie.split(";")[0] } });
    expect(me.json()).toEqual({ enabled: true, owner: true });
  });

  it("marks the cookie Secure when configured for HTTPS", async () => {
    await make({ OWNER_PASSWORD_HASH: HASH, SESSION_SECRET: SECRET, COOKIE_SECURE: true });
    expect(String((await login("correct horse battery")).headers["set-cookie"])).toContain("Secure");
  });

  it("refuses non-JSON sign-in bodies (no cross-site form posts)", async () => {
    await make();
    const r = await app.inject({ method: "POST", url: "/api/v1/auth/login", payload: "password=x", headers: { "content-type": "application/x-www-form-urlencoded" } });
    expect(r.statusCode).toBe(415);
  });

  it("rate limits sign-in attempts", async () => {
    await make();
    const codes = [];
    for (let i = 0; i < 7; i++) codes.push((await login("nope")).statusCode);
    expect(codes.slice(0, 5).every((c) => c === 401)).toBe(true);
    expect(codes.at(-1)).toBe(429);
  });

  it("signs out by clearing the cookie", async () => {
    await make();
    const r = await app.inject({ method: "POST", url: "/api/v1/auth/logout" });
    expect(String(r.headers["set-cookie"])).toContain("Max-Age=0");
  });
});

describe("live prices", () => {
  it("needs the owner session", async () => {
    const q = await make();
    const r = await app.inject("/api/v1/ticks/stream");
    expect(r.statusCode).toBe(401);
    expect(q.calls.some((c) => c.endsWith("/ticks"))).toBe(false);
  });

  it("streams prices to a signed-in owner as server-sent events", async () => {
    await make();
    const cookie = String((await login("correct horse battery")).headers["set-cookie"]).split(";")[0];
    const address = await app.listen({ host: "127.0.0.1", port: 0 });
    const ctrl = new AbortController();
    const res = await fetch(`${address}/api/v1/ticks/stream`, { headers: { cookie }, signal: ctrl.signal });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const reader = res.body!.getReader();
    let text = "";
    while (!text.includes("data: ")) text += new TextDecoder().decode((await reader.read()).value);
    while (!text.includes("\n\n", text.indexOf("data: "))) text += new TextDecoder().decode((await reader.read()).value);
    const line = text.split("\n").find((l) => l.startsWith("data: "))!;
    expect(JSON.parse(line.slice(6)).quotes[0].ltp).toBe(25000);
    ctrl.abort();
  });
});

describe("tick hub", () => {
  it("polls only while someone listens and sends only changes", async () => {
    vi.useFakeTimers();
    let n = 0;
    const bodies = [TICKS, TICKS, JSON.stringify({ ts: "t2", market_open: true, quotes: [] })];
    const load = vi.fn(async () => bodies[Math.min(n++, bodies.length - 1)]);
    const hub = new TickHub(load, { openMs: 1000, closedMs: 60000, retryMs: 5000 });
    const got: string[] = [];
    const off = hub.subscribe((p) => got.push(p));
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(1000);
    expect(load).toHaveBeenCalledTimes(3);
    expect(got).toHaveLength(2); // the repeated payload was not re-sent
    off();
    await vi.advanceTimersByTimeAsync(5000);
    expect(load).toHaveBeenCalledTimes(3); // nobody listening: no more calls
    hub.close();
    vi.useRealTimers();
  });
});
