import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import type { RouteName } from "@quantos/types";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { z } from "zod";
import { COOKIE, issueSession, readCookie, sessionCookie, verifyPassword, verifySession } from "./auth.ts";
import { type Config, loadConfig } from "./config.ts";
import { TickHub } from "./ticks.ts";
import { type Fetch, Upstream, UpstreamError } from "./upstream.ts";

/** Public routes: /api/v1/<name> -> quant service /internal/v1/<name>. */
export const ROUTES: readonly RouteName[] = ["meta", "today", "system"] as const;

/** Routes with one path parameter, validated here before anything reaches the quant service. */
const SYMBOL = z.string().regex(/^[A-Z0-9_]{2,24}$/);
const TICKER = z.string().regex(/^[A-Za-z][A-Za-z0-9.-]{0,9}$/).transform((t) => t.toUpperCase());
export const PARAM_ROUTES = [
  { path: "regimes", param: SYMBOL },
  { path: "volatility", param: SYMBOL },
  { path: "states", param: SYMBOL },
  { path: "events", param: SYMBOL },
  { path: "surface", param: SYMBOL },
  { path: "us", param: TICKER },
] as const;

export interface BuildOptions {
  config?: Partial<Config>;
  fetch?: Fetch;
}

export async function buildApp(opts: BuildOptions = {}): Promise<FastifyInstance> {
  const config = loadConfig(process.env, opts.config);
  const app = Fastify({
    logger: config.LOG_LEVEL === "silent" ? false : { level: config.LOG_LEVEL },
    bodyLimit: 4 * 1024, // the only body is the sign-in form; refuse anything larger outright
    trustProxy: true,
  });
  const upstream = new Upstream(config.QUANT_URL, {
    fetch: opts.fetch,
    timeoutMs: config.UPSTREAM_TIMEOUT_MS,
    freshMs: config.CACHE_FRESH_MS,
  });

  await app.register(helmet, {
    contentSecurityPolicy: { directives: { defaultSrc: ["'none'"], frameAncestors: ["'none'"] } },
    crossOriginResourcePolicy: { policy: "same-site" },
  });
  await app.register(cors, { origin: config.corsOrigins, methods: ["GET", "HEAD"], maxAge: 600 });
  await app.register(rateLimit, { max: config.RATE_LIMIT_PER_MIN, timeWindow: "1 minute" });

  app.setErrorHandler((err: Error & { statusCode?: number }, req, reply) => {
    const status = err instanceof UpstreamError ? err.status : (err.statusCode ?? 500);
    if (status >= 500) req.log.error({ err }, "request failed");
    const error = status === 429 ? "rate_limited" : status === 404 ? "not_found" : status >= 500 ? "upstream_unavailable" : "bad_request";
    reply.status(status).send({ error, message: err.message });
  });
  app.setNotFoundHandler((_req, reply) => reply.status(404).send({ error: "not_found", message: "No such route" }));

  app.get("/healthz", async () => ({ ok: true }));

  const serve = async (req: FastifyRequest, reply: FastifyReply, upstreamPath: string) => {
    const entry = await upstream.get(upstreamPath);
    reply.header("cache-control", "public, max-age=30, stale-while-revalidate=300");
    if (entry.etag) {
      reply.header("etag", entry.etag);
      if (req.headers["if-none-match"] === entry.etag) return reply.status(304).send();
    }
    return reply.type("application/json; charset=utf-8").send(entry.body);
  };

  for (const name of ROUTES) {
    app.get(`/api/v1/${name}`, (req, reply) => serve(req, reply, `/internal/v1/${name}`));
  }
  app.get("/api/v1/us", (req, reply) => serve(req, reply, "/internal/v1/us"));
  app.get("/api/v1/risk", (req, reply) => serve(req, reply, "/internal/v1/risk"));
  for (const r of PARAM_ROUTES) {
    app.get<{ Params: { id: string } }>(`/api/v1/${r.path}/:id`, (req, reply) => {
      const parsed = r.param.safeParse(req.params.id);
      if (!parsed.success) return reply.status(400).send({ error: "bad_request", message: `Invalid ${r.path} id` });
      return serve(req, reply, `/internal/v1/${r.path}/${encodeURIComponent(parsed.data)}`);
    });
  }
  // ------------------------------------------------------------ owner sign-in + live prices
  const authOn = Boolean(config.OWNER_PASSWORD_HASH && config.SESSION_SECRET);
  const isOwner = (req: FastifyRequest) =>
    authOn && verifySession(config.SESSION_SECRET!, readCookie(req.headers.cookie, COOKIE));
  const Login = z.object({ password: z.string().min(1).max(256) });

  app.get("/api/v1/auth/me", async (req, reply) => {
    reply.header("cache-control", "no-store");
    return { enabled: authOn, owner: isOwner(req) };
  });

  // JSON only (a cross-site form cannot send it without a CORS preflight, which is refused) + SameSite=Strict cookie
  app.post("/api/v1/auth/login", { config: { rateLimit: { max: 5, timeWindow: "1 minute" } } }, async (req, reply) => {
    reply.header("cache-control", "no-store");
    if (!authOn) return reply.status(503).send({ error: "auth_disabled", message: "Owner sign-in is not set up on this server" });
    const body = Login.safeParse(req.body);
    if (!body.success) return reply.status(400).send({ error: "bad_request", message: "Password required" });
    if (!(await verifyPassword(body.data.password, config.OWNER_PASSWORD_HASH!))) {
      req.log.warn("owner sign-in failed");
      return reply.status(401).send({ error: "unauthorized", message: "Wrong password" });
    }
    const ttl = config.SESSION_DAYS * 86_400_000;
    reply.header("set-cookie", sessionCookie(issueSession(config.SESSION_SECRET!, ttl), Math.floor(ttl / 1000), config.COOKIE_SECURE));
    return { enabled: true, owner: true };
  });

  app.post("/api/v1/auth/logout", async (_req, reply) => {
    reply.header("cache-control", "no-store").header("set-cookie", sessionCookie("", 0, config.COOKIE_SECURE));
    return { enabled: authOn, owner: false };
  });

  const loadTicks = async () => {
    const f = opts.fetch ?? fetch;
    const res = await f(new URL("/internal/v1/ticks", config.QUANT_URL), {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(config.UPSTREAM_TIMEOUT_MS),
    });
    if (!res.ok) throw new UpstreamError(`ticks returned ${res.status}`, 502);
    return res.text();
  };
  const hub = new TickHub(loadTicks, { openMs: config.TICK_OPEN_MS, closedMs: config.TICK_CLOSED_MS, retryMs: 10_000 }, app.log);
  const streams = new Set<() => void>();
  app.addHook("onClose", async () => {
    for (const close of streams) close();
    hub.close();
  });

  // Server-Sent Events: one-way, reconnects by itself, and carries the session cookie like any request
  app.get("/api/v1/ticks/stream", (req, reply) => {
    if (!isOwner(req)) return reply.status(401).send({ error: "unauthorized", message: "Sign in as the owner for live prices" });
    reply.hijack();
    const res = reply.raw;
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
      "x-accel-buffering": "no",
      "x-content-type-options": "nosniff",
    });
    res.write("retry: 5000\n\n");
    const unsubscribe = hub.subscribe((payload) => res.write(`data: ${payload}\n\n`));
    const ping = setInterval(() => res.write(": ping\n\n"), 20_000);
    const end = () => {
      clearInterval(ping);
      unsubscribe();
      streams.delete(close);
    };
    const close = () => {
      end();
      res.end();
    };
    streams.add(close);
    req.raw.on("close", end);
  });

  return app;
}
