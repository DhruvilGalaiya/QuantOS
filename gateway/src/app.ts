import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import type { RouteName } from "@quantos/types";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { z } from "zod";
import { type Config, loadConfig } from "./config.ts";
import { type Fetch, Upstream, UpstreamError } from "./upstream.ts";

/** Public routes: /api/v1/<name> -> quant service /internal/v1/<name>. */
export const ROUTES: readonly RouteName[] = ["meta", "today"] as const;

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
    bodyLimit: 16 * 1024, // the API is read-only today; refuse large bodies outright
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
  return app;
}
