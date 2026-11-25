import { z } from "zod";

// All settings come from the environment (never committed). Defaults suit local development.
const Env = z.object({
  HOST: z.string().default("127.0.0.1"),
  PORT: z.coerce.number().int().positive().default(8787),
  QUANT_URL: z.url().default("http://127.0.0.1:8001"),
  CORS_ORIGINS: z.string().default("http://localhost:5173,http://127.0.0.1:5173"),
  RATE_LIMIT_PER_MIN: z.coerce.number().int().positive().default(120),
  UPSTREAM_TIMEOUT_MS: z.coerce.number().int().positive().default(15000),
  CACHE_FRESH_MS: z.coerce.number().int().nonnegative().default(5000),
  LOG_LEVEL: z.enum(["fatal", "error", "warn", "info", "debug", "trace", "silent"]).default("info"),
  // owner sign-in (live prices). Both unset = sign-in off. Make them with `npm run hash-password -w gateway`.
  OWNER_PASSWORD_HASH: z.string().startsWith("scrypt$").optional(),
  SESSION_SECRET: z.string().min(32).optional(),
  SESSION_DAYS: z.coerce.number().positive().default(7),
  // Secure cookies need HTTPS; true whenever the gateway is served over HTTPS
  COOKIE_SECURE: z.stringbool().default(false),
  TICK_OPEN_MS: z.coerce.number().int().positive().default(3000),
  TICK_CLOSED_MS: z.coerce.number().int().positive().default(60000),
});

export type Config = z.infer<typeof Env> & { corsOrigins: string[] };

export function loadConfig(env: NodeJS.ProcessEnv = process.env, overrides: Partial<Config> = {}): Config {
  const parsed = Env.parse(env);
  const cfg = { ...parsed, ...overrides };
  return {
    ...cfg,
    corsOrigins: overrides.corsOrigins ?? cfg.CORS_ORIGINS.split(",").map((s) => s.trim()).filter(Boolean),
  };
}
