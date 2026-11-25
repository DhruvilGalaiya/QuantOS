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
