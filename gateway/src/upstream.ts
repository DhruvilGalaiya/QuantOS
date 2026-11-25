import { LRUCache } from "lru-cache";

export type Fetch = typeof fetch;

interface Entry {
  etag: string | null;
  body: string;
  checked: number; // when the upstream last confirmed this body
}

export class UpstreamError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * Client for the Python quant service with a revalidating cache: within `freshMs` the cached body is served
 * without a call; after that the upstream is asked with If-None-Match and a 304 keeps the cached body.
 * Concurrent requests for the same path share one upstream call.
 */
export class Upstream {
  private cache = new LRUCache<string, Entry>({ max: 200 });
  private inflight = new Map<string, Promise<Entry>>();

  constructor(
    private baseUrl: string,
    private opts: { fetch?: Fetch; timeoutMs: number; freshMs: number },
  ) {}

  async get(path: string): Promise<Entry> {
    const hit = this.cache.get(path);
    if (hit && Date.now() - hit.checked < this.opts.freshMs) return hit;
    const pending = this.inflight.get(path);
    if (pending) return pending;
    const p = this.load(path, hit).finally(() => this.inflight.delete(path));
    this.inflight.set(path, p);
    return p;
  }

  private async load(path: string, hit: Entry | undefined): Promise<Entry> {
    const f = this.opts.fetch ?? fetch;
    const headers: Record<string, string> = { accept: "application/json" };
    if (hit?.etag) headers["if-none-match"] = hit.etag;
    let res: Response;
    try {
      res = await f(new URL(path, this.baseUrl), { headers, signal: AbortSignal.timeout(this.opts.timeoutMs) });
    } catch (err) {
      if (hit) return hit; // serve the last good copy when the quant service is briefly down
      throw new UpstreamError(`quant service unreachable: ${(err as Error).message}`, 503);
    }
    if (res.status === 304 && hit) {
      const fresh = { ...hit, checked: Date.now() };
      this.cache.set(path, fresh);
      return fresh;
    }
    if (!res.ok) {
      if (hit && res.status >= 500) return hit;
      throw new UpstreamError(`quant service returned ${res.status}`, res.status >= 500 ? 502 : res.status);
    }
    const entry = { etag: res.headers.get("etag"), body: await res.text(), checked: Date.now() };
    this.cache.set(path, entry);
    return entry;
  }
}
