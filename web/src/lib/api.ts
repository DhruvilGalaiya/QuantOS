import type { ApiRoutes, RouteName } from "@quantos/types";
import { useQuery } from "@tanstack/react-query";

/**
 * Where data comes from:
 *   api    -> the Node gateway (/api/v1/<route>), used in development and owner mode
 *   static -> JSON exported at build time (/data/v1/<route>.json), used by the public site
 * Both return the same generated types.
 */
export const DATA_SOURCE: "api" | "static" = import.meta.env.VITE_DATA_SOURCE === "static" ? "static" : "api";

export function routeUrl(name: RouteName): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return DATA_SOURCE === "static" ? `${base}/data/v1/${name}.json` : `/api/v1/${name}`;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function fetchRoute<K extends RouteName>(name: K, signal?: AbortSignal): Promise<ApiRoutes[K]> {
  const res = await fetch(routeUrl(name), { signal, headers: { accept: "application/json" } });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      msg = (await res.json()).message ?? msg;
    } catch {
      /* keep the status message */
    }
    throw new ApiError(msg, res.status);
  }
  return res.json() as Promise<ApiRoutes[K]>;
}

export function useRoute<K extends RouteName>(name: K) {
  return useQuery({
    queryKey: ["route", name],
    queryFn: ({ signal }) => fetchRoute(name, signal),
    staleTime: DATA_SOURCE === "static" ? Infinity : 60_000,
    refetchInterval: DATA_SOURCE === "static" ? false : 5 * 60_000,
  });
}

/** Any API path under /api/v1 (or its static JSON file), e.g. "regimes/NIFTY_50", "us/AAPL". */
export function pathUrl(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  // symbols and tickers are upper case; static hosts match file names exactly, so /regimes/nifty_50 still works
  path = path.replace(/\/([^/]+)$/, (_, id: string) => `/${id.toUpperCase()}`);
  return DATA_SOURCE === "static" ? `${base}/data/v1/${path}.json` : `/api/v1/${path}`;
}

export async function fetchPath<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(pathUrl(path), { signal, headers: { accept: "application/json" } });
  if (!res.ok) {
    let msg = res.status === 404 ? "Not found" : `Request failed (${res.status})`;
    try {
      msg = (await res.json()).message ?? msg;
    } catch {
      /* static hosts answer 404 with HTML */
    }
    throw new ApiError(msg, res.status);
  }
  return res.json() as Promise<T>;
}

export function useApi<T>(path: string | null) {
  return useQuery({
    queryKey: ["path", path],
    queryFn: ({ signal }) => fetchPath<T>(path!, signal),
    enabled: path != null,
    staleTime: DATA_SOURCE === "static" ? Infinity : 60_000,
    placeholderData: (prev) => prev,
    retry: (n, err) => !(err instanceof ApiError && err.status === 404) && n < 1,
  });
}
