import type { Ticks } from "@quantos/types";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { ApiError, DATA_SOURCE } from "@/lib/api";

/**
 * Owner mode: sign in against the gateway (password checked there, session in an HttpOnly cookie the page never
 * sees), then live prices stream over Server-Sent Events. The public static site has no gateway, so all of this
 * stays off there.
 */
export interface Me {
  enabled: boolean;
  owner: boolean;
}

const OFF: Me = { enabled: false, owner: false };

export function useOwner() {
  return useQuery({
    queryKey: ["auth", "me"],
    queryFn: async ({ signal }): Promise<Me> => {
      const res = await fetch("/api/v1/auth/me", { signal, credentials: "same-origin" });
      return res.ok ? ((await res.json()) as Me) : OFF;
    },
    enabled: DATA_SOURCE === "api",
    initialData: DATA_SOURCE === "api" ? undefined : OFF,
    staleTime: 5 * 60_000,
  });
}

async function post(path: string, body?: object): Promise<Me> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError((data as { message?: string }).message ?? `Sign-in failed (${res.status})`, res.status);
  return data as Me;
}

export function useSignIn() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (password: string) => post("/api/v1/auth/login", { password }),
    onSuccess: (me) => qc.setQueryData(["auth", "me"], me),
  });
}

export function useSignOut() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => post("/api/v1/auth/logout"),
    onSuccess: (me) => qc.setQueryData(["auth", "me"], me),
  });
}

/**
 * One stream per page however many components read it: the first reader opens the EventSource, the last one to
 * leave closes it.
 */
type LiveState = { ticks: Ticks | null; connected: boolean };
const IDLE: LiveState = { ticks: null, connected: false };
let live: LiveState = IDLE;
let source: EventSource | null = null;
const listeners = new Set<() => void>();

function emit(next: LiveState) {
  live = next;
  for (const l of listeners) l();
}

function subscribeLive(listener: () => void) {
  listeners.add(listener);
  if (!source) {
    source = new EventSource("/api/v1/ticks/stream", { withCredentials: true });
    source.onopen = () => emit({ ...live, connected: true });
    source.onerror = () => emit({ ...live, connected: false }); // the browser reconnects on its own
    source.onmessage = (e) => {
      try {
        emit({ ticks: JSON.parse(e.data) as Ticks, connected: true });
      } catch {
        /* ignore a malformed frame */
      }
    };
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && source) {
      source.close();
      source = null;
      live = IDLE;
    }
  };
}

const noop = () => () => {};

/** Live index prices while the owner is signed in; null otherwise (callers fall back to end-of-day). */
export function useLiveTicks(): LiveState {
  const { data: me } = useOwner();
  const on = Boolean(me?.owner);
  const state = useSyncExternalStore(on ? subscribeLive : noop, () => (on ? live : IDLE), () => IDLE);
  return state;
}
