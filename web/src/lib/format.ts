// Number formatting for the terminal: Indian digit grouping for levels and rupees, signed percents.

const nf = (d: number) => new Intl.NumberFormat("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });

const ok = (x: number | null | undefined): x is number => typeof x === "number" && Number.isFinite(x);

export const DASH = "—";

/** 25349.65 -> "25,349.65" */
export function level(x: number | null | undefined, d = 2): string {
  return ok(x) ? nf(d).format(x) : DASH;
}

/** 0.0123 -> "1.2%" ; signed -> "+1.2%" */
export function pct(x: number | null | undefined, d = 1, signed = false): string {
  if (!ok(x)) return DASH;
  const s = nf(d).format(Math.abs(x * 100));
  return `${signed ? (x > 0 ? "+" : x < 0 ? "−" : "") : x < 0 ? "−" : ""}${s}%`;
}

/** volatility points: 0.0243 -> "+2.4 pts" */
export function pts(x: number | null | undefined, d = 1): string {
  if (!ok(x)) return DASH;
  return `${x > 0 ? "+" : x < 0 ? "−" : ""}${nf(d).format(Math.abs(x * 100))} pts`;
}

export function rupees(x: number | null | undefined, d = 0): string {
  return ok(x) ? `₹${nf(d).format(x)}` : DASH;
}

export function share(x: number | null | undefined): string {
  return ok(x) ? `${Math.round(x * 100)}%` : DASH;
}

export function tone(x: number | null | undefined): "up" | "down" | "flat" {
  return !ok(x) || x === 0 ? "flat" : x > 0 ? "up" : "down";
}

const IST = "Asia/Kolkata";

/** "2026-11-23" -> "Mon 23 Nov" */
export function day(iso: string | null | undefined, withYear = false): string {
  if (!iso) return DASH;
  const d = new Date(iso.length === 10 ? `${iso}T12:00:00+05:30` : iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short", day: "2-digit", month: "short", ...(withYear ? { year: "numeric" } : {}), timeZone: IST,
  }).format(d).replace(",", "");
}

/** ISO timestamp -> "25 Nov, 13:46 IST" */
export function stamp(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const s = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: IST,
  }).format(d);
  return `${s.replace(",", ",")} IST`;
}
