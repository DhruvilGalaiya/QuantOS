import type { Implied, SeriesPoint } from "@quantos/types";

export interface Cone {
  upper: SeriesPoint[];
  lower: SeriesPoint[];
}

const DAY = 86_400_000;
const iso = (t: number) => new Date(t).toISOString().slice(0, 10);

/**
 * The range option prices imply between now and expiry: spot ± move × √(elapsed / total), on weekdays only so
 * its spacing matches the trading-day price history. Starts at the later of the snapshot and the last close.
 * This is what options charge, not a forecast.
 */
export function impliedCone(imp: Implied, lastDate: string | undefined): Cone | null {
  if (!imp.expiry || !imp.move_points) return null;
  const snap = imp.snapshot_ts ? imp.snapshot_ts.slice(0, 10) : lastDate;
  if (!snap) return null;
  const startDay = lastDate && lastDate > snap ? lastDate : snap;
  const start = Date.parse(`${startDay}T00:00:00Z`);
  const end = Date.parse(`${imp.expiry}T00:00:00Z`);
  if (!(end > start)) return null;
  const upper: SeriesPoint[] = [];
  const lower: SeriesPoint[] = [];
  for (let t = start; t <= end; t += DAY) {
    const wd = new Date(t).getUTCDay();
    if ((wd === 0 || wd === 6) && t !== start && t !== end) continue;
    const w = imp.move_points * Math.sqrt((t - start) / (end - start));
    upper.push({ time: iso(t), value: imp.spot + w });
    lower.push({ time: iso(t), value: imp.spot - w });
  }
  return { upper, lower };
}
