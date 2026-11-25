// Small SVG charts for the terminal: one generic line/band chart with hover and keyboard stepping, bars and
// sparklines. Points are spaced evenly by index (trading days or categories), so weekends never leave gaps.
import { type ReactNode, useMemo, useState } from "react";
import { useWidth } from "@/components/cine";
import { cn } from "@/lib/utils";

export interface Series {
  id: string;
  data: (number | null | undefined)[];
  color: string;
  width?: number;
  dash?: string;
  /** fill from the line down to `areaTo` (default: bottom of the plot) */
  area?: string;
  areaTo?: number;
}
export interface Band {
  lo: (number | null | undefined)[];
  hi: (number | null | undefined)[];
  color: string;
}

export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => raw / s <= 1) ?? raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const ok = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

function linePath(data: (number | null | undefined)[], x: (i: number) => number, y: (v: number) => number) {
  let d = "";
  let pen = false;
  data.forEach((v, i) => {
    if (!ok(v)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

export function LineChart({
  labels,
  series,
  bands = [],
  gap,
  height = 300,
  yFormat = (v) => String(v),
  xFormat = (s) => s,
  xTicks = 5,
  yDomain,
  zero,
  readout,
  ariaLabel,
  className,
  markers,
}: {
  labels: string[];
  series: Series[];
  bands?: Band[];
  /** shade between two series: `pos` where a ≥ b, `neg` where a < b */
  gap?: { a: number; b: number; pos: string; neg: string };
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (label: string) => string;
  xTicks?: number;
  yDomain?: [number, number];
  zero?: boolean;
  readout?: (i: number) => ReactNode;
  ariaLabel: string;
  className?: string;
  /** dots drawn on a series at given indices */
  markers?: { series: number; color: string }[];
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const n = labels.length;
  const padL = 8, padR = 52, padT = 10, padB = 26;
  const W = Math.max(w, 10), H = height;
  const iw = W - padL - padR, ih = H - padT - padB;

  const [lo, hi] = useMemo(() => {
    if (yDomain) return yDomain;
    const vals = [...series.flatMap((s) => s.data), ...bands.flatMap((b) => [...b.lo, ...b.hi])].filter(ok);
    if (!vals.length) return [0, 1];
    let a = Math.min(...vals), b = Math.max(...vals);
    if (zero) {
      a = Math.min(a, 0);
      b = Math.max(b, 0);
    }
    const pad = (b - a) * 0.08 || Math.abs(b) * 0.1 || 1;
    return [a - pad, b + pad];
  }, [series, bands, yDomain, zero]);

  const x = (i: number) => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v: number) => padT + ih - ((v - lo) / (hi - lo)) * ih;
  const ticks = niceTicks(lo, hi, 4);
  const xIdx = n > 1 ? Array.from({ length: Math.min(xTicks, n) }, (_, k) => Math.round((k / Math.max(Math.min(xTicks, n) - 1, 1)) * (n - 1))) : [0];

  const gapPolys = useMemo(() => {
    if (!gap || !w) return [];
    const A = series[gap.a].data, B = series[gap.b].data;
    const polys: { d: string; pos: boolean }[] = [];
    let run: number[] = [];
    let sign: boolean | null = null;
    const flush = () => {
      if (run.length > 1 && sign != null) {
        const top = run.map((i) => `${x(i).toFixed(1)},${y(A[i] as number).toFixed(1)}`);
        const bot = [...run].reverse().map((i) => `${x(i).toFixed(1)},${y(B[i] as number).toFixed(1)}`);
        polys.push({ d: `M${top.join("L")}L${bot.join("L")}Z`, pos: sign });
      }
      run = [];
    };
    A.forEach((a, i) => {
      const b = B[i];
      if (!ok(a) || !ok(b)) {
        flush();
        sign = null;
        return;
      }
      const s = a >= b;
      if (sign !== null && s !== sign) {
        run.push(i);
        flush();
      }
      sign = s;
      run.push(i);
    });
    flush();
    return polys;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gap, series, w, lo, hi, n]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left - padL) / iw) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    setHover((h) => Math.max(0, Math.min(n - 1, (h ?? n - 1) + (e.key === "ArrowRight" ? step : -step))));
  };
  const at = hover ?? n - 1;

  return (
    <div ref={ref} className={cn("relative", className)}>
      {readout && <div className="mb-2 min-h-11 text-[13px] text-muted-foreground" aria-live="polite">{readout(at)}</div>}
      {w > 0 && (
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`${ariaLabel}. Use the left and right arrow keys to step through the data.`}
          tabIndex={0}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
          className="block touch-pan-y select-none outline-none focus-visible:ring-2 focus-visible:ring-saffron/70"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={padL} x2={padL + iw} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray={t === 0 && zero ? undefined : "2 4"} />
              <text x={padL + iw + 8} y={y(t) + 4} fontSize="11" fill="var(--faint)">
                {yFormat(t)}
              </text>
            </g>
          ))}
          {bands.map((b, k) => {
            const up = b.hi.map((v, i) => (ok(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null)).filter(Boolean);
            const dn = b.lo.map((v, i) => (ok(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null)).filter(Boolean).reverse();
            return up.length > 1 ? <path key={k} d={`M${up.join("L")}L${dn.join("L")}Z`} fill={b.color} /> : null;
          })}
          {gapPolys.map((p, k) => (
            <path key={k} d={p.d} fill={p.pos ? gap!.pos : gap!.neg} />
          ))}
          {series.map((s) =>
            s.area ? (
              <path
                key={`${s.id}-a`}
                d={`${linePath(s.data, x, y)}L${x(s.data.length - 1).toFixed(1)},${y(s.areaTo ?? lo).toFixed(1)}L${x(s.data.findIndex(ok)).toFixed(1)},${y(s.areaTo ?? lo).toFixed(1)}Z`}
                fill={s.area}
              />
            ) : null,
          )}
          {series.map((s) => (
            <path key={s.id} d={linePath(s.data, x, y)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.6} strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {markers?.map((m) =>
            series[m.series].data.map((v, i) => (ok(v) ? <circle key={`${m.series}-${i}`} cx={x(i)} cy={y(v)} r="4" fill={m.color} stroke="var(--background)" strokeWidth="2" /> : null)),
          )}
          {xIdx.map((i, k) => (
            <text key={i} x={x(i)} y={H - 6} fontSize="11" fill="var(--faint)" textAnchor={k === 0 ? "start" : k === xIdx.length - 1 ? "end" : "middle"}>
              {xFormat(labels[i])}
            </text>
          ))}
          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(at)} x2={x(at)} y1={padT} y2={padT + ih} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
              {series.map((s) => (ok(s.data[at]) ? <circle key={s.id} cx={x(at)} cy={y(s.data[at] as number)} r="3.5" fill={s.color} stroke="var(--background)" strokeWidth="1.5" /> : null))}
            </g>
          )}
        </svg>
      )}
    </div>
  );
}

export function BarChart({
  labels,
  values,
  height = 220,
  color,
  format = (v) => String(v),
  ariaLabel,
  baseline,
  baselineLabel,
}: {
  labels: string[];
  values: number[];
  height?: number;
  color?: (v: number, i: number) => string;
  format?: (v: number) => string;
  ariaLabel: string;
  baseline?: number;
  baselineLabel?: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const padT = 22, padB = values.some((v) => v < 0) ? 44 : 24;
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values, baseline ?? 0);
  const ih = height - padT - padB;
  const y = (v: number) => padT + ih - ((v - lo) / (hi - lo || 1)) * ih;
  const bw = w / Math.max(values.length, 1);
  return (
    <div ref={ref}>
      {w > 0 && (
        <svg width={w} height={height} role="img" aria-label={ariaLabel} className="block">
          <line x1="0" x2={w} y1={y(0)} y2={y(0)} stroke="var(--line-strong)" />
          {baseline != null && (
            <g>
              <line x1="0" x2={w} y1={y(baseline)} y2={y(baseline)} stroke="var(--faint)" strokeDasharray="3 4" />
              {baselineLabel && (
                <text x={w - 2} y={y(baseline) - 5} fontSize="11" fill="var(--faint)" textAnchor="end">
                  {baselineLabel}
                </text>
              )}
            </g>
          )}
          {values.map((v, i) => {
            const top = Math.min(y(v), y(0)), h = Math.abs(y(v) - y(0));
            return (
              <g key={i}>
                <rect x={i * bw + bw * 0.18} y={top} width={bw * 0.64} height={Math.max(h, 1)} rx="4" fill={color ? color(v, i) : "var(--accent-saffron)"} />
                <text x={i * bw + bw / 2} y={v >= 0 ? top - 6 : top + h + 14} fontSize="12" fontWeight="600" fill="var(--foreground)" textAnchor="middle">
                  {format(v)}
                </text>
                <text x={i * bw + bw / 2} y={height - 6} fontSize="11.5" fill="var(--faint)" textAnchor="middle">
                  {labels[i]}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

export function Sparkline({ data, width = 96, height = 28, className }: { data: number[]; width?: number; height?: number; className?: string }) {
  if (data.length < 2) return null;
  const lo = Math.min(...data), hi = Math.max(...data);
  const pts = data.map((v, i) => `${((i / (data.length - 1)) * width).toFixed(1)},${(height - 2 - ((v - lo) / (hi - lo || 1)) * (height - 4)).toFixed(1)}`);
  const up = data[data.length - 1] >= data[0];
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={className} aria-hidden>
      <path d={`M${pts.join("L")}`} fill="none" stroke={up ? "var(--up)" : "var(--down)"} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}
