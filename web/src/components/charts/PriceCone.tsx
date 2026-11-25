import type { Implied, SeriesPoint } from "@quantos/types";
import { ColorType, createChart, CrosshairMode, LineSeries, LineStyle, type Time, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { impliedCone } from "@/lib/cone";
import { day, level } from "@/lib/format";
import { cssVar, useTheme } from "@/lib/theme";

const PERIODS = [
  { label: "3M", sessions: 63 },
  { label: "6M", sessions: 126 },
  { label: "1Y", sessions: 260 },
] as const;

type Hover = { time: string; value: number; band?: [number, number] } | null;

const timeStr = (t: Time) =>
  typeof t === "string" ? t : typeof t === "number" ? new Date((t as UTCTimestamp) * 1000).toISOString().slice(0, 10) : `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}`;

/** One year of closes with the option-implied ±1σ range drawn forward to expiry. */
export function PriceCone({ history: full, implied, height = 240, label }: { history: SeriesPoint[]; implied: Implied | null | undefined; height?: number; label: string }) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useTheme((s) => s.theme);
  const [hover, setHover] = useState<Hover>(null);
  const [period, setPeriod] = useState<(typeof PERIODS)[number]["label"]>("3M");
  const history = useMemo(() => full.slice(-PERIODS.find((p) => p.label === period)!.sessions), [full, period]);
  const cone = useMemo(() => (implied ? impliedCone(implied, history.at(-1)?.time) : null), [implied, history]);

  useEffect(() => {
    if (!el.current || !history.length) return;
    // read colours from the chart's own subtree, so a forced-dark section (the landing page) is respected
    const fg = cssVar("--foreground", el.current);
    const muted = cssVar("--muted-foreground", el.current);
    const line = cssVar("--line", el.current);
    const saffron = cssVar("--accent-saffron", el.current);
    const chart = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: muted, fontFamily: "IBM Plex Sans", fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: line } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.08, bottom: 0.06 } },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true, lockVisibleTimeRangeOnResize: true },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: muted, labelVisible: false, style: LineStyle.Dotted }, horzLine: { color: muted, style: LineStyle.Dotted } },
      handleScroll: false,
      handleScale: false,
      localization: { priceFormatter: (p: number) => level(p, 0) },
    });
    const price = chart.addSeries(LineSeries, { color: fg, lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerRadius: 3 });
    price.setData(history);
    const bandOpts = { color: saffron, lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false } as const;
    const up = cone ? chart.addSeries(LineSeries, bandOpts) : null;
    const lo = cone ? chart.addSeries(LineSeries, bandOpts) : null;
    if (cone && up && lo) {
      up.setData(cone.upper);
      lo.setData(cone.lower);
    }
    chart.timeScale().fitContent();

    chart.subscribeCrosshairMove((p) => {
      if (!p.time) return setHover(null);
      const pv = p.seriesData.get(price) as { value: number } | undefined;
      const uv = up ? (p.seriesData.get(up) as { value: number } | undefined) : undefined;
      const lv = lo ? (p.seriesData.get(lo) as { value: number } | undefined) : undefined;
      const t = timeStr(p.time);
      if (pv) setHover({ time: t, value: pv.value });
      else if (uv && lv) setHover({ time: t, value: (uv.value + lv.value) / 2, band: [lv.value, uv.value] });
      else setHover(null);
    });
    return () => chart.remove();
  }, [history, cone, theme]);

  const last = history.at(-1);
  const shown = hover ?? (last ? { time: last.time, value: last.value } : null);

  return (
    <figure className="relative" aria-label={`${label}: daily closes${cone ? " with the option-implied range to expiry" : ""}`}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <p className="min-h-5 text-[12.5px] text-muted-foreground" aria-live="polite">
          {shown &&
            (shown.band ? (
              <>
                {day(shown.time)} <span className="text-saffron">implied range {level(shown.band[0], 0)} – {level(shown.band[1], 0)}</span>
              </>
            ) : (
              <>
                {day(shown.time, true)} <span className="num text-foreground">{level(shown.value)}</span>
              </>
            ))}
        </p>
        <div className="flex gap-1" role="group" aria-label="Chart period">
          {PERIODS.map((p) => (
            <button
              key={p.label}
              type="button"
              aria-pressed={period === p.label}
              onClick={() => setPeriod(p.label)}
              className={
                "min-h-8 min-w-9 rounded-[4px] px-2 text-[12px] font-medium transition-colors pointer-coarse:min-h-11 pointer-coarse:min-w-11 " +
                (period === p.label ? "bg-panel-2 text-foreground" : "text-muted-foreground hover:text-foreground")
              }
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={el} style={{ height }} />
    </figure>
  );
}
