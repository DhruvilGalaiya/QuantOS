// A daily close coloured day by day by a category (regime, market state), with the same colours as a strip
// along the bottom. Colours are CSS variable names, resolved on the chart's own element.
import { ColorType, createChart, CrosshairMode, HistogramSeries, LineSeries, LineStyle } from "lightweight-charts";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { day, level } from "@/lib/format";
import { cssVar, useTheme } from "@/lib/theme";

export interface CatPoint {
  time: string;
  close: number;
  cat: string;
}

export function CategoryLine({
  points,
  colours,
  labels,
  ariaLabel,
  decimals = 0,
  height = "h-[340px] md:h-[400px]",
  extra,
}: {
  points: CatPoint[];
  colours: Record<string, string>;
  labels: Record<string, string>;
  ariaLabel: string;
  decimals?: number;
  height?: string;
  extra?: ReactNode;
}) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useTheme((s) => s.theme);
  const [hover, setHover] = useState<CatPoint | null>(null);
  useEffect(() => {
    if (!el.current || !points.length) return;
    const v = (n: string) => cssVar(n, el.current!);
    const col = Object.fromEntries(Object.entries(colours).map(([k, c]) => [k, v(c)]));
    const chart = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: v("--faint"), fontFamily: "IBM Plex Sans", fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: v("--line") } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.06, bottom: 0.16 } },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: v("--muted-foreground"), style: LineStyle.Dotted, labelVisible: false }, horzLine: { visible: false, labelVisible: false } },
      handleScroll: false,
      handleScale: false,
      localization: { priceFormatter: (p: number) => level(p, decimals) },
    });
    const line = chart.addSeries(LineSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerRadius: 4 });
    line.setData(points.map((p) => ({ time: p.time, value: p.close, color: col[p.cat] ?? v("--foreground") })));
    const strip = chart.addSeries(HistogramSeries, { priceScaleId: "strip", priceLineVisible: false, lastValueVisible: false, base: 0 });
    chart.priceScale("strip").applyOptions({ scaleMargins: { top: 0.9, bottom: 0 }, visible: false });
    strip.setData(points.map((p) => ({ time: p.time, value: 1, color: col[p.cat] ?? v("--line") })));
    chart.timeScale().fitContent();
    const by = new Map(points.map((p) => [p.time, p]));
    chart.subscribeCrosshairMove((e) => setHover(typeof e.time === "string" ? (by.get(e.time) ?? null) : null));
    return () => chart.remove();
  }, [points, colours, theme, decimals]);
  const shown = hover ?? points.at(-1);
  return (
    <figure aria-label={ariaLabel}>
      <div className="mb-3 flex min-h-11 flex-wrap items-center justify-between gap-3 text-[13.5px]">
        <p className="text-muted-foreground" aria-live="polite">
          {shown && (
            <>
              {day(shown.time, true)} <span className="num font-semibold text-foreground">{level(shown.close, 2)}</span>{" "}
              <span className="inline-flex items-center gap-1.5 font-medium text-foreground">
                <span className="size-2 rounded-full" style={{ background: `var(${colours[shown.cat]})` }} aria-hidden />
                {labels[shown.cat]}
              </span>
            </>
          )}
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <ul className="flex flex-wrap gap-4 text-[13.5px]" aria-label="Legend">
            {Object.entries(labels).map(([k, l]) => (
              <li key={k} className="inline-flex items-center gap-1.5 font-medium">
                <span className="size-2 rounded-full" style={{ background: `var(${colours[k]})` }} aria-hidden />
                {l}
              </li>
            ))}
          </ul>
          {extra}
        </div>
      </div>
      <div ref={el} className={height} />
    </figure>
  );
}
