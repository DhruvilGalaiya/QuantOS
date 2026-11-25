import type { Events, EventStudy } from "@quantos/types";
import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router";
import Radar from "@/components/bits/Radar";
import { LineChart } from "@/components/charts/svg";
import { BigStat, CineHero, CineSection, InstrumentTabs, Method, nb, Reading, Segmented, Spot, StatRow, useReducedMotion, useWidth } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day, share } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const H_LABEL: Record<string, string> = { "1": "next day", "5": "next week", "21": "next month", "63": "next three months" };
const H_SHORT: Record<string, string> = { "1": "1 day", "5": "1 week", "21": "1 month", "63": "3 months" };
const pc = (v: number | null | undefined, d = 1, signed = true) => (v == null ? "—" : `${signed ? (v > 0 ? "+" : v < 0 ? "−" : "") : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)}%`);
const tone = (v?: number | null) => (v == null || v === 0 ? "" : v > 0 ? "text-up" : "text-down");

/** Forest plot: each event's average with its 95% interval, against the all-days average (the vertical line). */
function Forest({ events, h, selected, onSelect }: { events: EventStudy[]; h: string; selected: string; onSelect: (id: string) => void }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const stats = events.map((e) => e.stats[`h${h}`]).filter((s) => s?.n);
  const lo = Math.min(...stats.map((s) => s.ci_low ?? 0), 0), hi = Math.max(...stats.map((s) => s.ci_high ?? 0), 0);
  const pad = (hi - lo) * 0.08;
  const plotW = Math.max(w, 60);
  const x = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * plotW;
  const base = stats[0]?.baseline_mean ?? 0;
  return (
    <div className="overflow-hidden rounded-2xl border border-line">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-0 md:grid-cols-[16rem_minmax(0,1fr)_9rem]">
        <div className="hidden border-b border-line bg-panel px-4 py-2.5 text-[12.5px] text-muted-foreground md:block">Event</div>
        <div className="hidden border-b border-line bg-panel px-4 py-2.5 text-[12.5px] text-muted-foreground md:block">
          Average after the event, with its 95% range. The line is an ordinary day ({pc(base, 2)}).
        </div>
        <div className="hidden border-b border-line bg-panel px-4 py-2.5 text-right text-[12.5px] text-muted-foreground md:block">History says</div>
      </div>
      <ul>
        {events.map((e) => {
          const s = e.stats[`h${h}`];
          const active = e.id === selected;
          return (
            <li key={e.id} className="border-b border-line last:border-0">
              <button
                type="button"
                onClick={() => onSelect(e.id)}
                aria-pressed={active}
                className={cn(
                  "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 px-4 py-3 text-left transition-colors hover:bg-panel md:grid-cols-[16rem_minmax(0,1fr)_9rem]",
                  active && "bg-saffron/[0.07]",
                )}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">{e.name}</span>
                  <span className="block text-[12.5px] text-muted-foreground">
                    {e.group} · {e.count} times
                  </span>
                </span>
                <span className="order-3 col-span-2 md:order-none md:col-span-1" ref={e === events[0] ? ref : undefined}>
                  {s?.n && w > 0 ? (
                    <svg width={plotW} height={22} role="img" aria-label={`${e.name}: average ${pc(s.mean, 2)}, 95% range ${pc(s.ci_low, 1)} to ${pc(s.ci_high, 1)}`} className="block">
                      <line x1={x(0)} x2={x(0)} y1="0" y2="22" stroke="var(--line)" />
                      <line x1={x(base)} x2={x(base)} y1="0" y2="22" stroke="var(--muted-foreground)" strokeDasharray="3 3" />
                      <rect x={x(s.ci_low ?? 0)} y="8" width={Math.max(x(s.ci_high ?? 0) - x(s.ci_low ?? 0), 2)} height="6" rx="3" fill={s.unusual ? "var(--accent-saffron)" : "var(--line-strong)"} />
                      <circle cx={x(s.mean ?? 0)} cy="11" r="5" fill={s.unusual ? "var(--accent-saffron)" : "var(--foreground)"} stroke="var(--background)" strokeWidth="1.5" />
                    </svg>
                  ) : (
                    <span className="text-[13px] text-muted-foreground">Not enough cases at this horizon</span>
                  )}
                </span>
                <span className="text-right text-[13.5px]">
                  <span className={cn("num block font-semibold", tone(s?.mean))}>{pc(s?.mean, 2)}</span>
                  <span className={cn("block text-[12.5px]", s?.unusual ? "text-saffron" : "text-muted-foreground")}>{s?.unusual ? "Unusual" : "Like a normal day"}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Outcomes({ values, mean, base }: { values: number[]; mean?: number | null; base?: number | null }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  if (values.length < 3) return <p className="text-muted-foreground">Too few completed cases to draw the spread.</p>;
  const lo = Math.min(...values), hi = Math.max(...values);
  const bins = 24, step = (hi - lo) / bins || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((v) => counts[Math.min(bins - 1, Math.floor((v - lo) / step))]++);
  const H = 200, max = Math.max(...counts), bw = w / bins;
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * w;
  return (
    <div ref={ref}>
      {w > 0 && (
        <svg width={w} height={H + 22} role="img" aria-label="How the outcomes after this event were spread" className="block">
          {counts.map((c, i) => {
            const mid = lo + (i + 0.5) * step, h = (c / max) * H;
            return <rect key={i} x={i * bw + 0.5} y={H - h} width={Math.max(bw - 2, 1)} height={h} rx="2" fill={mid >= 0 ? "var(--up)" : "var(--down)"} opacity={0.8} />;
          })}
          <line x1="0" x2={w} y1={H + 0.5} y2={H + 0.5} stroke="var(--line-strong)" />
          {mean != null && <line x1={x(mean)} x2={x(mean)} y1="0" y2={H} stroke="var(--accent-saffron)" strokeWidth="2" />}
          {base != null && <line x1={x(base)} x2={x(base)} y1="0" y2={H} stroke="var(--muted-foreground)" strokeWidth="2" strokeDasharray="4 3" />}
          <text x="0" y={H + 16} fontSize="11" fill="var(--faint)">
            {pc(lo, 0)}
          </text>
          <text x={w} y={H + 16} fontSize="11" fill="var(--faint)" textAnchor="end">
            {pc(hi, 0)}
          </text>
        </svg>
      )}
      <p className="mt-2 text-[13px] text-muted-foreground">
        <span className="mr-1.5 inline-block h-0.5 w-4 bg-saffron align-middle" aria-hidden />
        this event's average
        <span className="ml-4 mr-1.5 inline-block w-4 border-t-2 border-dashed border-muted-foreground align-middle" aria-hidden />
        an ordinary day
      </p>
    </div>
  );
}

export function EventsPage() {
  const { symbol = "" } = useParams();
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<Events>(symbol ? `events/${symbol}` : null);
  const [h, setH] = useState("21");
  // the selected event belongs to one index; switching index falls back to this week's signal
  const [pick, setPick] = useState<{ symbol: string; id: string } | null>(null);
  const sel = pick?.symbol === symbol ? pick.id : null;
  const setSel = (id: string) => setPick({ symbol, id });
  const [rows, setRows] = useState(12);
  useEffect(() => {
    if (data) document.title = `${data.name} event lab · QuantOS`;
  }, [data]);
  const ordered = useMemo(() => {
    if (!data) return [];
    return [...data.events].sort((a, b) => (a.stats[`h${h}`]?.p_value ?? 1) - (b.stats[`h${h}`]?.p_value ?? 1));
  }, [data, h]);

  if (!symbol) return <Navigate to="/events/NIFTY_50" replace />;
  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="The event lab could not be loaded">{error.message}. Try another index or reload the page.</ErrorBox>;
  if (!data.as_of) return <ErrorBox title="No event studies yet">The analytics job has not built this index yet.</ErrorBox>;

  const recentIds = new Set(data.recent.map((r) => r.id));
  const chosen = data.events.find((e) => e.id === (sel ?? data.recent[0]?.id ?? ordered[0]?.id)) ?? ordered[0];
  const s = chosen.stats[`h${h}`];
  const nUnusual = data.events.filter((e) => e.stats[`h${h}`]?.unusual).length;
  const expected = data.events.length * data.alpha;
  const fired = data.recent.filter((r, i, a) => a.findIndex((x) => x.id === r.id) === i);
  const firedUnusual = fired.filter((r) => data.events.find((e) => e.id === r.id)?.stats[`h${h}`]?.unusual).length;
  const off = chosen.path.offsets;
  const outcomes = chosen.occurrences.map((o) => o[`h${h}` as "h1"]).filter((v): v is number => v != null);

  return (
    <>
      <CineHero
        background={!reduce && <Radar color="#EFA43A" speed={0.5} trail={0.6} ringCount={5} spokeCount={8} targets={Math.max(2, fired.length + 2)} glow={0.6} brightness={0.8} gridOpacity={0.25} centerX={0.78} centerY={0.5} scale={1.1} lightMode={theme === "light"} opacity={0.7} />}
        top={<InstrumentTabs label="Choose an index" items={data.symbols} active={data.symbol} hrefFor={(x) => `/events/${x}`} />}
        title={
          fired.length ? (
            <>
              {fired.length} signal{fired.length > 1 ? "s" : ""} fired on {nb(data.name)} this week.{" "}
              <span className="text-saffron">{firedUnusual ? `${firedUnusual} stand${firedUnusual > 1 ? "" : "s"} out from an ordinary day.` : "History says they look like ordinary days."}</span>
            </>
          ) : (
            <>No tracked signal fired on {nb(data.name)} this week</>
          )
        }
        lede={`What actually happened after the market events traders talk about, across ${data.sessions?.toLocaleString("en-IN")} sessions since ${data.since?.slice(0, 4)}. Every result is set against ordinary days with a confidence range, so a real tendency can be told from noise.`}
      >
        <StatRow>
          <BigStat label="Events tested" value={data.events.length} sub={`${data.horizons.length} horizons each`} />
          <BigStat label={`Unusual over the ${H_LABEL[h]}`} value={nUnusual} tone="saffron" sub={`p below ${data.alpha} with at least ${data.min_n} cases`} />
          <BigStat label="Expected by luck alone" value={expected} decimals={1} sub={`${data.events.length} tests at the 5% level`} />
          <BigStat label="History since" text={data.since?.slice(0, 4) ?? "—"} sub={`${data.n_boot?.toLocaleString("en-IN")} bootstrap samples per result`} />
        </StatRow>
      </CineHero>

      <CineSection
        id="library"
        title="Every event against an ordinary day"
        lede={`The dot is the average ${data.name} return over the ${H_LABEL[h]} after each event; the bar is its 95% range. Where the bar crosses the dashed line, the event is hard to tell apart from any other day.`}
        aside={<Segmented label="Look ahead" options={data.horizons.map((x) => ({ value: String(x), label: H_SHORT[String(x)] }))} value={h} onChange={setH} />}
      >
        <Forest events={ordered} h={h} selected={chosen.id} onSelect={(id) => (setSel(id), setRows(12))} />
        <Reading>
          <b>{nUnusual}</b> of {data.events.length} events differ from an ordinary day over the {H_LABEL[h]}. With {data.events.length} tests, about <b>{expected.toFixed(1)}</b> would pass by luck alone, so treat single hits with care. Most
          famous signals leave the market looking normal: the edge, when there is one, is usually in how wide outcomes are, not in their direction.
        </Reading>
      </CineSection>

      <CineSection
        id="detail"
        title={chosen.name}
        lede={`${chosen.description}${chosen.min_gap > 1 ? ` Repeats closer than ${chosen.min_gap} sessions count once.` : ""}${recentIds.has(chosen.id) ? " It fired this week." : ""}`}
      >
        {!s?.n ? (
          <p className="text-muted-foreground">This event has not occurred often enough, or recently enough, to measure over the {H_LABEL[h]}.</p>
        ) : (
          <>
            <dl className="grid grid-cols-2 gap-6 lg:grid-cols-4">
              <BigStat label="Times it happened" value={s.n} sub={chosen.last ? `last on ${day(chosen.last, true)}` : undefined} />
              <BigStat label={`Average over the ${H_LABEL[h]}`} text={pc(s.mean, 2)} tone={(s.mean ?? 0) >= 0 ? "up" : "down"} sub={`an ordinary day: ${pc(s.baseline_mean, 2)}`} />
              <BigStat label="Higher afterwards" value={(s.hit_rate ?? 0) * 100} suffix="%" sub={`an ordinary day: ${share(s.baseline_hit_rate)}`} />
              <BigStat label="Chance it is luck" text={s.p_value != null ? s.p_value.toFixed(2) : "—"} tone={s.unusual ? "saffron" : undefined} sub="p-value; below 0.05 is unusual" />
            </dl>
            <div className="mt-8 grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <Spot>
                <h3 className="text-[16px] font-semibold">The average path, ten sessions before to three months after</h3>
                <LineChart
                  className="mt-3"
                  ariaLabel={`Average ${data.name} path around ${chosen.name}, with its 95% band and the ordinary-day drift`}
                  labels={off.map(String)}
                  xTicks={7}
                  xFormat={(o) => (o === "0" ? "event" : Number(o) > 0 ? `+${o}` : o)}
                  height={300}
                  zero
                  bands={[{ lo: chosen.path.ci_low.map((v) => (v == null ? null : v * 100)), hi: chosen.path.ci_high.map((v) => (v == null ? null : v * 100)), color: "color-mix(in srgb, var(--accent-saffron) 16%, transparent)" }]}
                  series={[
                    { id: "base", data: chosen.path.baseline.map((v) => (v == null ? null : v * 100)), color: "var(--muted-foreground)", dash: "4 4", width: 1.4 },
                    { id: "mean", data: chosen.path.mean.map((v) => (v == null ? null : v * 100)), color: "var(--accent-saffron)", width: 2.4 },
                  ]}
                  yFormat={(t) => `${t}%`}
                  readout={(i) => (
                    <span className="flex flex-wrap gap-x-4">
                      <span>{off[i] === 0 ? "Event day" : `${off[i] > 0 ? "+" : ""}${off[i]} sessions`}</span>
                      <span>
                        After this event <b className="num text-foreground">{pc(chosen.path.mean[i], 2)}</b>
                      </span>
                      <span>
                        Range {pc(chosen.path.ci_low[i], 1)} to {pc(chosen.path.ci_high[i], 1)}
                      </span>
                      <span>Ordinary day {pc(chosen.path.baseline[i], 2)}</span>
                    </span>
                  )}
                />
              </Spot>
              <Spot>
                <h3 className="text-[16px] font-semibold">How the outcomes were spread</h3>
                <p className="mt-1 text-[14px] text-muted-foreground">
                  Return over the {H_LABEL[h]}, from {pc(s.worst, 0)} to {pc(s.best, 0)}.
                </p>
                <div className="mt-4">
                  <Outcomes values={outcomes} mean={s.mean} base={s.baseline_mean} />
                </div>
              </Spot>
            </div>
            <Reading>
              After <b>{chosen.name.toLowerCase()}</b>, {data.name} returned <b>{pc(s.mean, 2)}</b> on average over the {H_LABEL[h]} (95% range {pc(s.ci_low, 1)} to {pc(s.ci_high, 1)}), against {pc(s.baseline_mean, 2)} for an ordinary
              day.{" "}
              {s.unusual
                ? `Random samples of ordinary days rarely land this far from normal (p = ${s.p_value?.toFixed(2)}), so the difference is unusual, though ${data.events.length} events are tested at once.`
                : `Random samples of ordinary days often land this far from normal (p = ${s.p_value?.toFixed(2)}), so history shows no reliable tendency here.`}
            </Reading>
            <div className="mt-8 overflow-x-auto rounded-xl border border-line">
              <table className="w-full min-w-[560px] text-[14px]">
                <thead className="bg-panel text-left text-[12.5px] text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2.5 font-normal">Date</th>
                    {["1", "5", "21", "63"].map((x) => (
                      <th key={x} className={cn("px-4 py-2.5 text-right font-normal", x === h && "text-foreground")}>
                        After {H_SHORT[x]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="num">
                  {chosen.occurrences.slice(0, rows).map((o) => (
                    <tr key={o.date} className="border-t border-line">
                      <td className="px-4 py-2.5">{day(o.date, true)}</td>
                      {(["h1", "h5", "h21", "h63"] as const).map((k) => (
                        <td key={k} className={cn("px-4 py-2.5 text-right", tone(o[k]))}>
                          {o[k] == null ? <span className="text-muted-foreground">not yet</span> : pc(o[k], 1)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {rows < chosen.occurrences.length && (
              <button type="button" className="btn-ghost mt-4" onClick={() => setRows(chosen.occurrences.length)}>
                Show all {chosen.occurrences.length} occurrences
              </button>
            )}
          </>
        )}
        <Method>
          <p>Each event is defined with information available at that day's close. Returns run from that close to the close h sessions later, so nothing from the future leaks in.</p>
          <p>
            The 95% range is a bootstrap of the event average ({data.n_boot?.toLocaleString("en-IN")} resamples). The p-value is the share of equally sized random samples of all days whose average is at least as far from the
            all-days average. Overlapping windows and changing markets make outcomes less independent than they look, and small samples have wide ranges. This describes history; it is not a trading signal.
          </p>
        </Method>
      </CineSection>
    </>
  );
}
