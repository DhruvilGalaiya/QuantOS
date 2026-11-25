import type { RegimePoint, Regimes } from "@quantos/types";
import { ColorType, createChart, CrosshairMode, HistogramSeries, LineSeries, LineStyle } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useParams } from "react-router";
import FloatingLines from "@/components/bits/FloatingLines";
import { BigStat, nb, CineHero, CineSection, InstrumentTabs, Method, Reading, Spot, StatRow, useReducedMotion } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day, level, pct, share, tone } from "@/lib/format";
import { cssVar, useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const COLOUR: Record<string, string> = {
  BULL: "var(--up)", SIDE: "var(--normal)", BEAR: "var(--down)", LOW: "var(--calm)", MID: "var(--normal)", HIGH: "var(--down)",
};
const GLOW: Record<string, [string, string]> = {
  BULL: ["#2fbf8a", "#a6f0cf"], SIDE: ["#d9b44a", "#f5dd94"], BEAR: ["#ee5e5a", "#f8aaa7"],
  LOW: ["#4cc3c9", "#a9eef0"], MID: ["#d9b44a", "#f5dd94"], HIGH: ["#ee5e5a", "#f8aaa7"],
};
const WORD: Record<string, string> = { BULL: "bull", SIDE: "sideways", BEAR: "bear", LOW: "low-volatility", MID: "mid-volatility", HIGH: "high-volatility" };

function Chip({ regime, label }: { regime?: string | null; label?: string }) {
  if (!regime) return <span className="text-faint">—</span>;
  return (
    <span className="inline-flex items-center gap-1.5 text-[13.5px] font-medium">
      <span className="size-2 rounded-full" style={{ background: COLOUR[regime] }} aria-hidden />
      {label ?? regime}
    </span>
  );
}

// ------------------------------------------------------------------ timeline (price coloured by regime + strip)
function RegimeChart({ points, labels, isVix }: { points: RegimePoint[]; labels: Record<string, string>; isVix: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useTheme((s) => s.theme);
  const [hover, setHover] = useState<RegimePoint | null>(null);
  useEffect(() => {
    if (!el.current || !points.length) return;
    const c = (v: string) => cssVar(v, el.current!);
    const col: Record<string, string> = { BULL: c("--up"), SIDE: c("--normal"), BEAR: c("--down"), LOW: c("--calm"), MID: c("--normal"), HIGH: c("--down") };
    const chart = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: c("--faint"), fontFamily: "IBM Plex Sans", fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: c("--line") } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.06, bottom: 0.16 } },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: c("--muted-foreground"), style: LineStyle.Dotted, labelVisible: false }, horzLine: { visible: false, labelVisible: false } },
      handleScroll: false,
      handleScale: false,
      localization: { priceFormatter: (p: number) => level(p, isVix ? 1 : 0) },
    });
    const line = chart.addSeries(LineSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerRadius: 4 });
    line.setData(points.map((p) => ({ time: p.time, value: p.close, color: col[p.regime ?? ""] ?? c("--foreground") })));
    const strip = chart.addSeries(HistogramSeries, { priceScaleId: "strip", priceLineVisible: false, lastValueVisible: false, base: 0 });
    chart.priceScale("strip").applyOptions({ scaleMargins: { top: 0.9, bottom: 0 }, visible: false });
    strip.setData(points.map((p) => ({ time: p.time, value: 1, color: col[p.regime ?? ""] ?? c("--line") })));
    chart.timeScale().fitContent();
    const byTime = new Map(points.map((p) => [p.time, p]));
    chart.subscribeCrosshairMove((e) => setHover(typeof e.time === "string" ? (byTime.get(e.time) ?? null) : null));
    return () => chart.remove();
  }, [points, theme, isVix]);
  const shown = hover ?? points.at(-1);
  return (
    <figure aria-label="Daily close over the last two years, coloured by the regime assigned each day">
      <div className="mb-3 flex min-h-11 flex-wrap items-center justify-between gap-3 text-[13.5px]">
        <p className="text-muted-foreground" aria-live="polite">
          {shown && (
            <>
              {day(shown.time, true)} <span className="num font-semibold text-foreground">{level(shown.close, isVix ? 2 : 2)}</span>{" "}
              <Chip regime={shown.regime} label={labels[shown.regime ?? ""]} />
            </>
          )}
        </p>
        <ul className="flex flex-wrap gap-4" aria-label="Legend">
          {Object.entries(labels).map(([k, v]) => (
            <li key={k}>
              <Chip regime={k} label={v} />
            </li>
          ))}
        </ul>
      </div>
      <div ref={el} className="h-[340px] md:h-[400px]" />
    </figure>
  );
}

// ------------------------------------------------------------------ transitions
function Matrix({ data }: { data: Regimes }) {
  if (!data.matrix) return null;
  const s = data.states;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-separate border-spacing-1.5 text-[14px]">
        <caption className="sr-only">Chance of moving from today's regime (rows) to tomorrow's (columns)</caption>
        <thead>
          <tr>
            <th className="pb-1 text-left text-[12.5px] font-normal text-faint">From ↓ to →</th>
            {s.map((c) => (
              <th key={c} className="pb-1 text-center text-[13px] font-medium">
                <Chip regime={c} label={data.labels[c]} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {s.map((r, i) => (
            <tr key={r}>
              <th className="pr-2 text-left font-medium">
                <Chip regime={r} label={data.labels[r]} />
              </th>
              {s.map((c, j) => {
                const v = data.matrix![i][j];
                return (
                  <td
                    key={c}
                    className={cn("num h-14 rounded-lg text-center text-[15px] font-semibold", r === data.current?.regime && "ring-1 ring-saffron/50")}
                    style={{ background: `color-mix(in srgb, ${COLOUR[c]} ${Math.round(8 + v * 60)}%, transparent)` }}
                  >
                    {share(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ forecast + record
function ForecastPanel({ data }: { data: Regimes }) {
  const f = data.forecast;
  const sc = data.scorecard;
  if (!f) return null;
  const order = data.is_vix ? ["LOW", "MID", "HIGH"] : ["BULL", "SIDE", "BEAR"];
  const lbl = (k: string) => data.labels[k] ?? { BULL: "Bull", SIDE: "Sideways", BEAR: "Bear" }[k] ?? k;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Spot>
        <p className="text-[13.5px] text-muted-foreground">
          {f.target_date ? `Next session, ${day(f.target_date)}` : "Next session"} · {f.model === "V11" ? `V11 model, ${f.status?.toLowerCase()}` : "V10 trend model"}
        </p>
        <p className="mt-2 font-headline text-[34px] font-semibold leading-none">
          <span style={{ color: COLOUR[f.pick ?? ""] }}>{lbl(f.pick ?? "")}</span>
          <span className="ml-3 text-[18px] font-medium text-muted-foreground">{share(f.confidence)} likely</span>
        </p>
        <ul className="mt-6 space-y-3">
          {order.map((k) => (
            <li key={k}>
              <div className="flex justify-between text-[13.5px]">
                <span>{lbl(k)}</span>
                <span className="num font-medium">{share(f.probs[k])}</span>
              </div>
              <div className="mt-1.5 h-2 rounded-full bg-foreground/[0.07]">
                <div className="h-2 rounded-full transition-[width] duration-700" style={{ width: `${(f.probs[k] ?? 0) * 100}%`, background: COLOUR[k] }} />
              </div>
            </li>
          ))}
        </ul>
        {f.model === "V11" && <p className="mt-5 text-[13.5px] leading-snug text-muted-foreground">Bull means a close above +0.25%, Bear below −0.25%. The forecast becomes final after the US session closes, around 02:30 IST.</p>}
      </Spot>
      {sc && (
        <Spot>
          <p className="text-[13.5px] text-muted-foreground">
            Track record, {sc.n.toLocaleString("en-IN")} sessions {sc.walk_forward ? "it never saw in training" : "from its log"}
          </p>
          <div className="mt-5 space-y-5">
            {[
              ["Called the direction right", sc.direction_accuracy, "var(--accent-saffron)"],
              ["Always guessing up", sc.always_up, "var(--faint)"],
            ].map(([label, v, c]) => (
              <div key={label as string}>
                <div className="flex justify-between text-[14px]">
                  <span>{label as string}</span>
                  <span className="num font-semibold">{pct(v as number, 1)}</span>
                </div>
                <div className="mt-1.5 h-2.5 rounded-full bg-foreground/[0.07]">
                  <div className="h-2.5 rounded-full" style={{ width: `${((v as number) ?? 0) * 100}%`, background: c as string }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-6 text-[15px] leading-relaxed text-muted-foreground">
            {sc.direction_accuracy != null && sc.always_up != null && (
              <>
                That is <b className="text-foreground">{((sc.direction_accuracy - sc.always_up) * 100).toFixed(1)} points</b> {sc.direction_accuracy >= sc.always_up ? "better" : "worse"} than always guessing up,
                from {day(sc.start, true)} to {day(sc.end, true)}. A small edge: treat each call as a tilt, not a promise.
              </>
            )}
          </p>
        </Spot>
      )}
    </div>
  );
}

function Log({ data }: { data: Regimes }) {
  const [rows, setRows] = useState(0);
  const all = data.log.rows;
  if (!all.length) return null;
  const lbl = (k?: string | null) => (k ? (data.labels[k] ?? { BULL: "Bull", SIDE: "Sideways", BEAR: "Bear" }[k] ?? k) : "—");
  return (
    <div className="mt-8">
      {rows === 0 ? (
        <button type="button" onClick={() => setRows(30)} className="btn-ghost">
          Show the last 30 forecasts
        </button>
      ) : (
        <>
          <div className="overflow-x-auto rounded-xl border border-line">
            <table className="w-full min-w-[560px] text-[14px]">
              <thead className="bg-panel text-left text-[12.5px] text-faint">
                <tr>
                  <th className="px-4 py-2.5 font-normal">Made on</th>
                  <th className="px-4 py-2.5 font-normal">For</th>
                  <th className="px-4 py-2.5 font-normal">Forecast</th>
                  <th className="px-4 py-2.5 text-right font-normal">Confidence</th>
                  <th className="px-4 py-2.5 text-right font-normal">What happened</th>
                  <th className="px-4 py-2.5 font-normal">Result</th>
                </tr>
              </thead>
              <tbody className="num">
                {all.slice(0, rows).map((r) => (
                  <tr key={r.date} className="border-t border-line">
                    <td className="px-4 py-2.5 text-muted-foreground">{day(r.date)}</td>
                    <td className="px-4 py-2.5">{day(r.target)}</td>
                    <td className="px-4 py-2.5">
                      <Chip regime={r.forecast} label={lbl(r.forecast)} />
                    </td>
                    <td className="px-4 py-2.5 text-right">{share(r.confidence)}</td>
                    <td className={cn("px-4 py-2.5 text-right", tone(r.move) === "up" && "text-up", tone(r.move) === "down" && "text-down")}>
                      {r.move == null ? <span className="text-faint">Pending</span> : `${r.move > 0 ? "+" : r.move < 0 ? "−" : ""}${Math.abs(r.move).toFixed(2)}%`}
                    </td>
                    <td className="px-4 py-2.5">{r.hit == null ? <span className="text-faint">—</span> : r.hit ? <span className="text-up">Right</span> : <span className="text-down">Wrong</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows < all.length && (
            <button type="button" onClick={() => setRows((n) => n + 60)} className="btn-ghost mt-4">
              Show 60 more
            </button>
          )}
        </>
      )}
    </div>
  );
}

function Overview({ data }: { data: Regimes }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-[720px] text-[14px]">
        <thead className="bg-panel text-left text-[12.5px] text-faint">
          <tr>
            <th className="px-4 py-2.5 font-normal">Index</th>
            <th className="px-4 py-2.5 text-right font-normal">Last close</th>
            <th className="px-4 py-2.5 text-right font-normal">Day</th>
            <th className="px-4 py-2.5 font-normal">Regime</th>
            <th className="px-4 py-2.5 text-right font-normal">For</th>
            <th className="px-4 py-2.5 font-normal">Persistence</th>
            <th className="px-4 py-2.5 font-normal">Next session</th>
          </tr>
        </thead>
        <tbody className="num">
          {data.overview.map((r) => {
            const vix = r.symbol === "INDIA_VIX";
            const word = (k?: string | null) => (k ? ({ BULL: "Bull", SIDE: "Sideways", BEAR: "Bear", LOW: "Low vol", MID: "Mid vol", HIGH: "High vol" }[k] ?? k) : "—");
            return (
              <tr key={r.symbol} className={cn("border-t border-line transition-colors hover:bg-panel/70", r.symbol === data.symbol && "bg-saffron/[0.06]")}>
                <td className="px-4 py-2.5 font-medium">
                  <Link to={`/regimes/${r.symbol}`} className="link-draw inline-flex min-h-8 items-center">
                    {r.name}
                  </Link>
                </td>
                <td className="px-4 py-2.5 text-right">{level(r.close)}</td>
                <td className={cn("px-4 py-2.5 text-right", !vix && tone(r.change_pct) === "up" && "text-up", !vix && tone(r.change_pct) === "down" && "text-down")}>{pct(r.change_pct, 2, true)}</td>
                <td className="px-4 py-2.5">
                  <Chip regime={r.regime} label={word(r.regime)} />
                </td>
                <td className="px-4 py-2.5 text-right text-muted-foreground">{r.streak} sessions</td>
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 rounded-full bg-foreground/[0.08]">
                      <div className="h-1.5 rounded-full bg-saffron" style={{ width: `${r.persistence ?? 0}%` }} />
                    </div>
                    <span className="text-[13px] text-muted-foreground">{r.persistence != null ? Math.round(r.persistence) : "—"}</span>
                  </div>
                </td>
                <td className="px-4 py-2.5">
                  <Chip regime={r.next} label={word(r.next)} /> <span className="text-[12px] text-faint">{r.next_model}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ page
export function RegimesPage() {
  const { symbol = "" } = useParams();
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<Regimes>(symbol ? `regimes/${symbol}` : null);
  const mix = useMemo(() => {
    if (!data?.timeline.length) return [];
    const n = data.timeline.length;
    return data.states.map((s) => ({ s, share: data.timeline.filter((p) => p.regime === s).length / n }));
  }, [data]);
  useEffect(() => {
    if (data) document.title = `${data.name} market regime · QuantOS`;
  }, [data]);

  if (!symbol) return <Navigate to="/regimes/NIFTY_50" replace />;
  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="This regime page could not be loaded">{error.message}. Try another index or reload the page.</ErrorBox>;
  const cur = data.current;
  const r = cur?.regime ?? "SIDE";
  const dyn = data.dynamics.find((d) => d.regime === r);
  const lead = data.dynamics.length ? data.dynamics.reduce((a, b) => ((a.long_run ?? 0) > (b.long_run ?? 0) ? a : b)) : null;

  return (
    <>
      <CineHero
        glow={`color-mix(in srgb, ${COLOUR[r]} 34%, transparent)`}
        background={
          !reduce && (
            <FloatingLines
              linesGradient={GLOW[r]}
              enabledWaves={["middle", "bottom"]}
              lineCount={[6, 10]}
              lineDistance={[6, 5]}
              animationSpeed={0.6}
              amplitude={0.9}
              glow={0.6}
              brightness={0.85}
              edgeFade={0.6}
              interactive
              lightMode={theme === "light"}
              opacity={0.85}
            />
          )
        }
        top={<InstrumentTabs label="Choose an index" items={data.instruments} active={data.symbol} hrefFor={(s) => `/regimes/${s}`} />}
        title={
          cur ? (
            <>
              {nb(data.name)} has been in a <span style={{ color: COLOUR[r] }}>{WORD[r]}</span> regime for {cur.streak} session{cur.streak === 1 ? "" : "s"}
            </>
          ) : (
            `${data.name} market regime`
          )
        }
        lede={`A three-state hidden Markov model reads each day's returns and volatility and labels the trend ${data.is_vix ? "Low, Mid or High volatility" : "Bull, Sideways or Bear"}. This is where ${data.name} stands at the close of ${day(data.as_of, true)}.`}
      >
        <StatRow>
          <BigStat
            label={`Close, ${day(data.as_of)}`}
            value={cur?.close}
            decimals={2}
            sub={cur?.change_pct != null ? <span className={cn(!data.is_vix && (tone(cur.change_pct) === "up" ? "text-up" : "text-down"))}>{pct(cur.change_pct, 2, true)} on the day</span> : undefined}
          />
          <BigStat label="Persistence score" value={cur?.persistence ?? null} suffix="/100" sub={`${(cur?.persistence_level ?? "").toLowerCase()}: how entrenched the regime is`} tone="saffron" />
          <BigStat label="Chance it holds tomorrow" value={cur?.stay_probability != null ? cur.stay_probability * 100 : null} suffix="%" sub="from the model's latest state" />
          <BigStat label="Typical length" value={dyn?.lasts ?? null} decimals={1} suffix=" sessions" sub={`for a ${WORD[r]} regime`} />
        </StatRow>
      </CineHero>

      <CineSection id="timeline" title="Two years of regimes" lede="The line is the daily close, coloured by the regime the model assigned that day. The strip underneath shows the same labels on their own.">
        <Spot>
          <RegimeChart points={data.timeline} labels={data.labels} isVix={data.is_vix} />
        </Spot>
        {mix.length > 0 && (
          <Reading>
            Over these two years {data.name} spent{" "}
            {mix.map((m, k) => (
              <span key={m.s}>
                <b>{share(m.share)}</b> of sessions {data.labels[m.s].toLowerCase()}
                {k < mix.length - 2 ? ", " : k === mix.length - 2 ? " and " : "."}
              </span>
            ))}
          </Reading>
        )}
      </CineSection>

      <CineSection id="transitions" title="How regimes hand over" lede="Each cell is the chance of moving from one regime today to another tomorrow, estimated from the last three years.">
        <div className="grid gap-4 xl:grid-cols-[minmax(0,6fr)_minmax(0,5fr)]">
          <Spot>
            <Matrix data={data} />
          </Spot>
          <div className="grid gap-4 sm:grid-cols-3 xl:grid-cols-1">
            {data.dynamics.map((d) => (
              <Spot key={d.regime} glow={`color-mix(in srgb, ${COLOUR[d.regime]} 60%, transparent)`}>
                <div className="flex items-baseline justify-between gap-3">
                  <Chip regime={d.regime} label={d.label} />
                  {d.regime === r && <span className="text-[12px] font-medium text-saffron">now</span>}
                </div>
                <div className="mt-3 h-1.5 rounded-full bg-foreground/[0.07]" role="img" aria-label={`Continues into the next session ${share(d.stay)} of the time`}>
                  <div className="h-1.5 rounded-full" style={{ width: `${d.stay * 100}%`, background: COLOUR[d.regime] }} />
                </div>
                <p className="mt-3 text-[14.5px] leading-snug text-muted-foreground">
                  Holds {share(d.stay)} of days. Usually lasts <b className="text-foreground">{d.lasts.toFixed(1)} sessions</b>
                  {d.exits_to && (
                    <>
                      , then turns {data.labels[d.exits_to].toLowerCase()} {share(d.exits_share)} of the time
                    </>
                  )}
                  . Long run: <b className="text-foreground">{share(d.long_run)}</b> of days.
                </p>
              </Spot>
            ))}
          </div>
        </div>
        {lead && (
          <Reading>
            The market spends most of its time {lead.label.toLowerCase()} (<b>{share(lead.long_run)}</b> of days in the long run). A {WORD[r]} regime{" "}
            {dyn && <>continues into the next session <b>{share(dyn.stay)}</b> of the time.</>}
          </Reading>
        )}
      </CineSection>

      {data.forecast && (
        <CineSection id="forecast" title="Next session, with its track record" lede="The forecast and how often this model has been right. Both are shown so you can judge how much weight the call deserves.">
          <ForecastPanel data={data} />
          <Log data={data} />
        </CineSection>
      )}

      <CineSection id="overview" title="Every tracked index today" lede="The highlighted row is the index above. Pick any row to open its regime page.">
        <Overview data={data} />
        <Method>
          <p>
            <b className="text-foreground">Trend regime (V10).</b> A three-state Gaussian hidden Markov model per index, filtered causally, so a day's label uses only data up to that close.
          </p>
          <p>
            <b className="text-foreground">Next-session forecast (V11, NIFTY 50 and NIFTY BANK).</b> Predicts the next close-to-close move from the US session that trades after India closes, India's end-of-day state
            and the regime. Trained walk-forward: every logged forecast came from a model fitted only on earlier data.
          </p>
          <p>
            <b className="text-foreground">Persistence.</b> 60% chance the regime holds, 25% ten-session consistency and 15% streak length (capped at 10). Typical length is 1 ÷ (1 − chance it holds).
          </p>
          <p>India VIX regimes read as volatility levels: the model's Bear, Sideways and Bull states are shown as Low, Mid and High volatility.</p>
        </Method>
      </CineSection>
    </>
  );
}
