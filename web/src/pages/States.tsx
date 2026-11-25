import type { States } from "@quantos/types";
import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router";
import Aurora from "@/components/bits/Aurora";
import { CategoryLine } from "@/components/charts/CategoryLine";
import { LineChart } from "@/components/charts/svg";
import { BigStat, CineHero, CineSection, InstrumentTabs, Method, nb, Reading, Segmented, Spot, StatRow, useReducedMotion, useWidth } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day, level, share } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const VAR: Record<string, string> = { CALM: "--calm", NORMAL: "--normal", STRESSED: "--stressed" };
const HEX: Record<string, string[]> = {
  CALM: ["#1f6f74", "#4cc3c9", "#a9eef0"],
  NORMAL: ["#7a5f17", "#d9b44a", "#f5dd94"],
  STRESSED: ["#7d2421", "#ee5e5a", "#f8aaa7"],
};
const WORD: Record<string, string> = { CALM: "calm", NORMAL: "normal", STRESSED: "stressed" };
const MEANING: Record<string, string> = {
  CALM: "small, orderly moves",
  NORMAL: "ordinary day-to-day swings",
  STRESSED: "large, jumpy moves and more sharp falls",
};
const pc = (v: number | null | undefined, d = 1, signed = false) => (v == null ? "—" : `${signed ? (v > 0 ? "+" : v < 0 ? "−" : "") : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)}%`);

/** Next-month outcomes by state: the middle 8 in 10 as a bar on one shared axis, the average as a tick. */
function RangeRows({ data }: { data: States }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const rows = data.behaviour;
  const lo = Math.min(...rows.map((r) => r.fwd21_p10)) * 1.15, hi = Math.max(...rows.map((r) => r.fwd21_p90)) * 1.15;
  const x = (v: number) => ((v - lo) / (hi - lo)) * w;
  return (
    <div ref={ref} className="space-y-4">
      {w > 0 &&
        rows.map((r) => {
          const all = r.state === "ALL";
          const c = all ? "var(--muted-foreground)" : `var(${VAR[r.state]})`;
          return (
            <div key={r.state}>
              <div className="mb-1.5 flex items-baseline justify-between text-[14px]">
                <span className="inline-flex items-center gap-1.5 font-medium">
                  {!all && <span className="size-2 rounded-full" style={{ background: c }} aria-hidden />}
                  {all ? "All days" : data.labels[r.state]}
                </span>
                <span className="num text-muted-foreground">
                  {pc(r.fwd21_p10, 0, true)} to {pc(r.fwd21_p90, 0, true)}, average <b className="text-foreground">{pc(r.fwd21_mean, 1, true)}</b>
                </span>
              </div>
              <svg width={w} height={18} role="img" aria-label={`${all ? "All days" : data.labels[r.state]}: 8 in 10 next months between ${pc(r.fwd21_p10, 0, true)} and ${pc(r.fwd21_p90, 0, true)}`} className="block">
                <line x1="0" x2={w} y1="9" y2="9" stroke="var(--line)" />
                <line x1={x(0)} x2={x(0)} y1="0" y2="18" stroke="var(--line-strong)" />
                <rect x={x(r.fwd21_p10)} y="4" width={Math.max(x(r.fwd21_p90) - x(r.fwd21_p10), 2)} height="10" rx="5" fill={c} opacity={all ? 0.35 : 0.55} />
                <rect x={x(r.fwd21_mean) - 1.5} y="1" width="3" height="16" rx="1.5" fill="var(--foreground)" />
              </svg>
            </div>
          );
        })}
      <p className="text-[13.5px] text-muted-foreground">The bar covers 8 in 10 of the months that followed a day in each state; the white tick is the average. The vertical line is zero.</p>
    </div>
  );
}

const PERIODS = [
  { value: "1y", label: "1 year", years: 1 },
  { value: "3y", label: "3 years", years: 3 },
  { value: "5y", label: "5 years", years: 5 },
  { value: "all", label: "Since 2011", years: 99 },
] as const;
type Period = (typeof PERIODS)[number]["value"];

export function StatesPage() {
  const { symbol = "" } = useParams();
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<States>(symbol ? `states/${symbol}` : null);
  const [period, setPeriod] = useState<Period>("5y");
  useEffect(() => {
    if (data) document.title = `${data.name} market state · QuantOS`;
  }, [data]);
  const timeline = useMemo(() => {
    if (!data?.timeline.length) return [];
    const years = PERIODS.find((p) => p.value === period)!.years;
    const last = new Date(`${data.timeline.at(-1)!.time}T00:00:00Z`);
    last.setUTCFullYear(last.getUTCFullYear() - years);
    const cut = last.toISOString().slice(0, 10);
    return data.timeline.filter((p) => p.time >= cut).map((p) => ({ time: p.time, close: p.close, cat: p.state }));
  }, [data, period]);

  if (!symbol) return <Navigate to="/states/NIFTY_50" replace />;
  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="This market state page could not be loaded">{error.message}. Try another index or reload the page.</ErrorBox>;
  if (!data.as_of || !data.current || !data.transition || !data.drawdown) return <ErrorBox title="No market states yet">The analytics job has not built this index yet.</ErrorBox>;

  const cur = data.current;
  const s = cur.state;
  const b = data.behaviour.find((x) => x.state === s)!;
  const ep = data.episodes.find((x) => x.state === s)!;
  const si = data.states.indexOf(s);
  const stress = data.behaviour.find((x) => x.state === "STRESSED")!, calm = data.behaviour.find((x) => x.state === "CALM")!;
  const D = data.drawdown;
  const b10 = D.base_rates.find((r) => r.from === 0.1);
  const surv = data.survival;
  const maxN = Math.max(...Object.values(surv).map((v) => v.length));

  return (
    <>
      <CineHero
        glow={`color-mix(in srgb, var(${VAR[s]}) 30%, transparent)`}
        background={!reduce && <Aurora colorStops={HEX[s]} amplitude={1.1} blend={0.6} speed={0.6} glow={0.6} brightness={0.85} layers={3} lightMode={theme === "light"} opacity={0.75} />}
        top={<InstrumentTabs label="Choose an index" items={data.symbols} active={data.symbol} hrefFor={(x) => `/states/${x}`} />}
        title={
          <>
            {nb(data.name)} has been in a <span style={{ color: `var(${VAR[s]})` }}>{WORD[s]}</span> state for {cur.streak} session{cur.streak === 1 ? "" : "s"}
          </>
        }
        lede={`Markets switch between calm stretches and stormy ones. In a ${WORD[s]} state the market has typically seen ${MEANING[s]}. This describes the present and the past; it is not a forecast.`}
      >
        <StatRow>
          <BigStat label="Typical day in this state" text={`±${(b.typical_day * 100).toFixed(1)}%`} sub={`${pc(b.vol_annual)} a year`} tone="saffron" />
          <BigStat label="Model confidence" value={(cur.probs[s] ?? 0) * 100} suffix="%" sub={data.states.map((x) => `${data.labels[x]} ${share(cur.probs[x])}`).join(", ")} />
          <BigStat label="Still running a week later" value={cur.still_after_5 != null ? cur.still_after_5 * 100 : null} suffix="%" sub={`of past ${WORD[s]} spells that reached ${cur.streak} sessions`} />
          <BigStat label="Typical length" value={ep.median_days ?? null} decimals={0} suffix=" sessions" sub={`median of ${ep.n} past spells`} />
        </StatRow>
      </CineHero>

      <CineSection id="behaviour" title="How the market behaves in each state" lede="Measured on the days after each labelled day, so no future information is used.">
        <div className="grid gap-4 lg:grid-cols-3">
          {data.states.map((x) => {
            const r = data.behaviour.find((y) => y.state === x)!;
            return (
              <Spot key={x} glow={`color-mix(in srgb, var(${VAR[x]}) 60%, transparent)`} className={cn(x === s && "ring-1 ring-saffron/40")}>
                <div className="flex items-baseline justify-between">
                  <span className="inline-flex items-center gap-2 text-[17px] font-semibold">
                    <span className="size-2.5 rounded-full" style={{ background: `var(${VAR[x]})` }} aria-hidden />
                    {data.labels[x]}
                  </span>
                  {x === s && <span className="text-[12.5px] font-medium text-saffron">now</span>}
                </div>
                <div className="mt-4 h-1.5 rounded-full bg-foreground/[0.07]" role="img" aria-label={`${share(r.share)} of days`}>
                  <div className="h-1.5 rounded-full" style={{ width: `${r.share * 100}%`, background: `var(${VAR[x]})` }} />
                </div>
                <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-[14px]">
                  <div>
                    <dt className="text-muted-foreground">Share of days</dt>
                    <dd className="num text-[20px] font-semibold">{share(r.share)}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Typical day</dt>
                    <dd className="num text-[20px] font-semibold">±{(r.typical_day * 100).toFixed(1)}%</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Days falling 2%+</dt>
                    <dd className="num text-[20px] font-semibold">{(r.big_down_share * 100).toFixed(1)}%</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Next month higher</dt>
                    <dd className="num text-[20px] font-semibold">{share(r.fwd21_up)}</dd>
                  </div>
                </dl>
              </Spot>
            );
          })}
        </div>
        <Spot className="mt-4">
          <h3 className="mb-4 text-[16px] font-semibold">The month that followed</h3>
          <RangeRows data={data} />
        </Spot>
        <Reading>
          Stressed spells are rare (<b>{share(stress.share)}</b> of days) but carry most of the risk: a 2% fall came on <b>{(stress.big_down_share * 100).toFixed(0)}%</b> of the days that followed, against{" "}
          {(calm.big_down_share * 100).toFixed(1)}% in calm spells. The state tells you how big the swings tend to be, which is what position sizes and stop-losses should respond to.
        </Reading>
      </CineSection>

      <CineSection id="timeline" title="The state on every day" lede="The line is the daily close, coloured by the state the model assigned that day." aside={<Segmented label="Period" options={PERIODS.map((p) => ({ value: p.value, label: p.label }))} value={period} onChange={setPeriod} />}>
        <Spot>
          <CategoryLine points={timeline} colours={VAR} labels={data.labels} ariaLabel={`${data.name} daily close coloured by market state`} />
        </Spot>
      </CineSection>

      <CineSection id="duration" title="How long states last" lede="Each line is the share of past spells still running after a given number of sessions.">
        <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <Spot>
            <LineChart
              ariaLabel="Share of past spells still running by number of sessions, for each state"
              labels={Array.from({ length: maxN }, (_, i) => String(i + 1))}
              xTicks={6}
              xFormat={(n) => `${n}d`}
              height={300}
              yDomain={[0, 100]}
              series={data.states.map((x) => ({ id: x, data: surv[x].map((v) => v * 100), color: `var(${VAR[x]})`, width: x === s ? 2.6 : 1.6 }))}
              yFormat={(t) => `${t}%`}
              readout={(i) => (
                <span className="flex flex-wrap gap-x-4">
                  <span>After {i + 1} sessions</span>
                  {data.states.map((x) => (
                    <span key={x}>
                      <span className="mr-1 inline-block size-2 rounded-full align-middle" style={{ background: `var(${VAR[x]})` }} aria-hidden />
                      {data.labels[x]} <b className="num text-foreground">{surv[x][i] != null ? share(surv[x][i]) : "—"}</b>
                    </span>
                  ))}
                </span>
              )}
            />
          </Spot>
          <div className="grid gap-3">
            {data.episodes.map((e) => (
              <Spot key={e.state} glow={`color-mix(in srgb, var(${VAR[e.state]}) 60%, transparent)`}>
                <span className="inline-flex items-center gap-2 font-semibold">
                  <span className="size-2 rounded-full" style={{ background: `var(${VAR[e.state]})` }} aria-hidden />
                  {data.labels[e.state]}
                </span>
                <p className="mt-1.5 text-[14.5px] leading-snug text-muted-foreground">
                  {e.n} spells, typically <b className="text-foreground">{e.median_days?.toFixed(0)} sessions</b>, longest {e.longest}. Then turned{" "}
                  {Object.entries(e.next)
                    .map(([k, v]) => `${data.labels[k].toLowerCase()} ${share(v)}`)
                    .join(", ")}
                  .
                </p>
              </Spot>
            ))}
          </div>
        </div>
        <Spot className="mt-4">
          <h3 className="text-[16px] font-semibold">Chance of the same state tomorrow</h3>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] border-separate border-spacing-1.5 text-[14px]">
              <thead>
                <tr className="text-[12.5px] text-muted-foreground">
                  <th className="text-left font-normal">From today ↓ to tomorrow →</th>
                  {data.states.map((x) => (
                    <th key={x} className="text-center font-medium text-foreground">
                      {data.labels[x]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.states.map((r, i) => (
                  <tr key={r}>
                    <th className="pr-2 text-left font-medium">{data.labels[r]}</th>
                    {data.states.map((c, j) => {
                      const v = data.transition!.p[i][j];
                      return (
                        <td key={c} className={cn("num h-16 rounded-lg text-center", r === s && "ring-1 ring-saffron/50")} style={{ background: `color-mix(in srgb, var(${VAR[c]}) ${Math.round(6 + v * 55)}%, transparent)` }}>
                          <span className="block text-[15px] font-semibold">{(v * 100).toFixed(1)}%</span>
                          <span className="block text-[11.5px] text-muted-foreground">
                            {(data.transition!.lo[i][j] * 100).toFixed(1)}–{(data.transition!.hi[i][j] * 100).toFixed(1)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Spot>
        <Reading>
          From a {WORD[s]} day the market stayed {WORD[s]} the next day <b>{share(data.transition.p[si][si])}</b> of the time, which implies spells of about <b>{data.transition.duration[si].toFixed(0)} sessions</b> (90% band{" "}
          {data.transition.dur_lo[si].toFixed(0)}–{data.transition.dur_hi[si].toFixed(0)}). Today's state is the best description of the next few days, and says little about the next few months.
        </Reading>
      </CineSection>

      <CineSection
        id="falls"
        title="Every major fall since 2007"
        lede="How far the index has been below its previous high, and how long each fall took to recover."
        aside={
          <p className="max-w-[28ch] text-[14px] text-muted-foreground">
            Now <b className="text-foreground">{D.now < -0.005 ? `${pc(D.now, 1, true)} from the high` : "at the high"}</b> of {level(D.peak, 0)}.
          </p>
        }
      >
        <div className="grid gap-4 sm:grid-cols-3">
          {D.base_rates.map((r) => (
            <Spot key={r.from}>
              <p className="text-[13.5px] text-muted-foreground">
                Falls of {Math.round(r.from * 100)}% that reached {Math.round(r.to * 100)}%
              </p>
              <p className="mt-1.5 font-headline text-[30px] font-semibold leading-none">
                {r.went_on} of {r.n}
              </p>
              <p className="mt-2 text-[13.5px] text-muted-foreground">{r.median_recover_days != null ? `typical recovery ${r.median_recover_days.toFixed(0)} sessions` : "no completed recoveries yet"}</p>
            </Spot>
          ))}
        </div>
        <Spot className="mt-4">
          <LineChart
            ariaLabel={`${data.name} distance below its previous high since 2007`}
            labels={D.series.map((p) => p.time)}
            height={280}
            xFormat={(t) => t.slice(0, 4)}
            yDomain={[Math.min(...D.series.map((p) => p.value)) * 100 * 1.08, 2]}
            series={[
              { id: "dd", data: D.series.map((p) => p.value * 100), color: "var(--down)", width: 1.2, area: "color-mix(in srgb, var(--down) 22%, transparent)", areaTo: 0 },
              { id: "c10", data: D.series.map(() => -10), color: "var(--faint)", dash: "3 4", width: 1 },
              { id: "c20", data: D.series.map(() => -20), color: "var(--faint)", dash: "3 4", width: 1 },
            ]}
            yFormat={(t) => `${t}%`}
            readout={(i) => (
              <span>
                {day(D.series[i]?.time, true)}: <b className="num text-foreground">{pc(D.series[i]?.value, 1, true)}</b> below the previous high (dashed lines: 10% correction, 20% bear market)
              </span>
            )}
          />
        </Spot>
        <div className="mt-4 overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[620px] text-[14px]">
            <thead className="bg-panel text-left text-[12.5px] text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5 font-normal">Peak</th>
                <th className="px-4 py-2.5 font-normal">Bottom</th>
                <th className="px-4 py-2.5 text-right font-normal">Fall</th>
                <th className="px-4 py-2.5 text-right font-normal">Sessions falling</th>
                <th className="px-4 py-2.5 text-right font-normal">Sessions to recover</th>
              </tr>
            </thead>
            <tbody className="num">
              {D.falls.map((f) => (
                <tr key={f.peak} className="border-t border-line">
                  <td className="px-4 py-2.5">{day(f.peak, true)}</td>
                  <td className="px-4 py-2.5">{day(f.trough, true)}</td>
                  <td className="px-4 py-2.5 text-right font-medium text-down">{pc(f.depth, 1, true)}</td>
                  <td className="px-4 py-2.5 text-right">{f.days_down}</td>
                  <td className="px-4 py-2.5 text-right">{f.ongoing ? <span className="text-muted-foreground">still recovering</span> : f.days_to_recover?.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {b10 && b10.n > 0 && (
          <Reading>
            Since 2007, {data.name} had {b10.n} falls of 10% or more; <b>{b10.went_on}</b> of them ({share(b10.went_on / b10.n)}) deepened into a 20% bear market. The typical 10% fall took{" "}
            <b>{b10.median_recover_days?.toFixed(0)} sessions</b> to get back to the old high. Most corrections do not become bear markets, but the ones that do take far longer to heal.
          </Reading>
        )}
        <Method>
          <p>States: a three-state Gaussian hidden Markov model on daily return, 20-day volatility and the daily change in India VIX, refit every January on earlier years only. Each day's state uses data up to that day.</p>
          <p>
            The label changes only when another state's probability reaches {share(data.switch_prob)}, which stops one-day flickers. Durations use Kaplan–Meier survival of completed spells; transition bands are 90% moving-block
            bootstrap intervals. Falls are peak to trough on daily closes; recovery is the first close back at the old peak.
          </p>
        </Method>
      </CineSection>
    </>
  );
}
