import type { RiskMonitor } from "@quantos/types";
import { useEffect, useMemo, useState } from "react";
import Threads from "@/components/bits/Threads";
import { LineChart } from "@/components/charts/svg";
import { BigStat, CineHero, CineSection, Method, Reading, Segmented, Spot, StatRow, useReducedMotion, useWidth } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const pc = (v: number | null | undefined, d = 1, signed = false) => (v == null ? "—" : `${signed ? (v > 0 ? "+" : v < 0 ? "−" : "") : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)}%`);
const tone = (v?: number | null) => (v == null || v === 0 ? "" : v > 0 ? "text-up" : "text-down");
const inr = (v: number) => `${v < 0 ? "−" : ""}₹${Math.abs(Math.round(v)).toLocaleString("en-IN")}`;
const AMOUNTS = [100_000, 500_000, 1_000_000, 5_000_000];
const STORE = "quantos-risk-amount";

function useAmount(): [number, (n: number) => void] {
  const [amount, setAmount] = useState(() => {
    try {
      const v = Number(localStorage.getItem(STORE));
      return v >= 10_000 && v <= 1e9 ? v : 1_000_000;
    } catch {
      return 1_000_000;
    }
  });
  const set = (n: number) => {
    setAmount(n);
    try {
      localStorage.setItem(STORE, String(n));
    } catch {
      /* private mode: the amount just won't be remembered */
    }
  };
  return [amount, set];
}

function AmountPicker({ amount, onChange }: { amount: number; onChange: (n: number) => void }) {
  const [text, setText] = useState(String(amount));
  const commit = () => {
    const n = Math.round(Number(text.replace(/[^\d]/g, "")));
    if (n >= 10_000 && n <= 1e9) onChange(n);
    else setText(String(amount));
  };
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-[14px] text-muted-foreground">
        Money at risk
        <span className="relative">
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-foreground">₹</span>
          <input
            inputMode="numeric"
            value={Number(text.replace(/[^\d]/g, "") || 0).toLocaleString("en-IN")}
            onChange={(e) => setText(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => e.key === "Enter" && commit()}
            className="num h-11 w-40 rounded-xl border border-line bg-panel/70 pl-7 pr-3 text-[15px] text-foreground backdrop-blur focus:border-saffron/60 focus:outline-none"
            aria-describedby="amount-help"
          />
        </span>
      </label>
      <div className="flex gap-1.5" role="group" aria-label="Quick amounts">
        {AMOUNTS.map((a) => (
          <button
            key={a}
            type="button"
            aria-pressed={a === amount}
            onClick={() => onChange(a)}
            className={cn("min-h-11 rounded-full border px-3.5 text-[13px] font-medium transition-colors", a === amount ? "border-saffron/60 bg-saffron/15 text-foreground" : "border-line text-muted-foreground hover:text-foreground")}
          >
            {a >= 1e7 ? `₹${a / 1e7} cr` : `₹${a / 1e5} lakh`}
          </button>
        ))}
      </div>
      <span id="amount-help" className="sr-only">
        Used to turn percentages into rupees on this page. Between ₹10,000 and ₹100 crore.
      </span>
    </div>
  );
}

function Heatmap({ names, m }: { names: string[]; m: number[][] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const n = names.length, lab = 78, cell = Math.max(Math.min((w - lab) / n, 40), 14);
  const [hover, setHover] = useState<[number, number] | null>(null);
  return (
    <div ref={ref}>
      <p className="mb-2 min-h-6 text-[13px] text-muted-foreground" aria-live="polite">
        {hover ? (
          <>
            {names[hover[0]]} with {names[hover[1]]}: <b className="num text-foreground">{m[hover[0]][hover[1]].toFixed(2)}</b>
          </>
        ) : (
          "Point at a square to read the pair."
        )}
      </p>
      {w > 0 && (
        <svg width={lab + cell * n} height={lab + cell * n} role="img" aria-label="Correlation between every pair of sectors over the last three months" onPointerLeave={() => setHover(null)} className="block">
          {names.map((nm, i) => (
            <g key={nm}>
              <text x={lab - 6} y={lab + i * cell + cell / 2 + 4} fontSize="11" fill="var(--muted-foreground)" textAnchor="end">
                {nm}
              </text>
              <text x={lab + i * cell + cell / 2} y={lab - 6} fontSize="11" fill="var(--muted-foreground)" transform={`rotate(-50 ${lab + i * cell + cell / 2} ${lab - 6})`}>
                {nm}
              </text>
            </g>
          ))}
          {m.map((row, i) =>
            row.map((v, j) => (
              <rect
                key={`${i}-${j}`}
                x={lab + j * cell + 1}
                y={lab + i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx="3"
                fill={v >= 0 ? `color-mix(in srgb, var(--accent-saffron) ${Math.round(Math.min(v, 1) * 85)}%, var(--panel-2))` : `color-mix(in srgb, var(--calm) ${Math.round(Math.min(-v, 1) * 120)}%, var(--panel-2))`}
                stroke={hover && hover[0] === i && hover[1] === j ? "var(--foreground)" : "none"}
                onPointerEnter={() => setHover([i, j])}
              />
            )),
          )}
        </svg>
      )}
    </div>
  );
}

const PERIODS = [
  { value: "1y", label: "1 year", years: 1 },
  { value: "3y", label: "3 years", years: 3 },
  { value: "5y", label: "5 years", years: 5 },
  { value: "all", label: "Since 2008", years: 99 },
] as const;
type Period = (typeof PERIODS)[number]["value"];

export function RiskPage() {
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<RiskMonitor>("risk");
  const [amount, setAmount] = useAmount();
  const [period, setPeriod] = useState<Period>("5y");
  const [tailSym, setTailSym] = useState("NIFTY_50");
  const [crash, setCrash] = useState(0);
  const [held, setHeld] = useState<string[]>(["NIFTY_50"]);
  useEffect(() => {
    document.title = "Risk monitor · QuantOS";
  }, []);
  const series = useMemo(() => {
    if (!data?.series.length) return [];
    const years = PERIODS.find((p) => p.value === period)!.years;
    const last = new Date(`${data.series.at(-1)!.time}T00:00:00Z`);
    last.setUTCFullYear(last.getUTCFullYear() - years);
    const cut = last.toISOString().slice(0, 10);
    return data.series.filter((p) => p.time >= cut);
  }, [data, period]);
  const replay = useMemo(() => {
    const c = data?.crashes[crash];
    if (!c) return null;
    const have = held.filter((h) => c.returns[h]);
    if (!have.length) return { c, have, values: [] as number[] };
    let v = amount;
    const values = c.dates.map((_, i) => {
      const r = have.reduce((s, h) => s + Math.expm1(c.returns[h][i] ?? 0), 0) / have.length;
      v *= 1 + r;
      return v;
    });
    return { c, have, values };
  }, [data, crash, held, amount]);

  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="The risk monitor could not be loaded">{error.message}. Reload the page to try again.</ErrorBox>;
  if (!data.as_of || !data.now || !data.corr_matrix || !data.turbulence_base) return <ErrorBox title="No risk data yet">The analytics job has not built the risk monitor yet.</ErrorBox>;

  const N = data.now, TB = data.turbulence_base;
  const t = data.tail[tailSym] ?? Object.values(data.tail)[0];
  const f95 = t.fhs["0.95"], f99 = t.fhs["0.99"], mc = t.mc;
  const corrWord = N.avg_corr_pct < 0.35 ? "less than usual" : N.avg_corr_pct > 0.65 ? "more than usual" : "about as usual";
  const turbHigh = N.turbulence_pct > 0.9;
  const holdable = Object.keys(data.names).filter((k) => replay?.c.returns[k]);
  const minV = replay?.values.length ? Math.min(...replay.values) : amount;
  const bottomI = replay?.values.indexOf(minV) ?? 0;

  return (
    <>
      <CineHero
        glow={turbHigh ? "rgb(238 94 90 / 0.24)" : undefined}
        background={!reduce && <Threads color={theme === "dark" ? "#EFA43A" : "#c97a0c"} accentColor="#4CC3C9" amplitude={1} distance={0.2} lineCount={26} thickness={1} speed={0.35} fray={0.4} parting={0.6} brightness={0.75} fade={0.4} opacity={0.65} enableMouseInteraction />}
        title={
          <>
            Sectors are moving together {corrWord}.{" "}
            <span className={turbHigh ? "text-down" : "text-saffron"}>{turbHigh ? "Today's moves are unusual." : "Today's moves look ordinary."}</span>
          </>
        }
        lede={`How fragile the market is right now and how bad a bad day can get, in percent and in rupees on your own amount. Data to the close of ${day(data.as_of, true)}. Measurements, not forecasts.`}
      >
        <div className="mb-10">
          <AmountPicker key={amount} amount={amount} onChange={setAmount} />
        </div>
        <StatRow>
          <BigStat label="Sectors moving together" value={N.avg_corr} decimals={2} sub={`${N.avg_corr_level}, ${N.avg_corr_ordinal}`} />
          <BigStat label="Market turbulence, percentile" text={N.turbulence_ordinal?.replace(" percentile", "") ?? "—"} tone={turbHigh ? "down" : undefined} sub={`${N.turbulent_days_21} turbulent day${N.turbulent_days_21 === 1 ? "" : "s"} in the last month`} />
          <BigStat label="1 day in 20, a fall of at least" text={pc(f95.var, 2)} tone="down" sub={`${inr(f95.var * amount)} on your amount, ${t.name}`} />
          <BigStat label="Gap between sectors, 1 month" text={pc(N.dispersion)} sub={N.dispersion_ordinal ?? undefined} />
        </StatRow>
      </CineHero>

      <CineSection
        id="together"
        title="Are sectors moving together?"
        lede="When everything moves as one, spreading money across sectors protects less and shocks spread faster."
        aside={<Segmented label="Period" options={PERIODS.map((p) => ({ value: p.value, label: p.label }))} value={period} onChange={setPeriod} />}
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <Spot>
            <LineChart
              ariaLabel="Average correlation between sectors and risk concentration over time"
              labels={series.map((p) => p.time)}
              height={320}
              xFormat={(x) => x.slice(0, 4)}
              yDomain={[0, 1]}
              series={[
                { id: "corr", data: series.map((p) => p.avg_corr), color: "var(--accent-saffron)", width: 1.8 },
                { id: "abs", data: series.map((p) => p.absorption), color: "var(--calm)", width: 1.4, dash: "4 3" },
              ]}
              yFormat={(v) => v.toFixed(1)}
              readout={(i) => (
                <span className="flex flex-wrap gap-x-4">
                  <span>{day(series[i]?.time, true)}</span>
                  <span>
                    <span className="mr-1.5 inline-block h-0.5 w-3 bg-saffron align-middle" aria-hidden />
                    Average correlation <b className="num text-foreground">{series[i]?.avg_corr?.toFixed(2)}</b>
                  </span>
                  <span>
                    <span className="mr-1.5 inline-block w-3 border-t-2 border-dashed border-calm align-middle" aria-hidden />
                    Top two drivers explain <b className="num text-foreground">{pc(series[i]?.absorption, 0)}</b> of risk
                  </span>
                </span>
              )}
            />
          </Spot>
          <Spot>
            <h3 className="text-[16px] font-semibold">Every pair of sectors, last 3 months</h3>
            <div className="mt-3 overflow-x-auto">
              <Heatmap names={data.corr_matrix.names} m={data.corr_matrix.matrix} />
            </div>
          </Spot>
        </div>
        <Reading>
          Sectors move together at an average correlation of <b>{N.avg_corr.toFixed(2)}</b>, the {N.avg_corr_ordinal} since 2008.{" "}
          {N.avg_corr_pct < 0.35
            ? "Low co-movement means stock and sector choice matters more than the overall market, and spreading money across sectors genuinely diversifies."
            : N.avg_corr_pct > 0.65
              ? "High co-movement means most sectors rise and fall together, so spreading money across sectors protects less than usual."
              : "Co-movement is around its usual level."}
        </Reading>
      </CineSection>

      <CineSection id="turbulence" title="Are today's moves unusual?" lede="Each day's moves across all sectors, compared with the past year. A high reading means sectors moved by unusual amounts or in unusual combinations.">
        <Spot>
          <LineChart
            ariaLabel="Market turbulence over time with the turbulent-day line"
            labels={series.map((p) => p.time)}
            height={300}
            xFormat={(x) => x.slice(0, 4)}
            yDomain={[0, Math.max(6, ...series.map((p) => Math.min(p.turbulence ?? 0, 14))) * 1.05]}
            series={[
              { id: "t", data: series.map((p) => (p.turbulence == null ? null : Math.min(p.turbulence, 14))), color: "var(--foreground)", width: 1 },
              { id: "p95", data: series.map((p) => p.p95), color: "var(--down)", dash: "4 3", width: 1.4 },
            ]}
            yFormat={(v) => v.toFixed(0)}
            readout={(i) => {
              const p = series[i];
              const hot = p && p.turbulence != null && p.p95 != null && p.turbulence > p.p95;
              return (
                <span className="flex flex-wrap gap-x-4">
                  <span>{day(p?.time, true)}</span>
                  <span>
                    Turbulence <b className="num text-foreground">{p?.turbulence?.toFixed(1)}</b> (1 is typical)
                  </span>
                  <span className={hot ? "text-down" : ""}>{hot ? "A turbulent day: top 5% of the past" : `turbulent above ${p?.p95?.toFixed(1)}`}</span>
                </span>
              );
            }}
          />
        </Spot>
        <Reading>
          After the {TB.n_turbulent} turbulent days since 2009, NIFTY's volatility over the next month averaged <b>{pc(TB.fwd_vol_turbulent)}</b> against <b>{pc(TB.fwd_vol_all)}</b> normally: rough patches tend to continue.
          Returns were not worse ({pc(TB.fwd_ret_turbulent, 1, true)} against {pc(TB.fwd_ret_all, 1, true)}), so turbulence warns about the size of moves, not their direction.
        </Reading>
      </CineSection>

      <CineSection id="sectors" title="Every sector at a glance" lede={`Returns and risk of each sector, best last month first. The gap between sectors over the last month is ${pc(N.dispersion)}, the ${N.dispersion_ordinal}.`}>
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[760px] text-[14px]">
            <thead className="bg-panel text-left text-[12.5px] text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5 font-normal">Sector</th>
                <th className="px-4 py-2.5 text-right font-normal">1 week</th>
                <th className="px-4 py-2.5 text-right font-normal">1 month</th>
                <th className="px-4 py-2.5 text-right font-normal">3 months</th>
                <th className="px-4 py-2.5 text-right font-normal">1 year</th>
                <th className="px-4 py-2.5 text-right font-normal">From high</th>
                <th className="px-4 py-2.5 font-normal">Where in its 52-week range</th>
                <th className="px-4 py-2.5 text-right font-normal">Volatility</th>
              </tr>
            </thead>
            <tbody className="num">
              {data.scoreboard.map((r) => (
                <tr key={r.symbol} className="border-t border-line">
                  <td className="px-4 py-2.5 font-medium">{r.name}</td>
                  {[r.w1, r.m1, r.m3, r.y1].map((v, i) => (
                    <td key={i} className={cn("px-4 py-2.5 text-right", tone(v))}>
                      {pc(v, 1, true)}
                    </td>
                  ))}
                  <td className="px-4 py-2.5 text-right text-muted-foreground">{(r.off_high ?? 0) < -0.001 ? pc(r.off_high, 1, true) : "At the high"}</td>
                  <td className="px-4 py-2.5">
                    <div className="relative h-1.5 w-28 rounded-full bg-foreground/[0.08]">
                      <div className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-saffron ring-2 ring-background" style={{ left: `${(r.range52 ?? 0) * 100}%` }} />
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-right">{pc(r.vol21, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CineSection>

      <CineSection
        id="bad-day"
        title="How bad can a bad day get?"
        lede="Loss levels for a single day at today's volatility, and the range of outcomes over the next month, in percent and on your amount."
        aside={<Segmented label="Index" options={Object.values(data.tail).map((x) => ({ value: x.symbol, label: x.name }))} value={tailSym} onChange={setTailSym} />}
      >
        <dl className="grid grid-cols-2 gap-6 lg:grid-cols-4">
          <BigStat label="1 day in 20, a fall of at least" text={pc(f95.var, 2)} tone="down" sub={inr(-f95.var * amount)} />
          <BigStat label="On those days, the average fall" text={pc(f95.cvar, 2)} tone="down" sub={inr(-f95.cvar * amount)} />
          <BigStat label="1 day in 100, a fall of at least" text={pc(f99.var, 2)} tone="down" sub={inr(-f99.var * amount)} />
          <BigStat label="Worst day on record" text={pc(t.worst_days[0]?.ret, 1, true)} tone="down" sub={`${day(t.worst_days[0]?.date, true)}, ${inr(t.worst_days[0]?.ret * amount)}`} />
        </dl>
        <div className="mt-8 grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <Spot>
            <h3 className="text-[16px] font-semibold">10,000 simulated months from today</h3>
            <p className="mt-1 text-[14px] text-muted-foreground">Change over the next 21 sessions, from a GARCH-t model of the last ten years.</p>
            <McBars mc={mc} />
          </Spot>
          <Spot>
            <table className="w-full text-[14.5px]">
              <caption className="sr-only">Range of outcomes over the next month</caption>
              <tbody className="num">
                {(
                  [
                    ["1 in 20 worse than", mc.p05],
                    ["1 in 4 worse than", mc.p25],
                    ["Middle outcome", mc.p50],
                    ["1 in 4 better than", mc.p75],
                    ["1 in 20 better than", mc.p95],
                  ] as [string, number][]
                ).map(([label, v]) => (
                  <tr key={label} className="border-b border-line last:border-0">
                    <th className="py-3 text-left font-normal text-muted-foreground">{label}</th>
                    <td className={cn("py-3 text-right font-semibold", tone(v))}>{pc(v, 1, true)}</td>
                    <td className={cn("py-3 pl-4 text-right", tone(v))}>{inr(v * amount)}</td>
                  </tr>
                ))}
                <tr>
                  <th className="pt-3 text-left font-normal text-muted-foreground">Chance of a 10% fall or worse</th>
                  <td className="pt-3 text-right font-semibold" colSpan={2}>
                    {pc(mc.down10, 1)}
                  </td>
                </tr>
              </tbody>
            </table>
          </Spot>
        </div>
        <Reading>
          At today's volatility {t.name} loses <b>{pc(f95.var, 1)}</b> or more on about one day in twenty, roughly once a month, and <b>{pc(f99.var, 1)}</b> or more on one day in a hundred. Tested on{" "}
          {t.backtest.n.toLocaleString("en-IN")} days since {t.backtest.start.slice(0, 4)} using only data available at the time, the one-in-twenty line was crossed on <b>{pc(t.backtest.rate95, 1)}</b> of days (the target is 5%).
          {t.hill_alpha ? ` A fall twice as large is about ${(2 ** t.hill_alpha).toFixed(0)} times rarer, far more common than a bell curve suggests.` : ""}
        </Reading>
      </CineSection>

      <CineSection id="replay" title="What past crashes would do to your money today" lede="Pick a crash and what you hold. The page applies that period's actual daily moves to your amount, from the market's peak to three months after its bottom.">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Crash">
          {data.crashes.map((c, i) => (
            <button
              key={c.peak}
              type="button"
              aria-pressed={i === crash}
              onClick={() => setCrash(i)}
              className={cn("min-h-11 rounded-full border px-4 text-[13.5px] font-medium transition-colors", i === crash ? "border-down/60 bg-down/15 text-foreground" : "border-line text-muted-foreground hover:text-foreground")}
            >
              {new Date(`${c.peak}T12:00:00Z`).toLocaleDateString("en-GB", { month: "short", year: "numeric" })} · NIFTY {pc(c.depth, 0)}
            </button>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-1.5" role="group" aria-label="What you hold, in equal parts">
          {holdable.map((k) => {
            const on = held.includes(k);
            return (
              <button
                key={k}
                type="button"
                aria-pressed={on}
                onClick={() => setHeld((h) => (on ? h.filter((x) => x !== k) : [...h, k]))}
                className={cn("min-h-10 rounded-full border px-3.5 text-[13px] transition-colors", on ? "border-saffron/60 bg-saffron/15 text-foreground" : "border-line text-muted-foreground hover:text-foreground")}
              >
                {data.names[k]}
              </button>
            );
          })}
        </div>
        {replay && replay.values.length > 0 ? (
          <>
            <Spot className="mt-6">
              <LineChart
                ariaLabel="Value of your amount through the chosen crash"
                labels={replay.c.dates}
                height={300}
                xFormat={(d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-GB", { month: "short", year: "2-digit" })}
                series={[
                  { id: "v", data: replay.values, color: "var(--down)", width: 2, area: "color-mix(in srgb, var(--down) 14%, transparent)" },
                  { id: "start", data: replay.values.map(() => amount), color: "var(--faint)", dash: "3 4", width: 1 },
                ]}
                yFormat={(v) => (v >= 1e7 ? `₹${(v / 1e7).toFixed(1)}cr` : `₹${(v / 1e5).toFixed(1)}L`)}
                readout={(i) => (
                  <span className="flex flex-wrap gap-x-4">
                    <span>{day(replay.c.dates[i], true)}</span>
                    <span>
                      Your money <b className="num text-foreground">{inr(replay.values[i])}</b>
                    </span>
                    <span className={tone(replay.values[i] - amount)}>{pc(replay.values[i] / amount - 1, 1, true)}</span>
                  </span>
                )}
              />
            </Spot>
            <dl className="mt-8 grid grid-cols-2 gap-6 lg:grid-cols-3">
              <BigStat label="Lowest point" text={inr(minV)} tone="down" sub={`${pc(minV / amount - 1, 1, true)}, on ${day(replay.c.dates[bottomI], true)}`} />
              <BigStat label="Paper loss at the bottom" text={inr(minV - amount)} tone="down" sub={`after ${bottomI + 1} trading days`} />
              <BigStat label="Three months after the bottom" text={inr(replay.values.at(-1)!)} sub={pc(replay.values.at(-1)! / amount - 1, 1, true)} />
            </dl>
          </>
        ) : (
          <p className="mt-6 text-muted-foreground">Pick at least one holding with data for this crash to see the replay.</p>
        )}
        <Method>
          <p>
            One-day loss levels use filtered historical simulation: past returns rescaled to today's GARCH-t volatility. Expected shortfall is the average loss beyond the one-in-twenty line. The Monte Carlo draws 10,000 paths of 21
            sessions from the same model.
          </p>
          <p>
            Sector co-movement is the average pairwise correlation of 11 sector indices over 63 sessions; risk concentration is the share of variance explained by the top two principal components over a year. Turbulence is the
            Mahalanobis distance of today's sector returns from the past year's. The replay rebalances holdings to equal weights daily, before costs.
          </p>
        </Method>
      </CineSection>
    </>
  );
}

function McBars({ mc }: { mc: RiskMonitor["tail"][string]["mc"] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const first = Math.max(0, mc.hist.findIndex((v) => v > 4) - 1);
  const last = Math.min(mc.hist.length - 1, mc.hist.length - 1 - [...mc.hist].reverse().findIndex((v) => v > 4) + 1);
  const bins = mc.hist.slice(first, last + 1);
  const max = Math.max(...bins), H = 220, bw = w / bins.length;
  const lo = mc.edges[first], hi = mc.edges[last + 1];
  const x = (v: number) => ((v - lo) / (hi - lo)) * w;
  return (
    <div ref={ref} className="mt-4">
      {w > 0 && (
        <svg width={w} height={H + 22} role="img" aria-label="Spread of 10,000 simulated one-month outcomes" className="block">
          {bins.map((c, i) => {
            const mid = (mc.edges[first + i] + mc.edges[first + i + 1]) / 2, h = (c / max) * H;
            return <rect key={i} x={i * bw + 0.5} y={H - h} width={Math.max(bw - 1.5, 1)} height={h} rx="2" fill={mid < mc.p05 ? "var(--down)" : mid < 0 ? "color-mix(in srgb, var(--down) 50%, transparent)" : "color-mix(in srgb, var(--up) 70%, transparent)"} />;
          })}
          <line x1="0" x2={w} y1={H + 0.5} y2={H + 0.5} stroke="var(--line-strong)" />
          {[
            [mc.p05, "1 in 20 worse"],
            [mc.p95, "1 in 20 better"],
          ].map(([v, l]) => (
            <g key={l as string}>
              <line x1={x(v as number)} x2={x(v as number)} y1="10" y2={H} stroke="var(--foreground)" strokeDasharray="3 3" />
              <text x={x(v as number)} y="8" fontSize="11" fill="var(--muted-foreground)" textAnchor="middle">
                {l as string}
              </text>
            </g>
          ))}
          <text x="0" y={H + 16} fontSize="11" fill="var(--faint)">
            {pc(lo, 0, true)}
          </text>
          <text x={w} y={H + 16} fontSize="11" fill="var(--faint)" textAnchor="end">
            {pc(hi, 0, true)}
          </text>
        </svg>
      )}
    </div>
  );
}
