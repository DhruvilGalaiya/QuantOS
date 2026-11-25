import type { Decay, Volatility } from "@quantos/types";
import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router";
import LineWaves from "@/components/bits/LineWaves";
import { BarChart, LineChart } from "@/components/charts/svg";
import { BigStat, nb, CineHero, CineSection, InstrumentTabs, Method, Reading, Segmented, Spot, StatRow, useReducedMotion, useWidth } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day, rupees, share } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const P = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const PTS = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)} pts`);
const short = (iso: string) => {
  const d = new Date(`${iso}T12:00:00Z`);
  return d.toLocaleDateString("en-GB", { month: "short", year: "2-digit" }).replace(" ", " ’");
};

// ------------------------------------------------------------------ the premium sellers collect
const PERIODS = [
  { value: "1y", label: "1 year", years: 1 },
  { value: "3y", label: "3 years", years: 3 },
  { value: "5y", label: "5 years", years: 5 },
  { value: "all", label: "Since 2011", years: 99 },
] as const;
type Period = (typeof PERIODS)[number]["value"];

function Premium({ data }: { data: Volatility }) {
  const v = data.vrp!;
  const [period, setPeriod] = useState<Period>("3y");
  const pts = useMemo(() => {
    const years = PERIODS.find((p) => p.value === period)!.years;
    const last = new Date(`${v.series.at(-1)!.time}T00:00:00Z`);
    const cut = new Date(last);
    cut.setUTCFullYear(last.getUTCFullYear() - years);
    const iso = cut.toISOString().slice(0, 10);
    return v.series.filter((p) => p.time >= iso);
  }, [v.series, period]);
  const stats = useMemo(() => {
    const gaps = pts.filter((p) => p.vix != null && p.rv != null).map((p) => ({ t: p.time, g: p.vix! - p.rv! }));
    if (!gaps.length) return null;
    const best = gaps.reduce((a, b) => (b.g > a.g ? b : a));
    const worst = gaps.reduce((a, b) => (b.g < a.g ? b : a));
    return { n: gaps.length, paid: gaps.filter((x) => x.g > 0).length / gaps.length, avg: gaps.reduce((s, x) => s + x.g, 0) / gaps.length, best, worst };
  }, [pts]);
  const labels = pts.map((p) => p.time);
  const yr = v.yearly.slice(-12);
  return (
    <CineSection
      id="premium"
      title="The premium option sellers collect"
      lede="India VIX is the yearly move NIFTY options are priced for. The second line is how much NIFTY actually moved over the following month. Green is where options were priced for more than happened, so sellers kept the premium; red is where the market moved more."
      aside={<Segmented label="Period" options={PERIODS.map((p) => ({ value: p.value, label: p.label }))} value={period} onChange={setPeriod} />}
    >
      <Spot>
        <LineChart
          ariaLabel="India VIX against the volatility that followed over the next month"
          labels={labels}
          height={340}
          series={[
            { id: "vix", data: pts.map((p) => (p.vix == null ? null : p.vix * 100)), color: "var(--accent-saffron)", width: 1.8 },
            { id: "rv", data: pts.map((p) => (p.rv == null ? null : p.rv * 100)), color: "var(--foreground)", width: 1.2 },
          ]}
          gap={{ a: 0, b: 1, pos: "color-mix(in srgb, var(--up) 30%, transparent)", neg: "color-mix(in srgb, var(--down) 42%, transparent)" }}
          yFormat={(t) => `${t}%`}
          xFormat={short}
          readout={(i) => {
            const p = pts[i];
            if (!p) return null;
            const g = p.vix != null && p.rv != null ? p.vix - p.rv : null;
            return (
              <span className="flex flex-wrap gap-x-4 gap-y-1">
                <span>{day(p.time, true)}</span>
                <span>
                  <span className="mr-1.5 inline-block h-0.5 w-3 bg-saffron align-middle" aria-hidden />
                  VIX <b className="num text-foreground">{P(p.vix)}</b>
                </span>
                <span>
                  <span className="mr-1.5 inline-block h-0.5 w-3 bg-foreground align-middle" aria-hidden />
                  Moved next month <b className="num text-foreground">{p.rv == null ? "not known yet" : P(p.rv)}</b>
                </span>
                {g != null && <span className={g >= 0 ? "text-up" : "text-down"}>{g >= 0 ? "Sellers kept" : "Buyers won"} {PTS(Math.abs(g))}</span>}
              </span>
            );
          }}
        />
      </Spot>
      {stats && (
        <dl className="mt-8 grid grid-cols-2 gap-6 lg:grid-cols-4">
          <BigStat label="Days sellers were paid" value={stats.paid * 100} suffix="%" sub={`of ${stats.n.toLocaleString("en-IN")} days shown`} tone="up" />
          <BigStat label="Average gap" text={PTS(stats.avg)} sub="VIX minus the move that followed" />
          <BigStat label="Biggest seller win" text={PTS(stats.best.g)} sub={day(stats.best.t, true)} />
          <BigStat label="Biggest seller loss" text={PTS(stats.worst.g, 0)} sub={day(stats.worst.t, true)} tone="down" />
        </dl>
      )}
      <Reading>
        Since {v.since?.slice(0, 4)}, India VIX was above the volatility that followed on <b>{share(v.share_implied_above)}</b> of days, by <b>{PTS(v.mean_spread).replace("+", "")}</b> on average. The worst miss
        began on <b>{day(v.worst_date, true)}</b> ({PTS(v.worst_spread, 0)}). Sellers earn the premium in small, steady amounts and give it back in rare, large ones.
      </Reading>
      <div className="mt-10 grid gap-4 lg:grid-cols-2">
        <Spot>
          <h3 className="text-[16px] font-semibold">The premium by VIX level</h3>
          <p className="mt-1 text-[14px] text-muted-foreground">Grouped by the VIX known on the day, so there is no hindsight in the buckets.</p>
          <table className="mt-4 w-full text-[14px]">
            <thead className="text-left text-[12.5px] text-faint">
              <tr>
                <th className="py-2 font-normal">VIX</th>
                <th className="py-2 text-right font-normal">Sellers paid</th>
                <th className="py-2 text-right font-normal">Average premium</th>
              </tr>
            </thead>
            <tbody className="num">
              {v.by_vix_level.map((b) => (
                <tr key={b.bucket} className="border-t border-line">
                  <td className="py-2.5">
                    {b.bucket} <span className="text-faint">({(b.vix_range[0] * 100).toFixed(0)}–{(b.vix_range[1] * 100).toFixed(0)})</span>
                  </td>
                  <td className="py-2.5 text-right">{share(b.share_implied_above)}</td>
                  <td className={cn("py-2.5 text-right font-medium", b.mean_spread >= 0 ? "text-up" : "text-down")}>{PTS(b.mean_spread)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Spot>
        <Spot>
          <h3 className="text-[16px] font-semibold">Average premium each year</h3>
          <p className="mt-1 text-[14px] text-muted-foreground">VIX minus the move that followed, in volatility points.</p>
          <div className="mt-4">
            <BarChart
              ariaLabel="Average volatility premium by year"
              labels={yr.map((y) => `’${String(y.year).slice(2)}`)}
              values={yr.map((y) => +(y.mean_spread * 100).toFixed(1))}
              color={(x) => (x >= 0 ? "var(--up)" : "var(--down)")}
              format={(x) => x.toFixed(1)}
              height={210}
            />
          </div>
        </Spot>
      </div>
      <Method>
        <p>Implied: the India VIX close on day t (30-calendar-day NIFTY implied volatility). Realized that followed: close-to-close volatility of sessions t+1 to t+21, annualised with √252, known only 21 sessions later.</p>
        <p>“Premium now” uses trailing realized volatility, so it is a proxy for the premium being charged, not what will be earned. VIX is not the same contract as an at-the-money straddle, and the gap overstates what a seller nets after costs and margin.</p>
      </Method>
    </CineSection>
  );
}

// ------------------------------------------------------------------ cone
function Cone({ data }: { data: Volatility }) {
  const c = data.cone;
  if (!c.length) return null;
  const c21 = c.find((x) => x.horizon === 21);
  const word = c21 ? (c21.percentile < 0.4 ? "calmer than usual" : c21.percentile > 0.6 ? "more volatile than usual" : "about normal") : "";
  return (
    <CineSection id="cone" title="Where today's volatility sits in history" lede="For each look-back window, the shaded bands show how volatile the index has been since 2007. The bright line is today. Above the bands means unusually wild; below means unusually quiet.">
      <Spot>
        <LineChart
          ariaLabel="Volatility cone: today's realized volatility against its historical range for each window"
          labels={c.map((x) => `${x.horizon}`)}
          xTicks={c.length}
          xFormat={(h) => `${h} sessions`}
          height={320}
          bands={[
            { lo: c.map((x) => x.p10 * 100), hi: c.map((x) => x.p90 * 100), color: "color-mix(in srgb, var(--accent-saffron) 10%, transparent)" },
            { lo: c.map((x) => x.p25 * 100), hi: c.map((x) => x.p75 * 100), color: "color-mix(in srgb, var(--accent-saffron) 18%, transparent)" },
          ]}
          series={[
            { id: "med", data: c.map((x) => x.p50 * 100), color: "var(--accent-saffron)", dash: "4 4", width: 1.4 },
            { id: "now", data: c.map((x) => x.current * 100), color: "var(--foreground)", width: 2.4 },
          ]}
          markers={[{ series: 1, color: "var(--foreground)" }]}
          yFormat={(t) => `${t}%`}
          yDomain={[0, Math.max(...c.map((x) => Math.max(x.p90, x.current))) * 100 * 1.2]}
          readout={(i) => {
            const x = c[i];
            return (
              <span className="flex flex-wrap gap-x-4">
                <span>Last {x.horizon} sessions</span>
                <span>
                  Today <b className="num text-foreground">{P(x.current)}</b>, the {Math.round(x.percentile * 100)}th percentile
                </span>
                <span>Typical {P(x.p50)}</span>
                <span>Record {P(x.max, 0)}</span>
              </span>
            );
          }}
        />
      </Spot>
      {c21 && (
        <Reading>
          Over the last 21 sessions {data.name} moved at <b>{P(c21.current)}</b> a year, the <b>{Math.round(c21.percentile * 100)}th percentile</b> of every 21-session stretch since 2007: {word}. The typical level is{" "}
          {P(c21.p50)}.
        </Reading>
      )}
    </CineSection>
  );
}

// ------------------------------------------------------------------ GARCH
function GarchSection({ data }: { data: Volatility }) {
  const g = data.garch;
  if (!g?.term_structure?.length) return null;
  const params = [
    ["Shock weight α", g.alpha?.toFixed(3), "How much yesterday's surprise moves today's volatility."],
    ["Memory β", g.beta?.toFixed(3), "How much of yesterday's volatility carries into today."],
    ["Half-life", g.half_life_days ? `${g.half_life_days.toFixed(0)} sessions` : "—", "How long a shock takes to lose half its effect."],
    ["Tail thickness ν", g.nu?.toFixed(1), "Lower means fatter tails. A normal bell curve would be infinite."],
  ];
  const h = g.history ?? [];
  return (
    <CineSection
      id="garch"
      title="How volatility shocks fade"
      lede="Volatility clusters: calm days follow calm days and storms follow storms. A GARCH(1,1) model with fat tails measures how strongly a shock feeds into tomorrow and how fast it fades back to normal."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {params.map(([label, value, plain]) => (
          <Spot key={label}>
            <p className="text-[13.5px] text-muted-foreground">{label}</p>
            <p className="mt-1.5 font-headline text-[30px] font-semibold leading-none">{value}</p>
            <p className="mt-3 text-[14px] leading-snug text-muted-foreground">{plain}</p>
          </Spot>
        ))}
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Spot>
          <h3 className="text-[16px] font-semibold">Expected volatility by horizon</h3>
          <LineChart
            className="mt-3"
            ariaLabel="GARCH expected average volatility by horizon"
            labels={g.term_structure.map((t) => `${t.days}`)}
            xTicks={g.term_structure.length}
            xFormat={(d) => `${d}d`}
            height={240}
            series={[
              { id: "ts", data: g.term_structure.map((t) => t.vol * 100), color: "var(--accent-saffron)", width: 2.2 },
              { id: "lr", data: g.term_structure.map(() => (g.long_run_vol ?? 0) * 100), color: "var(--faint)", dash: "3 4", width: 1.2 },
            ]}
            markers={[{ series: 0, color: "var(--accent-saffron)" }]}
            yFormat={(t) => `${t}%`}
            readout={(i) => (
              <span>
                Next {g.term_structure[i].days} sessions: <b className="num text-foreground">{P(g.term_structure[i].vol)}</b> a year, long-run {P(g.long_run_vol)}
              </span>
            )}
          />
        </Spot>
        <Spot>
          <h3 className="text-[16px] font-semibold">Model volatility against what happened, 5 years</h3>
          <LineChart
            className="mt-3"
            ariaLabel="GARCH conditional volatility against 21-session realized volatility over five years"
            labels={h.map((p) => p.time)}
            height={240}
            xFormat={short}
            series={[
              { id: "rv", data: h.map((p) => (p.realized == null ? null : p.realized * 100)), color: "var(--faint)", width: 1.1 },
              { id: "g", data: h.map((p) => (p.garch == null ? null : p.garch * 100)), color: "var(--accent-saffron)", width: 1.6 },
            ]}
            yFormat={(t) => `${t}%`}
            readout={(i) => (
              <span className="flex flex-wrap gap-x-4">
                <span>{day(h[i]?.time, true)}</span>
                <span>
                  GARCH <b className="num text-foreground">{P(h[i]?.garch)}</b>
                </span>
                <span>Realized, 21 sessions {P(h[i]?.realized)}</span>
              </span>
            )}
          />
        </Spot>
      </div>
      <Reading>
        A shock loses half its effect in about <b>{g.half_life_days?.toFixed(0)} sessions</b>. Tails are fat (ν ≈ {g.nu?.toFixed(1)}): big days come far more often than a bell curve allows. Today's model volatility of{" "}
        <b>{P(g.current_vol)}</b> is {(g.current_vol ?? 0) > (g.long_run_vol ?? 0) ? "above" : "below"} its long-run {P(g.long_run_vol)}, so the model expects it to drift back. GARCH sizes moves; it never says which way.
      </Reading>
      <Method>
        <p>
          r = μ + ε, ε = σ·z with z Student-t(ν), and σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1). Fitted by maximum likelihood on daily log returns from {g.fit_start} ({g.n_obs?.toLocaleString("en-IN")} sessions) with the
          arch package. The horizon curve averages the forecast daily variances over each horizon.
        </p>
      </Method>
    </CineSection>
  );
}

// ------------------------------------------------------------------ overnight straddle
const VERDICT: Record<string, [string, string]> = {
  steady: ["Steady small edge", "text-up"],
  fragile: ["Wins often, one bad night wipes out many", "text-normal"],
  loss: ["Sellers lost on average", "text-down"],
};
const nightWord = (t: number) => (t === 1 ? "1 day" : `${t} days`);

function Histogram({ counts, edges }: { counts: number[]; edges: number[] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const H = 180, max = Math.max(...counts, 1), bw = w / counts.length;
  return (
    <div ref={ref}>
      {w > 0 && (
        <svg width={w} height={H + 22} role="img" aria-label="How many nights ended at each result per ₹100 of premium collected" className="block">
          {counts.map((c, i) => {
            const h = (c / max) * H;
            const mid = (edges[i] + edges[i + 1]) / 2;
            return <rect key={i} x={i * bw + 0.5} y={H - h} width={Math.max(bw - 1.5, 1)} height={h} rx="2" fill={mid >= 0 ? "var(--up)" : "var(--down)"} opacity={c ? 0.85 : 0.15} />;
          })}
          <line x1="0" x2={w} y1={H + 0.5} y2={H + 0.5} stroke="var(--line-strong)" />
          {[-300, -200, -100, 0, 100].map((t) => {
            const i = edges.indexOf(t);
            return i < 0 ? null : (
              <text key={t} x={i * bw} y={H + 16} fontSize="11" fill="var(--faint)" textAnchor={t === -300 ? "start" : "middle"}>
                {t === -300 ? "≤ −₹300" : `${t > 0 ? "+" : t < 0 ? "−" : ""}₹${Math.abs(t)}`}
              </text>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function Straddle({ decay }: { decay: Decay }) {
  const [pick, setPick] = useState(String(decay.rows[0]?.tdte ?? 1));
  const sel = decay.rows.find((r) => String(r.tdte) === pick) ?? decay.rows[0];
  const best = decay.rows.reduce((a, b) => (b.avg_rupees > a.avg_rupees ? b : a));
  const lost = best.avg_rupees > 0 ? Math.abs(best.worst_rupees) / best.avg_rupees : null;
  return (
    <CineSection
      id="straddle"
      title="What an overnight option seller kept"
      lede="The test: at 3:15 pm sell the call and the put at the strike nearest the index, and buy both back at 3:15 pm the next trading day. The seller keeps the premium if the index stays put and loses if it moves a lot."
    >
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[720px] text-[14px]">
          <thead className="bg-panel text-left text-[12.5px] text-faint">
            <tr>
              <th className="px-4 py-2.5 font-normal">Days left to expiry</th>
              <th className="px-4 py-2.5 text-right font-normal">Nights</th>
              <th className="px-4 py-2.5 text-right font-normal">Won</th>
              <th className="px-4 py-2.5 text-right font-normal">Average, 1 lot</th>
              <th className="px-4 py-2.5 text-right font-normal">Typical per ₹100</th>
              <th className="px-4 py-2.5 text-right font-normal">Worst, 1 lot</th>
              <th className="px-4 py-2.5 font-normal">What history says</th>
            </tr>
          </thead>
          <tbody className="num">
            {decay.rows.map((r) => (
              <tr key={r.tdte} className="border-t border-line">
                <td className="px-4 py-2.5 font-medium">{nightWord(r.tdte)}{r.tdte === 1 ? " (day before expiry)" : ""}</td>
                <td className="px-4 py-2.5 text-right">{r.nights}</td>
                <td className="px-4 py-2.5 text-right">{share(r.win_rate)}</td>
                <td className={cn("px-4 py-2.5 text-right font-medium", r.avg_rupees >= 0 ? "text-up" : "text-down")}>{r.avg_rupees >= 0 ? "+" : "−"}{rupees(Math.abs(r.avg_rupees))}</td>
                <td className="px-4 py-2.5 text-right">{r.median_per100 >= 0 ? "+" : "−"}₹{Math.abs(r.median_per100).toFixed(0)}</td>
                <td className="px-4 py-2.5 text-right text-down">−{rupees(Math.abs(r.worst_rupees))}</td>
                <td className={cn("px-4 py-2.5", VERDICT[r.verdict][1])}>{VERDICT[r.verdict][0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[13.5px] text-muted-foreground">
        Rupees for 1 lot at today's lot size ({decay.lot} units), before brokerage, taxes and slippage. “Typical per ₹100” is what the median night kept for every ₹100 of premium collected.
      </p>
      <div className="mt-8 grid gap-4 lg:grid-cols-2">
        <Spot>
          <h3 className="text-[16px] font-semibold">How often the seller made money</h3>
          <div className="mt-4">
            <BarChart ariaLabel="Share of profitable nights by days to expiry" labels={decay.rows.map((r) => nightWord(r.tdte))} values={decay.rows.map((r) => Math.round(r.win_rate * 100))} format={(x) => `${x}%`} baseline={50} baselineLabel="coin flip" />
          </div>
        </Spot>
        <Spot>
          <h3 className="text-[16px] font-semibold">Average result per night, 1 lot</h3>
          <div className="mt-4">
            <BarChart
              ariaLabel="Average rupee result per night by days to expiry"
              labels={decay.rows.map((r) => nightWord(r.tdte))}
              values={decay.rows.map((r) => Math.round(r.avg_rupees))}
              color={(x) => (x >= 0 ? "var(--up)" : "var(--down)")}
              format={(x) => `${x >= 0 ? "+" : "−"}₹${Math.abs(x).toLocaleString("en-IN")}`}
            />
          </div>
        </Spot>
      </div>
      <Spot className="mt-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-[16px] font-semibold">Every night, with {nightWord(sel.tdte)} left</h3>
          <Segmented label="Days left to expiry" options={decay.rows.map((r) => ({ value: String(r.tdte), label: nightWord(r.tdte) }))} value={pick} onChange={setPick} />
        </div>
        <p className="mt-1 text-[14px] text-muted-foreground">Profit or loss per ₹100 of premium collected. Green nights kept money; red nights paid out.</p>
        <div className="mt-5">
          <Histogram counts={sel.hist} edges={decay.edges} />
        </div>
      </Spot>
      <Reading title="What this means for a seller">
        Selling with <b>{nightWord(best.tdte)}</b> left worked best: profitable on <b>{share(best.win_rate)}</b> of {best.nights} nights and <b>{rupees(best.avg_rupees)}</b> per lot on average. But the worst night lost{" "}
        <b>{rupees(Math.abs(best.worst_rupees))}</b>
        {lost ? `, about ${lost.toFixed(0)} average nights of profit at once` : ""}. Size for the worst night, not the average one.
      </Reading>
      <Method>
        <p>
          Entry and exit are the 3:15 pm bar open of the call and put at the strike nearest the index on the entry day, held for one session. Per ₹100 = (premium sold − premium bought back) ÷ premium sold × 100. Sample:{" "}
          {decay.n} nights from {day(decay.start, true)} to {day(decay.end, true)}. Weekly NIFTY gives about 110 nights per row and monthly BANK NIFTY about 28, so BANK rows are far less certain.
        </p>
      </Method>
    </CineSection>
  );
}

// ------------------------------------------------------------------ rough volatility
function Rough({ data }: { data: Volatility }) {
  const h = data.hurst;
  if (!h?.H) return null;
  const qs = Object.keys(h.zeta).map(Number).sort((a, b) => a - b);
  return (
    <CineSection
      id="rough"
      title="Is volatility rough?"
      lede="Research from 2018 found that volatility jitters much more sharply than classic models assume, with a Hurst exponent near 0.1. Here is the same test on Indian data."
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Spot>
          <p className="text-[13.5px] text-muted-foreground">Hurst exponent of log-volatility</p>
          <p className="mt-2 font-headline text-[64px] font-semibold leading-none text-saffron">{h.H.toFixed(2)}</p>
          <p className="mt-4 text-[15px] leading-relaxed text-muted-foreground">
            A random walk scores 0.5. {data.name} scores far lower, so its volatility is rough: it lurches over days. Models that assume smooth volatility understate short-dated option risk.
          </p>
        </Spot>
        <Spot>
          <h3 className="text-[16px] font-semibold">Hurst exponent in rolling three-year windows</h3>
          <LineChart
            className="mt-3"
            ariaLabel="Rolling three-year Hurst exponent"
            labels={h.rolling.map((r) => r.end)}
            height={220}
            xFormat={short}
            yDomain={[0, 0.6]}
            series={[
              { id: "h", data: h.rolling.map((r) => r.H), color: "var(--accent-saffron)", width: 2 },
              { id: "b", data: h.rolling.map(() => 0.5), color: "var(--faint)", dash: "3 4", width: 1.2 },
            ]}
            yFormat={(t) => t.toFixed(1)}
            readout={(i) => (
              <span>
                Window ending {day(h.rolling[i]?.end, true)}: <b className="num text-foreground">H = {h.rolling[i]?.H.toFixed(2)}</b> (a random walk is 0.5)
              </span>
            )}
          />
        </Spot>
      </div>
      <Method>
        <p>
          Volatility proxy: {h.proxy}. For each lag Δ we measure the average |log σ(t+Δ) − log σ(t)|^q, fit its slope ζ(q) against log Δ, and H is the slope of ζ(q) against q (measured at q ={" "}
          {qs.join(", ")}). Daily ranges are noisier than the 5-minute data of the original study, so read the level as indicative and the comparison over time as the reliable part.
        </p>
      </Method>
    </CineSection>
  );
}

// ------------------------------------------------------------------ page
export function VolatilityPage() {
  const { symbol = "" } = useParams();
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<Volatility>(symbol ? `volatility/${symbol}` : null);
  useEffect(() => {
    if (data) document.title = `${data.name} volatility and the seller's edge · QuantOS`;
  }, [data]);

  if (!symbol) return <Navigate to="/volatility/NIFTY_50" replace />;
  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="This volatility page could not be loaded">{error.message}. Try another index or reload the page.</ErrorBox>;
  if (!data.as_of) return <ErrorBox title="No volatility study yet">The analytics job has not built this index yet. Run python -m quantos_platform.jobs analytics.</ErrorBox>;

  const v = data.vrp, g = data.garch, rv = data.realized;
  const title = v?.premium_now != null ? (
    <>
      Option buyers are paying <span className="text-saffron">{Math.abs(v.premium_now * 100).toFixed(1)} points</span> {v.premium_now >= 0 ? "more" : "less"} than {nb(data.name)} is moving
    </>
  ) : data.implied && g?.vol_21d ? (
    <>
      {nb(data.name)} options price <span className="text-saffron">{P(data.implied.atm_iv)}</span> a year against GARCH's {P(g.vol_21d)}
    </>
  ) : (
    <>
      {nb(data.name)} is moving at <span className="text-saffron">{P(rv?.yang_zhang)}</span> a year
    </>
  );

  return (
    <>
      <CineHero
        background={
          !reduce && (
            <LineWaves color1="#EFA43A" color2="#F2683C" color3="#4CC3C9" speed={0.25} lineCount={22} lineWidth={1} glow={0.5} brightness={0.75} warpIntensity={0.6} rotation={-12} edgeFade={0.5} lightMode={theme === "light"} opacity={0.75} />
          )
        }
        top={<InstrumentTabs label="Choose an index" items={data.indices} active={data.symbol} hrefFor={(s) => `/volatility/${s}`} />}
        title={title}
        lede="How much the market is moving, how much option prices say it will move, and who has historically been paid for the difference. Measurements and history, not forecasts of direction."
      >
        <StatRow>
          <BigStat label="Realized volatility, 21 sessions" value={rv?.yang_zhang != null ? rv.yang_zhang * 100 : null} decimals={1} suffix="%" sub={rv?.ordinal ? `${rv.ordinal} since 2007` : undefined} />
          <BigStat label="GARCH expected, next 21 sessions" value={g?.vol_21d != null ? g.vol_21d * 100 : null} decimals={1} suffix="%" sub={`long-run level ${P(g?.long_run_vol)}`} />
          {v ? (
            <BigStat label="India VIX" value={v.vix_now != null ? v.vix_now * 100 : null} decimals={1} sub={`${Math.round((v.vix_percentile_1y ?? 0) * 100)}th percentile of the past year`} />
          ) : (
            <BigStat label="Front at-the-money implied" value={data.implied ? data.implied.atm_iv * 100 : null} decimals={1} suffix="%" sub={data.implied?.expiry ? `expiry ${day(data.implied.expiry)}` : "no option snapshot"} />
          )}
          {v ? (
            <BigStat label="Premium now" text={PTS(v.premium_now)} tone={(v.premium_now ?? 0) >= 0 ? "up" : "down"} sub={v.premium_ordinal ? `VIX minus realized, ${v.premium_ordinal}` : undefined} />
          ) : (
            <BigStat label="Implied minus GARCH" text={data.implied && g?.vol_21d ? PTS(data.implied.atm_iv - g.vol_21d) : "—"} sub="what options charge over the model" />
          )}
        </StatRow>
      </CineHero>

      {v ? (
        <Premium data={data} />
      ) : (
        <p className="mt-16 max-w-[70ch] text-[15px] text-muted-foreground">A long implied-volatility history (India VIX) exists only for NIFTY 50, so the full premium study is on the NIFTY 50 page.</p>
      )}
      <Cone data={data} />
      <GarchSection data={data} />
      {data.decay ? <Straddle decay={data.decay} /> : null}
      <Rough data={data} />
    </>
  );
}
