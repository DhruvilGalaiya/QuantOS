import type { Quote, Surface } from "@quantos/types";
import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router";
import { LineChart, niceTicks } from "@/components/charts/svg";
import { Surface3D } from "@/components/charts/Surface3D";
import { BigStat, CineHero, CineSection, InstrumentTabs, Method, nb, Reading, Segmented, Spot, StatRow, useWidth } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useApi } from "@/lib/api";
import { day, level, stamp } from "@/lib/format";
import { cn } from "@/lib/utils";

const P = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const PTS = (v: number | null | undefined, d = 2) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)} pts`);
const mny = (k: number) => `${k >= 0 ? "+" : "−"}${Math.abs((Math.exp(k) - 1) * 100).toFixed(1)}%`;

/** One expiry's smile: market quotes with their bid–ask range, against the SVI and SABR fits. */
function SmileChart({ quotes }: { quotes: Quote[] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<Quote | null>(null);
  const q = useMemo(() => [...quotes].filter((x) => x.k != null).sort((a, b) => a.k! - b.k!), [quotes]);
  if (!q.length) return null;
  const H = 340, padL = 8, padR = 52, padT = 12, padB = 30;
  const iw = Math.max(w - padL - padR, 10), ih = H - padT - padB;
  const ivs = q.flatMap((x) => [x.iv, x.iv_bid, x.iv_ask, x.svi, x.sabr]).filter((v): v is number => v != null);
  const lo = Math.min(...ivs) * 0.97, hi = Math.max(...ivs) * 1.03;
  const k0 = q[0].k!, k1 = q[q.length - 1].k!;
  const x = (k: number) => padL + ((k - k0) / (k1 - k0 || 1)) * iw;
  const y = (v: number) => padT + ih - ((v - lo) / (hi - lo)) * ih;
  const path = (key: "svi" | "sabr") =>
    q.filter((p) => p[key] != null).map((p, i) => `${i ? "L" : "M"}${x(p.k!).toFixed(1)},${y(p[key]!).toFixed(1)}`).join("");
  const xt = niceTicks(k0, k1, 5), yt = niceTicks(lo, hi, 4);
  const shown = hover ?? q.reduce((a, b) => (Math.abs(b.k!) < Math.abs(a.k!) ? b : a));
  return (
    <div ref={ref}>
      <p className="mb-2 min-h-11 text-[13px] text-muted-foreground" aria-live="polite">
        Strike <b className="num text-foreground">{level(shown.strike, 0)}</b> ({mny(shown.k!)} vs forward, {shown.type === "CE" ? "call" : "put"}): market{" "}
        <b className="num text-foreground">{P(shown.iv)}</b>
        {shown.iv_bid != null && shown.iv_ask != null && ` (bid ${P(shown.iv_bid)}, ask ${P(shown.iv_ask)})`}, SVI {P(shown.svi)}, SABR {P(shown.sabr)}
      </p>
      {w > 0 && (
        <svg width={w} height={H} role="img" aria-label="Implied volatility smile for the chosen expiry" onPointerLeave={() => setHover(null)} className="block">
          {yt.map((t) => (
            <g key={t}>
              <line x1={padL} x2={padL + iw} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray="2 4" />
              <text x={padL + iw + 8} y={y(t) + 4} fontSize="11" fill="var(--faint)">
                {(t * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          {xt.map((t) => (
            <text key={t} x={x(t)} y={H - 8} fontSize="11" fill="var(--faint)" textAnchor="middle">
              {mny(t)}
            </text>
          ))}
          {k0 < 0 && k1 > 0 && <line x1={x(0)} x2={x(0)} y1={padT} y2={padT + ih} stroke="var(--line-strong)" strokeDasharray="3 3" />}
          {q.map((p) =>
            p.iv_bid != null && p.iv_ask != null ? (
              <line key={`w${p.strike}${p.type}`} x1={x(p.k!)} x2={x(p.k!)} y1={y(p.iv_bid)} y2={y(p.iv_ask)} stroke="var(--muted-foreground)" strokeOpacity="0.45" />
            ) : null,
          )}
          <path d={path("sabr")} fill="none" stroke="var(--calm)" strokeWidth="1.6" strokeDasharray="5 4" />
          <path d={path("svi")} fill="none" stroke="var(--accent-saffron)" strokeWidth="2.2" />
          {q.map((p) => (
            <circle
              key={`${p.strike}${p.type}`}
              cx={x(p.k!)}
              cy={y(p.iv)}
              r={hover === p ? 5 : 3}
              fill={p.used ? "var(--foreground)" : "transparent"}
              stroke="var(--foreground)"
              strokeOpacity={p.used ? 1 : 0.5}
              onPointerEnter={() => setHover(p)}
            />
          ))}
        </svg>
      )}
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-muted-foreground" aria-label="Legend">
        <li>
          <span className="mr-1.5 inline-block size-2 rounded-full bg-foreground align-middle" aria-hidden />
          Market quote (hollow: not used in the fit)
        </li>
        <li>
          <span className="mr-1.5 inline-block h-0.5 w-4 bg-saffron align-middle" aria-hidden />
          SVI fit
        </li>
        <li>
          <span className="mr-1.5 inline-block w-4 border-t-2 border-dashed border-calm align-middle" aria-hidden />
          SABR fit
        </li>
      </ul>
    </div>
  );
}

export function SurfacePage() {
  const { symbol = "" } = useParams();
  const { data, error, isPending } = useApi<Surface>(symbol ? `surface/${symbol}` : null);
  const [model, setModel] = useState<"svi" | "sabr">("svi");
  // the picked expiry belongs to one index; switching index falls back to its reference expiry
  const [pick, setPick] = useState<{ symbol: string; expiry: string } | null>(null);
  const expiry = pick?.symbol === symbol ? pick.expiry : null;
  const setExpiry = (e: string) => setPick({ symbol, expiry: e });
  useEffect(() => {
    if (data) document.title = `${data.name} volatility surface · QuantOS`;
  }, [data]);

  if (!symbol) return <Navigate to="/surface/NIFTY_50" replace />;
  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="This volatility surface could not be loaded">{error.message}. Try another index or reload the page.</ErrorBox>;
  if (!data.snapshot_ts || !data.grid || !data.ref)
    return <ErrorBox title="No surface yet">The option snapshot for {data.name} has not been fitted yet. It appears after the next vol job run.</ErrorBox>;

  const ref = data.ref;
  const e0 = expiry ?? ref.expiry;
  const quotes = data.points.filter((p) => p.expiry === e0);
  const ex = data.expiries;
  const skewWord = (ref.rr25 ?? 0) < -0.002 ? "puts cost more than calls" : (ref.rr25 ?? 0) > 0.002 ? "calls cost more than puts" : "puts and calls cost about the same";
  const ivRv = ref.atm_iv != null && data.realized_20d != null ? ref.atm_iv - data.realized_20d : null;
  const slope = ex.length > 1 && ex[0].atm_iv != null && ex.at(-1)!.atm_iv != null ? ex.at(-1)!.atm_iv! - ex[0].atm_iv! : null;

  return (
    <>
      <CineHero
        background={
          <div className="absolute inset-y-0 right-0 hidden w-[62%] opacity-60 md:block">
            <Surface3D ambient grid={{ k: data.grid.k, days: data.grid.days, iv: data.grid.svi }} height={560} label="" />
          </div>
        }
        top={<InstrumentTabs label="Choose an index" items={data.symbols} active={data.symbol} hrefFor={(s) => `/surface/${s}`} />}
        title={
          <>
            {nb(data.name)} options price <span className="text-saffron">{P(ref.atm_iv)}</span> a year at the money
          </>
        }
        lede={`At the ${day(ref.expiry)} expiry ${skewWord}. Implied volatility across ${data.quotes} option quotes and ${ex.length} expiries, fitted with SVI and SABR. Option prices as of ${stamp(data.snapshot_ts)}.`}
      >
        <StatRow>
          <BigStat label={`Spot, forward ${level(ref.forward, 0)}`} value={data.spot} decimals={2} sub={`forward for ${day(ref.expiry)}`} />
          <BigStat label={`At-the-money, ${day(ref.expiry)}`} value={ref.atm_iv != null ? ref.atm_iv * 100 : null} decimals={1} suffix="%" tone="saffron" sub="read off the fitted smile" />
          <BigStat label="Realized, last 20 sessions" value={data.realized_20d != null ? data.realized_20d * 100 : null} decimals={1} suffix="%" sub={`implied minus realized ${PTS(ivRv, 1)}`} />
          <BigStat label="No-arbitrage checks" text={data.arb_warnings === 0 ? "All pass" : `${data.arb_warnings} warning${data.arb_warnings > 1 ? "s" : ""}`} tone={data.arb_warnings === 0 ? "up" : undefined} sub="butterfly and calendar" />
        </StatRow>
      </CineHero>

      <CineSection
        id="surface"
        title="The whole surface, strikes by expiries"
        lede="Height and colour are implied volatility. Left to right runs from strikes below the forward to strikes above it; front to back runs from the nearest expiry to the furthest. Drag to turn it."
        aside={<Segmented label="Fit" options={[{ value: "svi", label: "SVI" }, { value: "sabr", label: "SABR" }]} value={model} onChange={setModel} />}
      >
        <Spot className="!p-2 sm:!p-4">
          <Surface3D grid={{ k: data.grid.k, days: data.grid.days, iv: model === "svi" ? data.grid.svi : data.grid.sabr }} label={`${data.name} implied volatility surface, ${model.toUpperCase()} fit`} />
        </Spot>
        <Reading>
          At the {day(ref.expiry)} expiry the 25-delta risk reversal is <b>{PTS(ref.rr25)}</b>, so {skewWord}; the butterfly is <b>{PTS(ref.bf25)}</b>, the extra charged for big moves either way.
          {slope != null && (
            <>
              {" "}
              From the nearest to the furthest expiry, at-the-money volatility {slope >= 0 ? "rises" : "falls"} by <b>{PTS(Math.abs(slope), 1).replace("+", "")}</b>
              {slope < 0 ? ": traders expect more movement soon than later." : ": traders expect more movement further out."}
            </>
          )}
        </Reading>
      </CineSection>

      <CineSection
        id="smile"
        title="The smile at each expiry"
        lede="Each dot is a market quote with its bid–ask range. The lines are the two models fitted through them. A close fit means the surface above can be trusted at that expiry."
        aside={<Segmented label="Expiry" options={ex.map((e) => ({ value: e.expiry, label: `${day(e.expiry).slice(4)} · ${e.days.toFixed(0)}d` }))} value={e0} onChange={setExpiry} />}
      >
        <Spot>
          <SmileChart quotes={quotes} />
        </Spot>
      </CineSection>

      <CineSection id="term" title="How volatility changes with time to expiry" lede="At-the-money volatility shows when the market expects movement. Skew and curvature show how lopsided and how fat-tailed it expects that movement to be.">
        <div className="grid gap-4 lg:grid-cols-2">
          <Spot>
            <h3 className="text-[16px] font-semibold">At-the-money volatility</h3>
            <LineChart
              className="mt-3"
              ariaLabel="At-the-money implied volatility by expiry"
              labels={ex.map((e) => e.expiry)}
              xTicks={ex.length}
              xFormat={(e) => day(e).slice(4)}
              height={240}
              series={[
                { id: "atm", data: ex.map((e) => (e.atm_iv == null ? null : e.atm_iv * 100)), color: "var(--accent-saffron)", width: 2.2 },
                { id: "rv", data: ex.map(() => (data.realized_20d ?? 0) * 100), color: "var(--faint)", dash: "3 4", width: 1.2 },
              ]}
              markers={[{ series: 0, color: "var(--accent-saffron)" }]}
              yFormat={(t) => `${t}%`}
              readout={(i) => (
                <span>
                  {day(ex[i].expiry)}, {ex[i].days.toFixed(0)} days: <b className="num text-foreground">{P(ex[i].atm_iv)}</b> (dashed: realized {P(data.realized_20d)})
                </span>
              )}
            />
          </Spot>
          <Spot>
            <h3 className="text-[16px] font-semibold">Skew and curvature</h3>
            <LineChart
              className="mt-3"
              ariaLabel="25-delta risk reversal and butterfly by expiry"
              labels={ex.map((e) => e.expiry)}
              xTicks={ex.length}
              xFormat={(e) => day(e).slice(4)}
              height={240}
              zero
              series={[
                { id: "rr", data: ex.map((e) => (e.rr25 == null ? null : e.rr25 * 100)), color: "var(--down)", width: 2 },
                { id: "bf", data: ex.map((e) => (e.bf25 == null ? null : e.bf25 * 100)), color: "var(--calm)", width: 2 },
              ]}
              markers={[
                { series: 0, color: "var(--down)" },
                { series: 1, color: "var(--calm)" },
              ]}
              yFormat={(t) => t.toFixed(1)}
              readout={(i) => (
                <span className="flex flex-wrap gap-x-4">
                  <span>{day(ex[i].expiry)}</span>
                  <span>
                    <span className="mr-1 inline-block size-2 rounded-full bg-down align-middle" aria-hidden />
                    Risk reversal <b className="num text-foreground">{PTS(ex[i].rr25)}</b>
                  </span>
                  <span>
                    <span className="mr-1 inline-block size-2 rounded-full bg-calm align-middle" aria-hidden />
                    Butterfly <b className="num text-foreground">{PTS(ex[i].bf25)}</b>
                  </span>
                </span>
              )}
            />
          </Spot>
        </div>
      </CineSection>

      <CineSection id="calibration" title="How well each expiry is fitted" lede="Fit error is the typical gap between the model and the market, in volatility points. The two checks confirm the surface allows no free-money trades.">
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[760px] text-[14px]">
            <thead className="bg-panel text-left text-[12.5px] text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5 font-normal">Expiry</th>
                <th className="px-4 py-2.5 text-right font-normal">Quotes</th>
                <th className="px-4 py-2.5 text-right font-normal">At the money</th>
                <th className="px-4 py-2.5 text-right font-normal">SVI error</th>
                <th className="px-4 py-2.5 text-right font-normal">SABR error</th>
                <th className="px-4 py-2.5 text-right font-normal">SABR ρ / ν</th>
                <th className="px-4 py-2.5 font-normal">Butterfly</th>
                <th className="px-4 py-2.5 font-normal">Calendar</th>
              </tr>
            </thead>
            <tbody className="num">
              {ex.map((e) => (
                <tr key={e.expiry} className={cn("border-t border-line", e.expiry === e0 && "bg-saffron/[0.06]")}>
                  <td className="px-4 py-2.5 font-medium">
                    {day(e.expiry)} <span className="font-normal text-muted-foreground">· {e.days.toFixed(1)}d</span>
                  </td>
                  <td className="px-4 py-2.5 text-right">{e.n_points}</td>
                  <td className="px-4 py-2.5 text-right">{P(e.atm_iv)}</td>
                  <td className="px-4 py-2.5 text-right">{e.svi_rmse != null ? (e.svi_rmse * 100).toFixed(2) : "—"}</td>
                  <td className="px-4 py-2.5 text-right">{e.sabr_rmse != null ? (e.sabr_rmse * 100).toFixed(2) : "—"}</td>
                  <td className="px-4 py-2.5 text-right">
                    {e.sabr_rho != null ? e.sabr_rho.toFixed(2) : "—"} / {e.sabr_nu != null ? e.sabr_nu.toFixed(2) : "—"}
                  </td>
                  <td className={cn("px-4 py-2.5", e.butterfly_ok ? "text-up" : "text-down")}>{e.butterfly_ok ? "Pass" : "Fails"}</td>
                  <td className={cn("px-4 py-2.5", e.calendar_ok == null ? "text-muted-foreground" : e.calendar_ok ? "text-up" : "text-down")}>
                    {e.calendar_ok == null ? "Last expiry" : e.calendar_ok ? "Pass" : "Fails"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Method>
          <p>Forward from put–call parity on each expiry (F = K + (C − P) ÷ discount factor). Implied volatility from out-of-the-money mid prices with Black-76; quotes with too wide a spread or no trade are kept but not fitted.</p>
          <p>
            SVI (raw parameterisation) per expiry, checked for butterfly arbitrage (a non-negative implied density). Calendar check: total variance never falls for a later expiry at the same log-moneyness. SABR with β = 1 for
            comparison. The 25-delta risk reversal is the call's volatility minus the put's; the butterfly is their average minus at the money.
          </p>
        </Method>
      </CineSection>
    </>
  );
}
