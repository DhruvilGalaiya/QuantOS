/*
  QuantOS landing page.
  Kind: software (fintech analytics). One effect: Topography contour terrain in the hero (reads as a volatility
  surface). Moments: the scroll-revealed principle statement and the live pipeline. Every visual is the product's
  own UI fed by today's real data (static JSON on the public site, the gateway in development).
*/
import type { IndexBrief, Today } from "@quantos/types";
import { ArrowUpRight } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { siGithub, siFastapi, siFastify, siNodedotjs, siNumpy, siPandas, siPython, siReact, siScikitlearn, siScipy, siSqlite, siTailwindcss, siTypescript, siVite } from "simple-icons";
import BlurText from "@/components/bits/BlurText";
import Counter from "@/components/bits/Counter";
import { LogoLoop } from "@/components/bits/LogoLoop";
import Magnet from "@/components/bits/Magnet";
import ScrollReveal from "@/components/bits/ScrollReveal";
import SpotlightCard from "@/components/bits/SpotlightCard";
import Topography from "@/components/bits/Topography";
import { PriceCone } from "@/components/charts/PriceCone";
import { IndexPanel } from "@/pages/Today";
import { useRoute } from "@/lib/api";
import { day, level, pct, pts, rupees, share, tone } from "@/lib/format";
import { cn } from "@/lib/utils";

const REPO = "https://github.com/DhruvilGalaiya/QuantOS";
const PRIMARY = "Open today's brief";

function useReducedMotion() {
  const [reduce, setReduce] = useState(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduce(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduce;
}

function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <path d="M5 22 L12.5 14.5 L18 18.5 L27 8.5" fill="none" stroke="var(--accent-saffron)" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ------------------------------------------------------------------ nav
function Nav() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 24);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-40 transition-[background-color,border-color] duration-300",
        scrolled ? "border-b border-white/[0.07] bg-[#060a12]/70 backdrop-blur-xl backdrop-saturate-150" : "border-b border-transparent",
      )}
    >
      <div className="wrap flex h-16 items-center gap-8">
        <Link to="/" className="flex min-h-11 items-center gap-2.5" aria-label="QuantOS home">
          <Mark className="size-6" />
          <span className="font-headline text-[19px] font-semibold tracking-tight">QuantOS</span>
        </Link>
        <nav aria-label="Sections" className="hidden items-center gap-7 text-[14px] text-muted-foreground md:flex">
          <a href="#measures" className="link-draw">What it measures</a>
          <a href="#methods" className="link-draw">Methods</a>
          <a href={REPO} className="link-draw" target="_blank" rel="noreferrer">Source code</a>
        </nav>
        <Link to="/today" className="btn-solid ml-auto !min-h-11 !px-4 !text-[14px] max-sm:hidden">
          {PRIMARY}
        </Link>
      </div>
    </header>
  );
}

// ------------------------------------------------------------------ hero
function LiveLine({ i, asOf }: { i: IndexBrief; asOf: string | null | undefined }) {
  const t = tone(i.market.change_pct);
  return (
    <p className="inline-flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-[13.5px] text-muted-foreground backdrop-blur-md">
      <span className={cn("live-dot size-1.5 rounded-full", t === "down" ? "bg-down text-down" : "bg-up text-up")} aria-hidden />
      <span>{i.name}</span>
      <Counter value={i.market.close} locale="en-IN" decimals={2} fontSize={14} fontWeight={600} textColor="var(--foreground)" />
      <span className={cn("num font-medium", t === "down" ? "text-down" : "text-up")}>{pct(i.market.change_pct, 2, true)}</span>
      <span>at the close of {day(asOf)}</span>
    </p>
  );
}

function Hero({ data, reduce }: { data: Today | undefined; reduce: boolean }) {
  const nifty = data?.indices.find((i) => i.symbol === "NIFTY_50");
  return (
    <section className="relative isolate flex min-h-screen items-center overflow-hidden pb-16 pt-28" aria-label="QuantOS" data-enter>
      <div className="absolute inset-0 -z-10" aria-hidden>
        <Topography
          color="#EFA43A"
          lineCount={30}
          lineWidth={1.1}
          elevation={1.25}
          tilt={38}
          speed={reduce ? 0 : 0.18}
          glow={0.75}
          fog={0.5}
          depthOfField={0.35}
          brightness={0.95}
          hoverEffect="lift"
          clickEffect="pulse"
          mouseInteraction={!reduce}
        />
        {/* keep the headline legible over the terrain, then fade into the page */}
        <div className="absolute inset-0 bg-[radial-gradient(60%_50%_at_50%_46%,rgb(6_10_18/0.93),rgb(6_10_18/0.55)_70%,rgb(6_10_18/0.15))]" />
        <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-[#060a12]" />
      </div>

      <div className="wrap flex flex-col items-center text-center">
        <div className="min-h-9">{nifty && <LiveLine i={nifty} asOf={data?.as_of} />}</div>
        <h1 className="headline mt-7 max-w-[16ch] text-[clamp(2.6rem,1.2rem+5.6vw,5.9rem)]">
          {reduce ? (
            "Volatility and risk analytics for Indian index traders"
          ) : (
            <BlurText text="Volatility and risk analytics for Indian index traders" animateBy="words" trigger="load" delay={60} stepDuration={0.38} className="justify-center" />
          )}
        </h1>
        <p className="mt-6 max-w-[58ch] text-[clamp(1rem,0.9rem+0.4vw,1.2rem)] leading-relaxed text-muted-foreground">
          QuantOS measures how NIFTY 50 and NIFTY BANK are moving, what option sellers are being paid and what followed similar days since 2007. Free, rebuilt after every NSE close.
        </p>
        <div className="mt-9 flex w-full flex-col items-center justify-center gap-3 sm:w-auto sm:flex-row">
          <Magnet padding={60} magnetStrength={4} maxOffset={10} wrapperClassName="w-full sm:w-auto" innerClassName="w-full sm:w-auto" disabled={reduce}>
            <Link to="/today" className="btn-solid w-full sm:w-auto">
              {PRIMARY}
            </Link>
          </Magnet>
          <a href="#measures" className="btn-ghost w-full sm:w-auto">
            See what it measures
          </a>
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ section shell
function Section({ id, title, lede, children, className, screens = 1, aside }: { id: string; title: ReactNode; lede?: ReactNode; children: ReactNode; className?: string; screens?: 1 | 2; aside?: ReactNode }) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-h`}
      className={cn("relative flex scroll-mt-16 flex-col justify-center py-20 md:py-24", screens === 2 ? "md:min-h-[200vh]" : "md:min-h-screen", className)}
    >
      <div className="wrap">
        <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_auto]">
          <div>
            <h2 id={`${id}-h`} className="h-section max-w-[20ch]">
              {title}
            </h2>
            {lede && <p className="mt-5 max-w-[56ch] text-[17px] leading-relaxed text-muted-foreground">{lede}</p>}
          </div>
          {aside}
        </div>
        {children}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ the product, in its frame
function BriefFrame({ data }: { data: Today }) {
  return (
    <Section id="brief" screens={2} title="Today's market brief on one screen" lede="Both indices with their regime, volatility, option-implied range and the numbers behind them, rebuilt from the latest close.">
      <div className="relative mt-16">
        <div className="horizon pointer-events-none absolute inset-x-0 -top-24 h-72" aria-hidden />
        <div className="horizon-line pointer-events-none absolute inset-x-[12%] -top-px h-px" aria-hidden />
        <div className="relative overflow-hidden rounded-xl border border-white/10 bg-background shadow-[0_40px_120px_-40px_rgb(239_164_58/0.35)]">
          <div className="flex h-10 items-center gap-2 border-b border-line bg-panel px-4">
            <span className="size-2.5 rounded-full bg-white/15" aria-hidden />
            <span className="size-2.5 rounded-full bg-white/15" aria-hidden />
            <span className="size-2.5 rounded-full bg-white/15" aria-hidden />
            <span className="mx-auto rounded-md bg-panel-2 px-3 py-0.5 font-mono text-[12px] text-muted-foreground">quantos / today</span>
          </div>
          <div className="grid grid-cols-1 gap-4 p-3 sm:p-5 xl:grid-cols-2">
            {data.indices.map((i) => (
              <IndexPanel key={i.symbol} i={i} heading="h3" />
            ))}
          </div>
        </div>
        <div className="mt-8 flex justify-center">
          <Link to="/today" className="btn-ghost">
            Read the full brief
            <ArrowUpRight className="size-4" aria-hidden />
          </Link>
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ the implied range
function RangeSection({ data }: { data: Today }) {
  const n = data.indices.find((i) => i.symbol === "NIFTY_50") ?? data.indices[0];
  const imp = n?.implied;
  if (!n || !imp) return null;
  return (
    <Section id="range" title="The range option prices charge for">
      <div className="mt-14 grid items-start gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div>
          <p className="max-w-[44ch] text-[17px] leading-relaxed text-muted-foreground">
            The at-the-money call and put tell you how far traders are paying to insure against. Sellers profit while {n.name} stays inside the range; buyers need a bigger move.
          </p>
          <dl className="mt-10 space-y-7">
            <div>
              <dt className="text-[14px] text-muted-foreground">Range to {day(imp.expiry)}</dt>
              <dd className="mt-1 flex items-baseline gap-2 font-headline text-[clamp(2rem,1.4rem+2vw,3rem)] font-semibold tracking-tight">
                <Counter value={Math.round(imp.lower)} locale="en-IN" fontSize="1em" />
                <span className="text-faint">–</span>
                <Counter value={Math.round(imp.upper)} locale="en-IN" fontSize="1em" />
              </dd>
            </div>
            <div className="grid grid-cols-2 gap-6 border-t border-white/10 pt-6">
              <div>
                <dt className="text-[14px] text-muted-foreground">Move priced in</dt>
                <dd className="num mt-1 text-[22px] font-semibold">
                  ±{level(imp.move_points, 0)} <span className="text-[15px] font-normal text-muted-foreground">pts, ±{pct(imp.move_pct, 2)}</span>
                </dd>
              </div>
              <div>
                <dt className="text-[14px] text-muted-foreground">Call + put at {level(imp.atm_strike, 0)}</dt>
                <dd className="num mt-1 text-[22px] font-semibold">{rupees(imp.straddle)}</dd>
              </div>
            </div>
            <p className="text-[15px] text-muted-foreground">By option pricing the index finishes inside this range about 2 times in 3.</p>
          </dl>
        </div>
        <div className="rounded-xl border border-white/10 bg-panel/60 p-4 backdrop-blur sm:p-6">
          <PriceCone history={n.history} implied={imp} height={340} label={n.name} />
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ what it measures (bento of live fragments)
const STATE_COLOUR = { CALM: "var(--calm)", NORMAL: "var(--normal)", STRESSED: "var(--stressed)" } as const;
const REGIME_COLOUR = { BULL: "var(--up)", SIDEWAYS: "var(--normal)", BEAR: "var(--down)" } as const;

function Card({ title, body, to, children, className }: { title: string; body: string; to: string; children: ReactNode; className?: string }) {
  return (
    <Link to={to} className={cn("lift group block h-full min-w-0 rounded-2xl focus-visible:outline-offset-4", className)}>
      <SpotlightCard
        className="flex h-full flex-col rounded-2xl! p-6! sm:p-7!"
        spotlightColor="rgba(239, 164, 58, 0.55)"
        intensity={0.12}
        borderGlow={0.8}
        style={{ "--spotlight-card-surface": "#0d1526", "--spotlight-card-border": "rgba(255,255,255,0.08)" } as React.CSSProperties}
      >
        <div className="min-h-[132px] flex-1">{children}</div>
        <h3 className="mt-6 text-[17px] font-semibold">
          {title}
          <ArrowUpRight className="ml-1 inline size-4 align-[-2px] text-faint transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-saffron" aria-hidden />
        </h3>
        <p className="mt-1.5 text-[14.5px] leading-snug text-muted-foreground">{body}</p>
      </SpotlightCard>
    </Link>
  );
}

function StateBars({ labels, active, colours }: { labels: [string, string][]; active: string | undefined; colours: Record<string, string> }) {
  return (
    <div className="flex gap-2">
      {labels.map(([key, label]) => (
        <span
          key={key}
          className={cn("flex-1 rounded-md border px-2 py-2 text-center text-[13px] font-medium transition-colors", key === active ? "text-[#0b1322]" : "border-white/10 text-faint")}
          style={key === active ? { background: colours[key], borderColor: colours[key] } : undefined}
        >
          {label}
        </span>
      ))}
    </div>
  );
}

function Smile({ imp }: { imp: NonNullable<IndexBrief["implied"]> }) {
  // a quadratic through the 25-delta put, at-the-money and 25-delta call vols (from risk reversal and butterfly)
  const atm = imp.atm_iv, rr = imp.rr25 ?? 0, bf = imp.bf25 ?? 0;
  const put = atm + bf - rr / 2, call = atm + bf + rr / 2;
  const a = (put + call) / 2 - atm, b = (call - put) / 2;
  const xs = Array.from({ length: 41 }, (_, k) => -1.6 + (3.2 * k) / 40);
  const ys = xs.map((x) => atm + b * x + a * x * x);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const px = (x: number) => ((x + 1.6) / 3.2) * 300;
  const py = (y: number) => 100 - ((y - lo) / Math.max(hi - lo, 1e-4)) * 80;
  const d = xs.map((x, k) => `${k ? "L" : "M"}${px(x).toFixed(1)},${py(ys[k]).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 300 120" className="h-[120px] w-full" role="img" aria-label={`Front-expiry implied volatility smile, at the money ${pct(atm)}`}>
      {[30, 60, 90].map((y) => (
        <line key={y} x1="0" x2="300" y1={y} y2={y} stroke="rgba(255,255,255,0.06)" />
      ))}
      <path d={d} fill="none" stroke="var(--accent-saffron)" strokeWidth="2.2" />
      <circle cx={px(0)} cy={py(atm)} r="4" fill="var(--accent-saffron)" />
      <text x={px(0) + 8} y={py(atm) - 8} fill="var(--foreground)" fontSize="13" fontFamily="IBM Plex Sans">
        {pct(atm)} at the money
      </text>
      <text x="0" y="116" fill="var(--faint)" fontSize="11" fontFamily="IBM Plex Sans">Lower strikes</text>
      <text x="300" y="116" fill="var(--faint)" fontSize="11" fontFamily="IBM Plex Sans" textAnchor="end">Higher strikes</text>
    </svg>
  );
}

function McHistogram({ risk }: { risk: NonNullable<Today["risk"]> }) {
  const { mc_hist: h, mc_edges: e } = risk;
  if (!h.length || e.length !== h.length + 1) return null;
  // trim the long empty tails so the shape is readable
  const first = Math.max(0, h.findIndex((v) => v > 4) - 1);
  const last = Math.min(h.length - 1, h.length - 1 - [...h].reverse().findIndex((v) => v > 4) + 1);
  const bins = h.slice(first, last + 1);
  const max = Math.max(...bins);
  const w = 300 / bins.length;
  return (
    <svg viewBox="0 0 300 110" className="h-[110px] w-full" role="img" aria-label={`${risk.mc_paths.toLocaleString("en-IN")} simulated ${risk.mc_days}-session NIFTY returns`}>
      {bins.map((v, k) => {
        const mid = (e[first + k] + e[first + k + 1]) / 2;
        const ht = (v / max) * 92;
        return <rect key={k} x={k * w + 0.5} y={96 - ht} width={Math.max(w - 1, 0.5)} height={ht} rx="1" fill={mid < (risk.mc_p05 ?? -1) ? "var(--down)" : mid < 0 ? "rgba(238,94,90,0.45)" : "rgba(239,164,58,0.75)"} />;
      })}
      <line x1="0" x2="300" y1="96.5" y2="96.5" stroke="rgba(255,255,255,0.12)" />
      <text x="0" y="109" fill="var(--faint)" fontSize="11" fontFamily="IBM Plex Sans">Loss</text>
      <text x="300" y="109" fill="var(--faint)" fontSize="11" fontFamily="IBM Plex Sans" textAnchor="end">Gain</text>
    </svg>
  );
}

function Measures({ data }: { data: Today }) {
  const n = data.indices.find((i) => i.symbol === "NIFTY_50") ?? data.indices[0];
  const v = n?.vol;
  const vixMax = Math.max(v?.vix ?? 0, v?.rv21 ?? 0, 0.01) * 1.15;
  return (
    <Section id="measures" screens={2} title="Six measurements behind every reading" lede="Each card shows today's value for NIFTY 50. Open one to see the full study.">
      <div className="mt-14 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-6">
        <Card className="lg:col-span-3" to="/regimes" title="Labels bull, sideways and bear regimes" body="A three-state Gaussian hidden Markov model fitted on returns, volatility and sector spread.">
          <StateBars labels={[["BULL", "Bull"], ["SIDEWAYS", "Sideways"], ["BEAR", "Bear"]]} active={n?.regime?.regime} colours={REGIME_COLOUR} />
          {n?.regime && (
            <p className="mt-5 text-[15px] text-muted-foreground">
              <span className="font-headline text-[34px] font-semibold text-foreground">{n.regime.streak}</span> sessions in a {n.regime.label.toLowerCase()} regime
            </p>
          )}
        </Card>
        <Card className="lg:col-span-3" to="/states" title="Shows how violently the market moves" body="Calm, normal or stressed, from realized volatility, with how long each state usually lasts.">
          <StateBars labels={[["CALM", "Calm"], ["NORMAL", "Normal"], ["STRESSED", "Stressed"]]} active={n?.state?.state} colours={STATE_COLOUR} />
          {n?.state?.still_after_5 != null && (
            <p className="mt-5 text-[15px] text-muted-foreground">
              <span className="font-headline text-[34px] font-semibold text-foreground">{share(n.state.still_after_5)}</span> chance it is still {n.state.label.toLowerCase()} in 5 sessions
            </p>
          )}
        </Card>
        <Card className="lg:col-span-2" to="/volatility" title="Compares implied with realized volatility" body="India VIX against how much NIFTY really moved. The gap is the seller's edge.">
          {v?.vix != null && v.rv21 != null && (
            <div className="space-y-3">
              {[
                ["India VIX", v.vix, "var(--accent-saffron)"],
                ["Realized, 21 sessions", v.rv21, "var(--calm)"],
              ].map(([label, x, c]) => (
                <div key={label as string}>
                  <div className="flex justify-between text-[13px]">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="num font-medium">{pct(x as number)}</span>
                  </div>
                  <div className="mt-1.5 h-2 rounded-full bg-white/[0.06]">
                    <div className="h-2 rounded-full" style={{ width: `${((x as number) / vixMax) * 100}%`, background: c as string }} />
                  </div>
                </div>
              ))}
              <p className="num pt-1 text-[15px]">
                Premium <span className={cn("font-semibold", tone(v.vrp_now) === "up" ? "text-up" : "text-down")}>{pts(v.vrp_now)}</span>
              </p>
            </div>
          )}
        </Card>
        <Card className="lg:col-span-2" to="/surface" title="Maps implied volatility by strike" body="SVI and SABR fits across strikes and expiries, refreshed from the live option chain.">
          {n?.implied && <Smile imp={n.implied} />}
        </Card>
        <Card className="lg:col-span-2" to="/risk" title="Sizes the worst day in twenty" body={`${data.risk ? data.risk.mc_paths.toLocaleString("en-IN") : "10,000"} GARCH-t paths over ${data.risk?.mc_days ?? 21} sessions, plus value at risk at today's volatility.`}>
          {data.risk && (
            <>
              <McHistogram risk={data.risk} />
              <p className="num mt-2 text-[15px]">
                1-day 95% loss <span className="font-semibold text-down">{pct(data.risk.var95, 1)}</span>
                <span className="text-muted-foreground">, {rupees(data.risk.var95 * 1e5)} per ₹1 lakh</span>
              </p>
            </>
          )}
        </Card>
        <Card className="md:col-span-2 lg:col-span-6" to="/events" title="Tests what followed past signals" body="Event studies with bootstrap confidence intervals: 52-week highs, losing streaks, VIX spikes and more.">
          <div className="-mx-1 overflow-x-auto">
            <table className="w-full min-w-[520px] text-[14px]">
              <thead>
                <tr className="text-left text-[12.5px] text-faint">
                  <th className="px-1 pb-2 font-normal">Signal this week</th>
                  <th className="px-1 pb-2 font-normal">Index</th>
                  <th className="px-1 pb-2 text-right font-normal">Past cases</th>
                  <th className="px-1 pb-2 text-right font-normal">Avg next month</th>
                  <th className="px-1 pb-2 text-right font-normal">Higher a month later</th>
                </tr>
              </thead>
              <tbody className="num">
                {data.events.slice(0, 3).map((e) => (
                  <tr key={`${e.index}-${e.id}`} className="border-t border-white/[0.06]">
                    <td className="px-1 py-2 font-medium">{e.name}</td>
                    <td className="px-1 py-2 text-muted-foreground">{e.index}</td>
                    <td className="px-1 py-2 text-right">{e.n}</td>
                    <td className={cn("px-1 py-2 text-right", tone(e.mean) === "up" ? "text-up" : "text-down")}>{pct(e.mean, 2, true)}</td>
                    <td className="px-1 py-2 text-right">{share(e.hit_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ pipeline (pulsing diagram)
type Node = { id: string; x: number; y: number; w: number; label: string; sub: string; accent?: boolean };

function Pipeline({ nodes, links, viewBox, className }: { nodes: Node[]; links: [string, string][]; viewBox: string; className?: string }) {
  const at = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const H = 58;
  const [, , vw] = viewBox.split(" ").map(Number);
  const vertical = vw < 600;
  const path = (a: Node, b: Node) => {
    if (vertical) {
      const x1 = a.x + a.w / 2, y1 = a.y + H, x2 = b.x + b.w / 2, y2 = b.y;
      const my = (y1 + y2) / 2;
      return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
    }
    const x1 = a.x + a.w, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
    const mx = (x1 + x2) / 2;
    return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
  };
  return (
    <svg viewBox={viewBox} className={className} role="img" aria-label="Data pipeline: Upstox market data, nightly jobs, four models, the API, and this site">
      {links.map(([a, b], k) => (
        <g key={`${a}-${b}`}>
          <path d={path(at[a], at[b])} className="pipe-line" />
          <path d={path(at[a], at[b])} className="pipe-pulse" style={{ animationDelay: `${(k % 4) * 0.55}s` }} />
        </g>
      ))}
      {nodes.map((n) => (
        <g key={n.id}>
          <rect x={n.x} y={n.y} width={n.w} height={H} rx="10" fill={n.accent ? "rgba(239,164,58,0.1)" : "#0d1526"} stroke={n.accent ? "rgba(239,164,58,0.6)" : "rgba(255,255,255,0.1)"} />
          <text x={n.x + 16} y={n.y + 25} fill="var(--foreground)" fontSize="15" fontWeight="600" fontFamily="IBM Plex Sans">
            {n.label}
          </text>
          <text x={n.x + 16} y={n.y + 44} fill="var(--muted-foreground)" fontSize="12.5" fontFamily="IBM Plex Sans">
            {n.sub}
          </text>
        </g>
      ))}
    </svg>
  );
}

const MODEL_NODES = [
  { id: "hmm", label: "Gaussian HMM", sub: "3 market regimes" },
  { id: "garch", label: "GARCH(1,1), Student-t", sub: "volatility forecast" },
  { id: "svi", label: "SVI and SABR", sub: "volatility surface" },
  { id: "mc", label: "Monte Carlo", sub: "10,000 paths" },
];
const LINKS: [string, string][] = [
  ["src", "jobs"],
  ...MODEL_NODES.map((m) => ["jobs", m.id] as [string, string]),
  ...MODEL_NODES.map((m) => [m.id, "api"] as [string, string]),
  ["api", "site"],
];
const WIDE: Node[] = [
  { id: "src", x: 0, y: 152, w: 170, label: "Upstox API", sub: "prices, option chains" },
  { id: "jobs", x: 215, y: 152, w: 180, label: "Nightly jobs", sub: "checks and features" },
  ...MODEL_NODES.map((m, k) => ({ ...m, x: 450, y: 10 + k * 95, w: 230, accent: true })),
  { id: "api", x: 735, y: 152, w: 200, label: "FastAPI and Node", sub: "typed, cached API" },
  { id: "site", x: 990, y: 152, w: 170, label: "QuantOS", sub: "this site" },
];
const TALL: Node[] = [
  { id: "src", x: 80, y: 0, w: 200, label: "Upstox API", sub: "prices, option chains" },
  { id: "jobs", x: 80, y: 100, w: 200, label: "Nightly jobs", sub: "checks and features" },
  ...MODEL_NODES.map((m, k) => ({ ...m, x: k % 2 ? 186 : 0, y: 210 + Math.floor(k / 2) * 92, w: 174, label: m.label.replace("GARCH(1,1), Student-t", "GARCH-t") })),
  { id: "api", x: 80, y: 420, w: 200, label: "FastAPI and Node", sub: "typed, cached API" },
  { id: "site", x: 80, y: 520, w: 200, label: "QuantOS", sub: "this site" },
];

function Methods() {
  const facts: [string, string][] = [
    ["2007", "daily history starts"],
    ["18", "NSE indices, India VIX included"],
    ["0", "days of look-ahead in model tests"],
  ];
  return (
    <Section
      id="methods"
      title="From the NSE close to this page"
      lede="After 3:45 pm IST each trading day, jobs pull prices and option chains, check them, refit the models and publish the results here."
      aside={
        <dl className="grid grid-cols-3 gap-6 lg:gap-10">
          {facts.map(([n, label]) => (
            <div key={label} className="max-w-[12ch]">
              <dt className="sr-only">{label}</dt>
              <dd className="font-headline text-[34px] font-semibold leading-none">{n}</dd>
              <dd className="mt-2 text-[14px] leading-snug text-muted-foreground">{label}</dd>
            </div>
          ))}
        </dl>
      }
    >
      <div className="mt-12 rounded-2xl border border-white/[0.07] bg-[#0a1120] p-5 sm:p-8">
        <Pipeline nodes={WIDE} links={LINKS} viewBox="0 0 1160 368" className="hidden w-full md:block" />
        <Pipeline nodes={TALL} links={LINKS} viewBox="0 0 360 580" className="mx-auto w-full max-w-[360px] md:hidden" />
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ principle (scroll moment)
function Principle({ reduce }: { reduce: boolean }) {
  const text =
    "QuantOS never tells you where NIFTY goes tomorrow. It shows where the market stands, what options charge for the next move and what history did after days like this one. The decision stays with you.";
  return (
    <section id="principle" aria-label="Measurements, not predictions" className="relative flex items-center py-24 md:min-h-screen">
      <div className="wrap">
        <p className="text-[15px] font-medium text-saffron">Measurements, not predictions</p>
        {reduce ? (
          <h2 className="headline mt-6 max-w-[1040px] text-[clamp(2rem,1.1rem+3.4vw,4rem)] leading-[1.08]">{text}</h2>
        ) : (
          <ScrollReveal
            baseOpacity={0.14}
            enableBlur
            blurStrength={6}
            baseRotation={0}
            containerClassName="mt-6 max-w-[1040px]"
            textClassName="headline !text-[clamp(2rem,1.1rem+3.4vw,4rem)] !leading-[1.08]"
            wordAnimationEnd="center center"
          >
            {text}
          </ScrollReveal>
        )}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ stack, with the real API response as its picture
const STACK = [siPython, siPandas, siNumpy, siScipy, siScikitlearn, siFastapi, siSqlite, siTypescript, siReact, siNodedotjs, siFastify, siVite, siTailwindcss];

function ApiWindow({ data }: { data: Today | undefined }) {
  const n = data?.indices.find((i) => i.symbol === "NIFTY_50");
  const body = n
    ? {
        as_of: data?.as_of,
        indices: [
          {
            name: n.name,
            close: n.market.close,
            regime: n.regime?.label,
            state: n.state?.label,
            rv21: +(n.vol.rv21 ?? 0).toFixed(4),
            vix: +(n.vol.vix ?? 0).toFixed(4),
            implied: n.implied ? { expiry: n.implied.expiry, lower: Math.round(n.implied.lower), upper: Math.round(n.implied.upper) } : null,
          },
        ],
        risk: data?.risk ? { var95: +data.risk.var95.toFixed(4), mc_paths: data.risk.mc_paths } : null,
      }
    : null;
  const json = body ? JSON.stringify(body, null, 2) : "";
  // colour keys, strings and numbers like an editor would
  const html = json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?)/g, (_m, str, colon, num) =>
      str ? (colon ? `<span class="text-calm">${str}</span>${colon}` : `<span class="text-saffron">${str}</span>`) : `<span class="text-up">${num}</span>`,
    );
  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-[#0a1120] shadow-[0_30px_80px_-30px_rgb(0_0_0/0.8)]">
      <div className="flex items-center gap-3 border-b border-white/[0.07] px-4 py-2.5 font-mono text-[12.5px]">
        <span className="rounded bg-up/15 px-1.5 py-0.5 font-medium text-up">GET</span>
        <span className="text-muted-foreground">/api/v1/today</span>
        <span className="ml-auto text-faint">200 OK</span>
      </div>
      <pre className="overflow-x-auto p-5 font-mono text-[12.5px] leading-relaxed text-muted-foreground">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}

function Stack({ data, reduce }: { data: Today | undefined; reduce: boolean }) {
  const logos = useMemo(
    () =>
      STACK.map((s) => ({
        title: s.title,
        node: (
          <span className="flex items-center gap-2.5 rounded-xl bg-white px-4 py-2.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)]">
            <svg viewBox="0 0 24 24" className="size-6" aria-hidden>
              <path d={s.path} fill={`#${s.hex}`} />
            </svg>
            <span className="text-[14px] font-semibold text-[#0b1322]">{s.title}</span>
          </span>
        ),
      })),
    [],
  );
  return (
    <section id="stack" aria-labelledby="stack-h" className="flex flex-col justify-center py-20 md:min-h-screen md:py-24">
      <div className="wrap grid items-start gap-12 lg:grid-cols-2">
        <div>
          <h2 id="stack-h" className="h-section max-w-[20ch]">
            Built in Python, TypeScript and React
          </h2>
          <p className="mt-5 max-w-[48ch] text-[17px] leading-relaxed text-muted-foreground">
            The models run in a tested Python engine. A Node gateway serves them as a typed JSON API to this React site. Every line is open on GitHub.
          </p>
          <a href={REPO} target="_blank" rel="noreferrer" className="btn-ghost mt-8">
            <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
              <path d={siGithub.path} fill="currentColor" />
            </svg>
            Read the source code
          </a>
        </div>
        <ApiWindow data={data} />
      </div>
      <div className="mt-16">
        <LogoLoop logos={logos} intro={false} speed={reduce ? 0 : 42} gap={14} logoHeight={44} pauseOnHover fadeOut fadeOutColor="#060a12" ariaLabel="Technologies QuantOS is built with" />
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ close + footer, one screen together
function Close({ asOf }: { asOf: string | null | undefined }) {
  return (
    <div className="relative isolate flex min-h-screen flex-col overflow-hidden">
      <div className="horizon pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-full rotate-180 opacity-80" aria-hidden />
      <section aria-labelledby="close-h" className="flex flex-1 items-center py-24">
        <div className="wrap flex flex-col items-center text-center">
          <h2 id="close-h" className="headline max-w-[16ch] text-[clamp(2.4rem,1.2rem+4.4vw,5rem)]">
            Today's brief is free to read
          </h2>
          <p className="mt-5 max-w-[48ch] text-[17px] text-muted-foreground">No account and no paywall. The numbers update after every close.</p>
          <Link to="/today" className="btn-solid mt-9">
            {PRIMARY}
          </Link>
        </div>
      </section>
      <footer className="border-t border-white/[0.07] py-10 text-[14px] text-muted-foreground">
        <div className="wrap grid gap-6 md:grid-cols-[1fr_auto]">
          <div>
            <Link to="/" className="inline-flex min-h-11 items-center gap-2 text-foreground" aria-label="QuantOS home">
              <Mark className="size-5" />
              <span className="font-headline text-[17px] font-semibold">QuantOS</span>
            </Link>
            <p className="mt-1 max-w-[64ch] text-[13px] text-faint">
              Analysis and education, not investment advice. Market data from NSE via the Upstox API{asOf ? `, last session ${day(asOf, true)}` : ""}. Product names and logos are trademarks of their owners. © {new Date().getFullYear()} QuantOS
            </p>
          </div>
          <nav aria-label="Footer" className="flex flex-wrap items-start gap-x-6">
            <Link to="/today" className="link-draw inline-flex min-h-11 items-center">Today</Link>
            <a href="#methods" className="link-draw inline-flex min-h-11 items-center">Methods</a>
            <a href={REPO} target="_blank" rel="noreferrer" className="link-draw inline-flex min-h-11 items-center">GitHub</a>
          </nav>
        </div>
      </footer>
    </div>
  );
}

export function LandingPage() {
  const { data } = useRoute("today");
  const reduce = useReducedMotion();
  useEffect(() => {
    document.title = "QuantOS: volatility and risk analytics for Indian index traders";
  }, []);
  return (
    <div className="landing dark min-h-dvh overflow-x-clip text-foreground">
      <a href="#brief" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-panel focus:px-3 focus:py-2">
        Skip to content
      </a>
      <Nav />
      <main>
        <Hero data={data} reduce={reduce} />
        {data && data.indices.length > 0 && (
          <>
            <BriefFrame data={data} />
            <RangeSection data={data} />
            <Measures data={data} />
          </>
        )}
        <Methods />
        <Principle reduce={reduce} />
        <Stack data={data} reduce={reduce} />
      </main>
      <Close asOf={data?.as_of} />
    </div>
  );
}
