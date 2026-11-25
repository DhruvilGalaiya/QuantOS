import type { Ratio, Statement, UsCompany } from "@quantos/types";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { AreaSeries, ColorType, createChart, CrosshairMode, LineStyle } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import Counter from "@/components/bits/Counter";
import { CineHero, CineSection, Segmented, Spot } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { ApiError, useApi } from "@/lib/api";
import { day } from "@/lib/format";
import { cssVar, useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const sp = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(d)}%`);
const toneCls = (x: number | null | undefined) => (x == null || x === 0 ? "" : x > 0 ? "text-up" : "text-down");

function money(x: number, cur = "$") {
  const a = Math.abs(x), s = x < 0 ? "−" : "";
  if (a >= 1e12) return `${s}${cur}${(a / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `${s}${cur}${(a / 1e9).toFixed(a >= 1e11 ? 0 : 1)}B`;
  if (a >= 1e6) return `${s}${cur}${(a / 1e6).toFixed(0)}M`;
  return `${s}${cur}${a.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

function ratioValue(r: Ratio, cur: string) {
  switch (r.kind) {
    case "money":
      return money(r.value, cur);
    case "pct":
      return `${(r.value * 100).toFixed(r.key === "dividend_yield" ? 2 : 0)}%`;
    case "pct_signed":
      return sp(r.value, 0);
    default:
      return `${r.value.toFixed(r.value >= 10 ? 0 : 2)}${r.key === "beta" || r.key.endsWith("pe") || r.key === "price_to_book" ? "×" : ""}`;
  }
}

/** Three or four plain sentences from the company's own numbers, so the page opens with a story, not a grid. */
function storyOf(c: UsCompany): string[] {
  const r = Object.fromEntries(c.ratios.map((x) => [x.key, x.value]));
  const name = (c.name ?? c.ticker).replace(/,? Inc\.?$|,? Corporation$|,? Corp\.?$| Holdings?,? Inc\.?$/, "");
  const out: string[] = [];
  if (r.net_margin != null) {
    let s = `${name} keeps $${Math.round(r.net_margin * 100)} of profit from every $100 of sales`;
    if (r.revenue_growth != null) s += `, and its sales ${r.revenue_growth >= 0 ? "grew" : "fell"} ${Math.abs(r.revenue_growth * 100).toFixed(0)}% on a year earlier`;
    out.push(`${s}.`);
  }
  if (r.pe != null) out.push(`Investors pay $${r.pe.toFixed(0)} for each $1 of its yearly profit${r.pe > 35 ? ", a price that assumes strong growth ahead" : r.pe < 15 ? ", a modest price for its earnings" : ""}.`);
  if (c.returns.ytd != null) out.push(`The share price is ${c.returns.ytd >= 0 ? "up" : "down"} ${Math.abs(c.returns.ytd * 100).toFixed(0)}% this year${c.returns.y5 != null ? ` and ${c.returns.y5 >= 0 ? "up" : "down"} ${Math.abs(c.returns.y5 * 100).toFixed(0)}% over five years` : ""}.`);
  if (r.debt_to_equity != null && r.debt_to_equity > 2) out.push(`It carries more debt than most: $${r.debt_to_equity.toFixed(1)} for every $1 of shareholders' money.`);
  return out;
}

// ------------------------------------------------------------------ price chart
const RANGES = [
  { value: "1m", label: "1M", days: 31 },
  { value: "6m", label: "6M", days: 183 },
  { value: "1y", label: "1Y", days: 366 },
  { value: "5y", label: "5Y", days: 5 * 366 },
  { value: "all", label: "All", days: 99999 },
] as const;
type Range = (typeof RANGES)[number]["value"];

function PriceChart({ c }: { c: UsCompany }) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useTheme((s) => s.theme);
  const [range, setRange] = useState<Range>("1y");
  const pts = useMemo(() => {
    const days = RANGES.find((r) => r.value === range)!.days;
    const cut = new Date(new Date(`${c.history.at(-1)!.time}T00:00:00Z`).getTime() - days * 864e5).toISOString().slice(0, 10);
    return c.history.filter((p) => p.time >= cut);
  }, [c.history, range]);
  const change = pts.length > 1 ? pts.at(-1)!.value / pts[0].value - 1 : null;
  const [hover, setHover] = useState<{ time: string; value: number } | null>(null);
  useEffect(() => {
    if (!el.current || !pts.length) return;
    const v = (n: string) => cssVar(n, el.current!);
    const col = (change ?? 0) >= 0 ? v("--up") : v("--down");
    const chart = createChart(el.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: v("--faint"), fontFamily: "IBM Plex Sans", fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: v("--line") } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: v("--muted-foreground"), style: LineStyle.Dotted, labelVisible: false }, horzLine: { visible: false, labelVisible: false } },
      handleScroll: false,
      handleScale: false,
      localization: { priceFormatter: (p: number) => `$${p.toFixed(p >= 1000 ? 0 : 2)}` },
    });
    const s = chart.addSeries(AreaSeries, { lineColor: col, lineWidth: 2, topColor: `color-mix(in srgb, ${col} 28%, transparent)`, bottomColor: "transparent", priceLineVisible: false });
    // lightweight-charts needs concrete colours: resolve the mix once through a probe element
    const probe = document.createElement("span");
    probe.style.color = `color-mix(in srgb, ${col} 28%, transparent)`;
    el.current.appendChild(probe);
    s.applyOptions({ topColor: getComputedStyle(probe).color });
    probe.remove();
    s.setData(pts);
    chart.timeScale().fitContent();
    const by = new Map(pts.map((p) => [p.time, p]));
    chart.subscribeCrosshairMove((e) => setHover(typeof e.time === "string" ? (by.get(e.time) ?? null) : null));
    return () => chart.remove();
  }, [pts, theme, change]);
  const shown = hover ?? pts.at(-1);
  return (
    <Spot>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="min-h-11 text-[14px] text-muted-foreground" aria-live="polite">
          {shown && (
            <>
              {day(shown.time, true)} <b className="num text-foreground">${shown.value.toFixed(2)}</b>
              {change != null && !hover && (
                <span className={cn("ml-3", toneCls(change))}>
                  {sp(change)} over {RANGES.find((r) => r.value === range)!.label === "All" ? "the full history" : `the last ${RANGES.find((r) => r.value === range)!.label}`}
                </span>
              )}
            </>
          )}
        </p>
        <Segmented label="Chart period" options={RANGES.map((r) => ({ value: r.value, label: r.label }))} value={range} onChange={setRange} />
      </div>
      <div ref={el} className="mt-3 h-[300px] md:h-[380px]" aria-label={`${c.name} share price, adjusted for splits and dividends`} role="img" />
    </Spot>
  );
}

// ------------------------------------------------------------------ statements
const TABLES = [
  { value: "annual", label: "Yearly results" },
  { value: "quarterly", label: "Quarterly results" },
  { value: "balance", label: "Balance sheet" },
  { value: "cashflow", label: "Cash flow" },
  { value: "ratios", label: "Ratios" },
] as const;
type TableKey = (typeof TABLES)[number]["value"];

function cell(v: number | null | undefined, kind: string, label: string, cur: string) {
  if (v == null) return "—";
  if (kind === "pct") return `${(v * 100).toFixed(1)}%`;
  if (kind === "eps") return `${cur}${v.toFixed(2)}`;
  if (kind === "times") return `${v.toFixed(2)}×`;
  if (label.includes("(M)")) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return money(v, cur);
}

function Statements({ c }: { c: UsCompany }) {
  const [tab, setTab] = useState<TableKey>("annual");
  const [open, setOpen] = useState(false);
  const t: Statement | null | undefined = c.financials[tab];
  const available = TABLES.filter((x) => (c.financials[x.value]?.rows?.length ?? 0) > 0);
  if (!available.length) return null;
  const cur = c.currency === "USD" ? "$" : `${c.currency} `;
  return (
    <CineSection id="statements" title={`${c.name}'s financial statements`} lede="Revenue, profit, debt and cash, as reported. Pick a statement; values are in the company's reporting currency.">
      {!open ? (
        <button type="button" className="btn-ghost" onClick={() => setOpen(true)}>
          Show the statements
        </button>
      ) : (
        <>
          <Segmented label="Statement" options={available.map((x) => ({ value: x.value, label: x.label }))} value={tab} onChange={setTab} />
          {t && (
            <div className="mt-5 overflow-x-auto rounded-xl border border-line">
              <table className="w-full min-w-[560px] text-[14px]">
                <thead className="bg-panel text-[12.5px] text-faint">
                  <tr>
                    <th className="sticky left-0 bg-panel px-4 py-2.5 text-left font-normal">Line</th>
                    {t.columns.map((col) => (
                      <th key={col} className="px-4 py-2.5 text-right font-normal">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="num">
                  {t.rows.map((r) => (
                    <tr key={r.label} className="border-t border-line">
                      <td className={cn("sticky left-0 bg-background px-4 py-2.5", r.bold ? "font-semibold" : "text-muted-foreground")}>{r.label}</td>
                      {r.values.map((v, i) => (
                        <td key={i} className={cn("px-4 py-2.5 text-right", r.bold && "font-semibold", v != null && v < 0 && "text-down")}>
                          {cell(v, r.kind, r.label, cur)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </CineSection>
  );
}

// ------------------------------------------------------------------ page
const GROUPS = ["Valuation", "Profitability", "Growth", "Income", "Safety", "Risk"];

export function UsCompanyPage() {
  const { ticker = "" } = useParams();
  const { data: c, error, isPending } = useApi<UsCompany>(`us/${ticker.toUpperCase()}`);
  const [more, setMore] = useState(false);
  useEffect(() => {
    if (c) document.title = `${c.name} (${c.ticker}) explained · QuantOS`;
  }, [c]);

  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error)
    return (
      <ErrorBox title={error instanceof ApiError && error.status === 404 ? `${ticker.toUpperCase()} is not in the list` : "This company could not be loaded"}>
        {error instanceof ApiError && error.status === 404 ? "QuantOS covers the largest S&P 500 and Nasdaq-100 companies. " : `${error.message}. `}
        <Link to="/us" className="text-saffron underline underline-offset-2">
          Back to all companies
        </Link>
      </ErrorBox>
    );

  const cur = c.currency === "USD" ? "$" : `${c.currency} `;
  const story = storyOf(c);
  const groups = GROUPS.map((g) => ({ g, items: c.ratios.filter((r) => r.group === g) })).filter((x) => x.items.length);
  const returns: [string, number | null | undefined][] = [
    ["1 month", c.returns.m1],
    ["6 months", c.returns.m6],
    ["This year", c.returns.ytd],
    ["1 year", c.returns.y1],
    ["5 years", c.returns.y5],
  ];

  return (
    <>
      <CineHero
        glow={c.change_pct >= 0 ? "rgb(47 191 138 / 0.2)" : "rgb(238 94 90 / 0.2)"}
        top={
          <Link to="/us" className="inline-flex min-h-11 items-center gap-2 text-[14px] text-muted-foreground transition-colors hover:text-foreground">
            <ArrowLeft className="size-4" aria-hidden />
            All companies
          </Link>
        }
        title={c.name ?? c.ticker}
        lede={c.summary.lead ? c.summary.lead.split(/(?<=\.)\s+(?=[A-Z])/)[0] : undefined}
      >
        <div className="flex flex-wrap items-end gap-x-10 gap-y-6">
          <div>
            <p className="text-[13.5px] text-muted-foreground">
              {c.ticker}
              {c.exchange ? ` on ${c.exchange}` : ""} · close of {day(c.as_of, true)}
            </p>
            <p className="mt-1.5 flex items-baseline gap-3">
              <span className="font-headline text-[clamp(2.4rem,1.8rem+2vw,3.4rem)] font-semibold leading-none tracking-tight">
                <Counter value={c.price} decimals={2} prefix={cur} locale="en-US" fontSize="1em" />
              </span>
              <span className={cn("num text-[17px] font-medium", toneCls(c.change_pct))}>{sp(c.change_pct, 2)}</span>
            </p>
          </div>
          <dl className="flex flex-wrap gap-2">
            {returns.map(([label, v]) => (
              <div key={label} className="rounded-xl border border-line bg-panel/60 px-3.5 py-2 backdrop-blur">
                <dt className="text-[12px] text-faint">{label}</dt>
                <dd className={cn("num text-[15px] font-semibold", toneCls(v))}>{sp(v, 0)}</dd>
              </div>
            ))}
          </dl>
        </div>
      </CineHero>

      <section aria-label="Price history" className="mt-6">
        <PriceChart c={c} />
      </section>

      {story.length > 0 && (
        <CineSection id="story" title={`${(c.name ?? c.ticker).split(" ")[0]} at a glance`}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <div className="space-y-4 text-[clamp(1.15rem,1rem+0.5vw,1.45rem)] leading-[1.55] text-foreground/90">
              {story.map((s) => (
                <p key={s}>{s}</p>
              ))}
            </div>
            <div className="grid gap-3">
              {c.facts.map((f) => (
                <Spot key={f.title}>
                  <p className="text-[13px] font-medium text-saffron">{f.title}</p>
                  <p className="mt-1 text-[15px] leading-snug">{f.text}</p>
                </Spot>
              ))}
            </div>
          </div>
        </CineSection>
      )}

      {groups.length > 0 && (
        <CineSection id="numbers" title="The numbers, in plain words" lede="Each figure comes with one sentence on what it means for this company. Tap through at your own pace.">
          <div className="space-y-10">
            {groups.map(({ g, items }) => (
              <div key={g}>
                <h3 className="mb-3 text-[15px] font-semibold text-muted-foreground">{g}</h3>
                <div className={cn("grid gap-4 sm:grid-cols-2", items.length === 4 ? "xl:grid-cols-4" : items.length >= 3 ? "xl:grid-cols-3" : "")}>
                  {items.map((r) => (
                    <Spot key={r.key}>
                      <p className="text-[13.5px] text-muted-foreground">{r.label}</p>
                      <p className={cn("mt-1.5 font-headline text-[30px] font-semibold leading-none tracking-tight", r.kind === "pct_signed" && toneCls(r.value))}>{ratioValue(r, cur)}</p>
                      <p className="mt-3 text-[14.5px] leading-snug text-muted-foreground">{r.plain}</p>
                    </Spot>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CineSection>
      )}

      {c.summary.lead && (
        <CineSection id="about" title={`What ${(c.name ?? c.ticker).split(" ")[0]} does`}>
          <div className="max-w-[70ch] space-y-4 text-[16.5px] leading-[1.7] text-muted-foreground">
            <p>{c.summary.lead}</p>
            {c.summary.rest && more && <p>{c.summary.rest}</p>}
          </div>
          <div className="mt-5 flex flex-wrap gap-3">
            {c.summary.rest && (
              <button type="button" className="btn-ghost" aria-expanded={more} onClick={() => setMore((m) => !m)}>
                {more ? "Show less" : "Read the full description"}
              </button>
            )}
            {c.website && /^https?:\/\//.test(c.website) && (
              <a href={c.website} target="_blank" rel="noreferrer noopener" className="btn-ghost">
                Visit the company website
                <ArrowUpRight className="size-4" aria-hidden />
              </a>
            )}
          </div>
        </CineSection>
      )}

      <Statements c={c} />
      <p className="mt-16 border-t border-line pt-6 text-[13.5px] text-muted-foreground">
        Prices and company facts from Yahoo Finance{c.fundamentals_as_of ? `, facts updated ${day(c.fundamentals_as_of.slice(0, 10), true)}` : ""}. Analysis and education, not investment advice.
      </p>
    </>
  );
}
