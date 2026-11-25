import type { UsRow, UsUniverse } from "@quantos/types";
import { Search } from "lucide-react";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import DotGrid from "@/components/bits/DotGrid";
import { Sparkline } from "@/components/charts/svg";
import { CineHero, CineSection, Segmented, Spot, useReducedMotion } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import Counter from "@/components/bits/Counter";
import { useApi } from "@/lib/api";
import { day } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const usd = (x: number) => `$${x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const sp = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(d)}%`);
const cap = (x: number | null | undefined) => (x == null ? "—" : x >= 1e12 ? `$${(x / 1e12).toFixed(2)}T` : `$${(x / 1e9).toFixed(0)}B`);
const toneCls = (x: number | null | undefined) => (x == null || x === 0 ? "" : x > 0 ? "text-up" : "text-down");

const SORTS = [
  { value: "size", label: "Largest" },
  { value: "today", label: "Best today" },
  { value: "ytd", label: "Best this year" },
  { value: "calm", label: "Steadiest" },
] as const;
type Sort = (typeof SORTS)[number]["value"];

function Avatar({ t }: { t: string }) {
  return (
    <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-foreground/[0.06] font-headline text-[13px] font-semibold tracking-tight text-foreground ring-1 ring-line" aria-hidden>
      {t.slice(0, 4)}
    </span>
  );
}

function Mover({ s, kind }: { s: UsRow; kind: "up" | "down" }) {
  return (
    <Link to={`/us/${s.ticker}`} className="lift group block rounded-2xl">
      <Spot glow={kind === "up" ? "rgba(47,191,138,0.5)" : "rgba(238,94,90,0.5)"}>
        <div className="flex items-center gap-3">
          <Avatar t={s.ticker} />
          <div className="min-w-0">
            <p className="truncate font-medium">{s.name}</p>
            <p className="text-[13px] text-muted-foreground">{s.ticker}</p>
          </div>
        </div>
        <p className={cn("mt-4 font-headline text-[30px] font-semibold leading-none", kind === "up" ? "text-up" : "text-down")}>
          <Counter value={+(s.d1 * 100).toFixed(2)} decimals={2} prefix={s.d1 > 0 ? "+" : "−"} suffix="%" fontSize="1em" format={{ signDisplay: "never" }} />
        </p>
        <p className="mt-1 text-[13px] text-muted-foreground">at {usd(s.close)}</p>
      </Spot>
    </Link>
  );
}

export function UsStocksPage() {
  const reduce = useReducedMotion();
  const theme = useTheme((s) => s.theme);
  const { data, error, isPending } = useApi<UsUniverse>("us");
  const [q, setQ] = useState("");
  const query = useDeferredValue(q);
  const [sector, setSector] = useState("All");
  const [sort, setSort] = useState<Sort>("size");
  const [limit, setLimit] = useState(24);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    document.title = "Large US companies, explained · QuantOS";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const ql = query.trim().toLowerCase();
    const r = data.stocks.filter((s) => (sector === "All" || s.sector === sector) && (!ql || s.ticker.toLowerCase().includes(ql) || s.name.toLowerCase().includes(ql)));
    const by: Record<Sort, (a: UsRow, b: UsRow) => number> = {
      size: (a, b) => (b.market_cap ?? 0) - (a.market_cap ?? 0),
      today: (a, b) => b.d1 - a.d1,
      ytd: (a, b) => (b.ytd ?? -9) - (a.ytd ?? -9),
      calm: (a, b) => (a.vol20 ?? 9) - (b.vol20 ?? 9),
    };
    return [...r].sort(by[sort]);
  }, [data, query, sector, sort]);

  if (isPending) return <Skeleton className="h-[70vh]" />;
  if (error) return <ErrorBox title="The US stock list could not be loaded">{error.message}. Reload the page to try again.</ErrorBox>;
  const byDay = [...data.stocks].sort((a, b) => b.d1 - a.d1);
  const up = byDay.slice(0, 3), down = byDay.slice(-3).reverse();
  const rising = data.stocks.filter((s) => s.d1 > 0).length;

  return (
    <>
      <CineHero
        glow="rgb(76 195 201 / 0.22)"
        background={
          !reduce && (
            <DotGrid dotSize={3} gap={22} baseColor={theme === "dark" ? "#1c2740" : "#d5dbe5"} activeColor="#EFA43A" proximity={130} shockRadius={220} shockStrength={4} glow={0.4} fade={0.35} />
          )
        }
        title={
          <>
            {data.stocks.length} large US companies, explained in plain words
          </>
        }
        lede={`The biggest names in the S&P 500 and the Nasdaq-100. Open any company to see what it does, how much it earns and how its share price has moved. Prices at the close of ${day(data.as_of, true)}.`}
      >
        <label className="relative block max-w-2xl">
          <span className="sr-only">Search companies</span>
          <Search className="pointer-events-none absolute left-5 top-1/2 z-10 size-5 -translate-y-1/2 text-faint" aria-hidden />
          <input
            ref={input}
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setLimit(24);
            }}
            placeholder="Search by name or ticker, like Apple or NVDA"
            className="h-14 w-full rounded-2xl border border-line bg-panel/80 pl-14 pr-14 text-[16px] text-foreground shadow-[0_20px_60px_-30px_rgb(0_0_0/0.6)] backdrop-blur-xl placeholder:text-faint focus:border-saffron/60 focus:outline-none"
          />
          <kbd className="absolute right-4 top-1/2 hidden -translate-y-1/2 rounded-md border border-line px-2 py-0.5 font-mono text-[12px] text-faint sm:block">/</kbd>
        </label>
        <p className="mt-4 text-[14px] text-muted-foreground">
          {rising} of {data.stocks.length} closed higher on {day(data.as_of)}.
        </p>
      </CineHero>

      {!query && (
        <CineSection id="movers" title="Biggest moves on the last day" lede="The three largest rises and falls in the list.">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {up.map((s) => (
              <Mover key={s.ticker} s={s} kind="up" />
            ))}
            {down.map((s) => (
              <Mover key={s.ticker} s={s} kind="down" />
            ))}
          </div>
        </CineSection>
      )}

      <CineSection
        id="list"
        title={query ? `Companies matching “${query}”` : "Every company in the list"}
        lede="Each row shows the share price, the move on the last day and this year, and a three-month price line."
        aside={<Segmented label="Sort by" options={SORTS as unknown as { value: Sort; label: string }[]} value={sort} onChange={setSort} />}
      >
        <div className="-mx-1 mb-5 flex gap-1.5 overflow-x-auto px-1 pb-1 [scrollbar-width:none]" role="group" aria-label="Filter by sector">
          {["All", ...data.sectors].map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={sector === s}
              onClick={() => {
                setSector(s);
                setLimit(24);
              }}
              className={cn(
                "min-h-10 shrink-0 rounded-full border px-4 text-[13.5px] font-medium transition-colors",
                sector === s ? "border-saffron/60 bg-saffron/15 text-foreground" : "border-line text-muted-foreground hover:text-foreground",
              )}
            >
              {s}
            </button>
          ))}
        </div>
        {rows.length === 0 ? (
          <p className="text-muted-foreground">No company in the list matches “{query}”. Try a shorter name or the ticker.</p>
        ) : (
          <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel/40">
            {rows.slice(0, limit).map((s) => (
              <li key={s.ticker}>
                <Link to={`/us/${s.ticker}`} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3.5 transition-colors hover:bg-panel sm:grid-cols-[auto_minmax(0,1fr)_7rem_6rem_5.5rem_5.5rem] sm:px-5">
                  <Avatar t={s.ticker} />
                  <span className="min-w-0">
                    <span className="block truncate font-medium">{s.name}</span>
                    <span className="block truncate text-[13px] text-muted-foreground">
                      {s.ticker} · {s.sector} · {cap(s.market_cap)}
                    </span>
                  </span>
                  <Sparkline data={s.spark} className="hidden sm:block" width={112} height={30} />
                  <span className="num text-right font-medium">{usd(s.close)}</span>
                  <span className={cn("num hidden text-right text-[14px] sm:block", toneCls(s.d1))}>
                    {sp(s.d1, 2)}
                    <span className="block text-[11.5px] text-faint">today</span>
                  </span>
                  <span className={cn("num hidden text-right text-[14px] sm:block", toneCls(s.ytd))}>
                    {sp(s.ytd, 0)}
                    <span className="block text-[11.5px] text-faint">this year</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
        {rows.length > limit && (
          <button type="button" className="btn-ghost mt-6" onClick={() => setLimit((n) => n + 48)}>
            Show {Math.min(48, rows.length - limit)} more companies
          </button>
        )}
        <p className="mt-6 text-[13.5px] text-muted-foreground">Prices from Yahoo Finance, adjusted for splits and dividends. Company facts refresh weekly.</p>
      </CineSection>
    </>
  );
}
