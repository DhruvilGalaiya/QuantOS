// The landing page's visual language for terminal pages: a cinematic hero with one live background, large
// Funnel Display headings, spotlight cards and rolling numbers. Shared by Regimes, Volatility and US Stocks.
import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import Counter from "@/components/bits/Counter";
import SpotlightCard from "@/components/bits/SpotlightCard";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

export function useReducedMotion() {
  const [reduce, setReduce] = useState(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduce(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduce;
}

/** Width of an element, kept current with a ResizeObserver (for SVG charts). */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Instrument switcher: pills that are links, so every instrument has its own address. */
export function InstrumentTabs({ items, active, hrefFor, label }: { items: { symbol: string; name: string }[]; active: string; hrefFor: (s: string) => string; label: string }) {
  return (
    <nav aria-label={label} className="-mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1 [scrollbar-width:none]">
      {items.map((it) => (
        <Link
          key={it.symbol}
          to={hrefFor(it.symbol)}
          aria-current={it.symbol === active ? "page" : undefined}
          className={cn(
            "inline-flex min-h-10 shrink-0 items-center rounded-full border px-4 text-[13.5px] font-medium transition-[background-color,border-color,color] duration-200 active:scale-[0.97]",
            it.symbol === active
              ? "border-saffron/60 bg-saffron/15 text-foreground"
              : "border-line bg-panel/50 text-muted-foreground backdrop-blur hover:border-line-strong hover:text-foreground",
          )}
        >
          {it.name}
        </Link>
      ))}
    </nav>
  );
}

/**
 * Page hero: a live background (one React Bits effect per page), a glow in the page's colour, a sentence
 * headline built from today's data and up to four rolling figures.
 */
export function CineHero({
  background,
  glow = "rgb(239 164 58 / 0.32)",
  top,
  title,
  lede,
  children,
}: {
  background?: ReactNode;
  glow?: string;
  top?: ReactNode;
  title: ReactNode;
  lede?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <section
      className="relative isolate -mx-4 -mt-6 overflow-hidden px-4 pb-14 pt-10 md:-mx-8 md:-mt-8 md:px-8 md:pb-20 md:pt-14"
      style={{ "--hero-glow": glow } as CSSProperties}
      data-enter
    >
      <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden>
        {background && <div className="absolute inset-0 opacity-70 dark:opacity-90">{background}</div>}
        <div className="absolute inset-0 bg-[radial-gradient(70%_60%_at_15%_0%,var(--hero-glow),transparent_70%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(90deg,var(--background)_0%,color-mix(in_srgb,var(--background)_82%,transparent)_45%,color-mix(in_srgb,var(--background)_30%,transparent)_100%)]" />
        <div className="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-b from-transparent to-background" />
      </div>
      {top && <div className="mb-8">{top}</div>}
      <h1 className="headline max-w-[18ch] text-[clamp(2.3rem,1.3rem+3.6vw,4.4rem)]">{title}</h1>
      {lede && <p className="mt-5 max-w-[60ch] text-[16.5px] leading-relaxed text-muted-foreground">{lede}</p>}
      {children && <div className="mt-10">{children}</div>}
    </section>
  );
}

/** Big figure with a rolling counter. `value` is already in display units (e.g. 14.97 for 14.97%). */
export function BigStat({
  label,
  value,
  decimals = 0,
  prefix,
  suffix,
  sub,
  tone,
  text,
}: {
  label: ReactNode;
  value?: number | null;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  sub?: ReactNode;
  tone?: "up" | "down" | "saffron";
  text?: string;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[13.5px] text-muted-foreground">{label}</dt>
      <dd
        className={cn(
          "mt-1.5 font-headline text-[clamp(1.9rem,1.4rem+1.4vw,2.7rem)] font-semibold leading-none tracking-tight",
          tone === "up" && "text-up",
          tone === "down" && "text-down",
          tone === "saffron" && "text-saffron",
        )}
      >
        {text ?? (value == null || !Number.isFinite(value) ? "—" : <Counter value={value} decimals={decimals} locale="en-IN" prefix={prefix} suffix={suffix} fontSize="1em" />)}
      </dd>
      {sub && <dd className="mt-2 text-[13px] leading-snug text-faint">{sub}</dd>}
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return <dl className="grid grid-cols-2 gap-x-6 gap-y-8 lg:grid-cols-4">{children}</dl>;
}

export function CineSection({ id, title, lede, aside, children, className }: { id?: string; title: ReactNode; lede?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section id={id} aria-labelledby={id ? `${id}-h` : undefined} className={cn("mt-20 scroll-mt-20 md:mt-28", className)}>
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_auto]">
        <div>
          <h2 id={id ? `${id}-h` : undefined} className="h-section max-w-[22ch] !text-[clamp(1.9rem,1.2rem+2.2vw,3rem)]">
            {title}
          </h2>
          {lede && <p className="mt-4 max-w-[62ch] text-[16px] leading-relaxed text-muted-foreground">{lede}</p>}
        </div>
        {aside}
      </div>
      <div className="mt-10">{children}</div>
    </section>
  );
}

/** A spotlight card that follows the theme. */
export function Spot({ children, className, glow = "rgba(239, 164, 58, 0.5)" }: { children: ReactNode; className?: string; glow?: string }) {
  const theme = useTheme((s) => s.theme);
  return (
    <SpotlightCard
      theme={theme}
      className={cn("rounded-2xl! p-5! sm:p-6!", className)}
      spotlightColor={glow}
      intensity={theme === "dark" ? 0.1 : 0.06}
      borderGlow={0.7}
      style={theme === "dark" ? ({ "--spotlight-card-surface": "var(--panel)", "--spotlight-card-border": "rgba(255,255,255,0.08)" } as CSSProperties) : undefined}
    >
      {children}
    </SpotlightCard>
  );
}

/** Segmented control (periods, views). */
export function Segmented<T extends string>({ options, value, onChange, label }: { options: readonly T[] | { value: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  const opts = (options as (T | { value: T; label: string })[]).map((o) => (typeof o === "string" ? { value: o, label: o } : o));
  return (
    <div role="group" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-full border border-line bg-panel/60 p-1 backdrop-blur">
      {opts.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "min-h-9 rounded-full px-3.5 text-[13px] font-medium transition-colors pointer-coarse:min-h-11",
            o.value === value ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Plain-language callout: a short reading of the numbers in one or two sentences. */
export function Reading({ children, title = "In plain words" }: { children: ReactNode; title?: string }) {
  return (
    <div className="mt-6 max-w-[74ch] rounded-2xl bg-saffron/[0.07] px-5 py-4 ring-1 ring-saffron/15">
      <p className="text-[13px] font-medium text-saffron">{title}</p>
      <p className="mt-1.5 text-[16px] leading-relaxed text-muted-foreground [&_b]:font-semibold [&_b]:text-foreground">{children}</p>
    </div>
  );
}

/** "How it is calculated" disclosure for method notes, closed by default. */
export function Method({ children }: { children: ReactNode }) {
  return (
    <details className="group mt-6 max-w-[80ch] text-[14px] text-muted-foreground">
      <summary className="inline-flex min-h-11 cursor-pointer list-none items-center gap-2 font-medium text-foreground/90 hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="inline-block transition-transform duration-200 group-open:rotate-90" aria-hidden>
          ›
        </span>
        How it is calculated
      </summary>
      <div className="mt-2 space-y-2 leading-relaxed">{children}</div>
    </details>
  );
}

/** Keep names like "NIFTY 50" on one line inside headlines. */
export const nb = (s: string) => s.replace(/ /g, "\u00A0");
