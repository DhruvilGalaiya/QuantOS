// Small building blocks shared by every page.
import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-3xl">
      <h1 className="font-display text-[28px] font-semibold leading-tight tracking-tight md:text-[32px]">{title}</h1>
      {children && <div className="mt-2 text-[15px] text-muted-foreground">{children}</div>}
    </header>
  );
}

export function Section({ title, lede, children, className }: { title: string; lede?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("mt-12", className)} aria-label={title}>
      <h2 className="font-display text-[19px] font-semibold tracking-tight">{title}</h2>
      {lede && <p className="mt-1 max-w-[72ch] text-muted-foreground">{lede}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** A term with a plain-language definition on hover / focus. */
export function Term({ children, def }: { children: ReactNode; def: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className="cursor-help underline decoration-faint decoration-dotted underline-offset-[3px]">
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72 text-[12.5px] leading-snug">{def}</TooltipContent>
    </Tooltip>
  );
}

export interface StatRow {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "up" | "down" | "flat";
}

/** Label / value rows separated by hairlines; values right-aligned with tabular figures. */
export function StatList({ rows, className }: { rows: StatRow[]; className?: string }) {
  return (
    <dl className={cn("divide-y divide-line", className)}>
      {rows.map((r, i) => (
        <div key={i} className="flex items-baseline justify-between gap-4 py-2">
          <dt className="text-muted-foreground">{r.label}</dt>
          <dd className="text-right">
            <span className={cn("num font-medium", r.tone === "up" && "text-up", r.tone === "down" && "text-down")}>{r.value}</span>
            {r.sub && <span className="block text-[12px] text-faint">{r.sub}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Renders **bold** markers from service text as emphasis. */
export function Rich({ text }: { text: string }) {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 ? (
          <strong key={i} className="num font-semibold text-foreground">
            {p}
          </strong>
        ) : (
          p
        ),
      )}
    </>
  );
}

const TONES = {
  up: "text-up border-up/35",
  down: "text-down border-down/35",
  calm: "text-calm border-calm/35",
  normal: "text-normal border-normal/35",
  stressed: "text-stressed border-stressed/35",
  saffron: "text-saffron border-saffron/40",
  muted: "text-muted-foreground border-line-strong",
} as const;

export function Pill({ tone = "muted", children }: { tone?: keyof typeof TONES; children: ReactNode }) {
  return <span className={cn("inline-flex items-center rounded-[3px] border px-1.5 py-px text-[12px] font-medium", TONES[tone])}>{children}</span>;
}

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("rounded-md border border-line bg-panel", className)}>{children}</div>;
}

/** Wide tables scroll sideways inside their panel instead of the page. */
export function TableFrame({ children }: { children: ReactNode }) {
  return (
    <Panel className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-[13.5px] [&_td]:border-t [&_td]:border-line [&_td]:px-3 [&_td]:py-2.5 [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:text-[12.5px] [&_th]:font-medium [&_th]:text-muted-foreground [&_.r]:text-right">
        {children}
      </table>
    </Panel>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="mt-3 max-w-[80ch] text-[12.5px] leading-relaxed text-faint">{children}</p>;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-panel-2", className)} />;
}

export function ErrorBox({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Panel className="max-w-2xl border-down/40 p-5">
      <p className="font-medium">{title}</p>
      <div className="mt-1 text-muted-foreground">{children}</div>
    </Panel>
  );
}
