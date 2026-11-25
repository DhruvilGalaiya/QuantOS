import { Moon, Search, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router";
import { CommandPalette } from "@/components/CommandPalette";
import { useRoute } from "@/lib/api";
import { day, level, pct, tone } from "@/lib/format";
import { ALL_NAV, NAV } from "@/lib/nav";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

function MarketStatus() {
  const { data } = useRoute("meta");
  if (!data) return null;
  const live = data.mode !== "static";
  const open = live && data.market_open;
  return (
    <span className="flex items-center gap-2 text-muted-foreground" title={live ? undefined : "Public site: end-of-day data"}>
      <span className={cn("size-1.5 rounded-full", open ? "bg-up" : "bg-faint")} aria-hidden />
      {open ? "NSE open" : live ? "NSE closed" : "End-of-day data"}
    </span>
  );
}

function Ticker() {
  const { data } = useRoute("today");
  if (!data?.indices.length) return null;
  const vix = data.indices.find((i) => i.vol.vix != null)?.vol.vix;
  return (
    <ul className="hidden items-center gap-5 font-mono text-[12.5px] lg:flex" aria-label="Index levels at last close">
      {data.indices.map((i) => (
        <li key={i.symbol} className="flex items-baseline gap-2">
          <span className="text-muted-foreground">{i.name}</span>
          <span className="num">{level(i.market.close)}</span>
          <span className={cn("num", { up: "text-up", down: "text-down", flat: "text-muted-foreground" }[tone(i.market.change_pct)])}>
            {pct(i.market.change_pct, 2, true)}
          </span>
        </li>
      ))}
      {vix != null && (
        <li className="flex items-baseline gap-2">
          <span className="text-muted-foreground">India VIX</span>
          <span className="num">{level(vix * 100, 2)}</span>
        </li>
      )}
    </ul>
  );
}

function Rail({ className }: { className?: string }) {
  return (
    <nav aria-label="Pages" className={className}>
      {NAV.map((g) => (
        <div key={g.group} className="mb-6">
          <p className="mb-1.5 px-3 text-[12px] text-faint">{g.group}</p>
          <ul>
            {g.items.map((it) => (
              <li key={it.to}>
                <NavLink
                  to={it.to}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-2.5 rounded-sm border-l-2 border-transparent px-3 py-1.5 text-[13.5px] text-muted-foreground transition-colors hover:text-foreground",
                      isActive && "border-saffron bg-panel-2 text-foreground",
                    )
                  }
                >
                  <it.icon className="size-4 shrink-0 opacity-80" aria-hidden />
                  <span className="truncate">{it.label}</span>
                  {!it.ready && <span className="ml-auto text-[11px] text-faint">soon</span>}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

/** "g" then a page key jumps to that page (g t = Today, g r = Regimes, ...). */
function useGoShortcuts() {
  const navigate = useNavigate();
  useEffect(() => {
    let armed = 0;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.isContentEditable || /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
      if (e.key === "g") {
        armed = Date.now();
        return;
      }
      if (Date.now() - armed < 1200) {
        const hit = ALL_NAV.find((n) => n.key === e.key);
        if (hit) navigate(hit.to);
        armed = 0;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);
}

export function Shell() {
  const { theme, toggle } = useTheme();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const { data: today } = useRoute("today");
  useGoShortcuts();

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:bg-panel focus:px-3 focus:py-2">
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-line bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
        <div className="flex h-12 items-center gap-6 px-4 md:px-5">
          <NavLink to="/" className="flex items-center gap-2" aria-label="QuantOS home">
            <svg viewBox="0 0 32 32" className="size-5" aria-hidden>
              <path d="M5 22 L12.5 14.5 L18 18.5 L27 8.5" fill="none" stroke="var(--accent-saffron)" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="font-display text-[17px] font-semibold tracking-tight">QuantOS</span>
          </NavLink>
          <Ticker />
          <div className="ml-auto flex items-center gap-4 text-[12.5px]">
            <MarketStatus />
            {today?.as_of && <span className="hidden text-muted-foreground sm:inline">Close of {day(today.as_of)}</span>}
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="flex items-center gap-2 rounded-sm border border-line px-2.5 py-1 text-muted-foreground transition-colors hover:border-line-strong hover:text-foreground"
            >
              <Search className="size-3.5" aria-hidden />
              <span className="hidden sm:inline">Search</span>
              <kbd className="hidden rounded-[3px] bg-panel-2 px-1 font-mono text-[11px] sm:inline">⌘K</kbd>
            </button>
            <button
              type="button"
              onClick={toggle}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              className="rounded-sm p-1.5 text-muted-foreground transition-colors hover:text-foreground"
            >
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>
        {/* phones and tablets: pages as a scrolling strip instead of the rail */}
        <nav aria-label="Pages" className="flex gap-2 overflow-x-auto border-t border-line px-3 py-1 [scrollbar-width:none] md:hidden">
          {ALL_NAV.map((it) => (
            <NavLink
              key={it.to}
              to={it.to}
              className={({ isActive }) =>
                cn("inline-flex min-h-11 shrink-0 items-center rounded-sm px-3 text-[13.5px] text-muted-foreground", isActive && "bg-panel-2 text-foreground")
              }
            >
              {it.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <div className="flex">
        <aside className="sticky top-12 hidden h-[calc(100dvh-3rem)] w-56 shrink-0 overflow-y-auto border-r border-line py-5 md:block">
          <Rail />
        </aside>
        <main id="main" className="min-w-0 flex-1 px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}
