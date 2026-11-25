import { Activity, CalendarRange, Globe2, LayoutDashboard, Layers3, RefreshCw, ShieldAlert, Waves, Sigma, type LucideIcon } from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  /** second key of the "g <key>" shortcut */
  key: string;
  ready: boolean;
  blurb: string;
}

export const NAV: { group: string; items: NavItem[] }[] = [
  {
    group: "Market",
    items: [
      { to: "/today", label: "Today", icon: LayoutDashboard, key: "t", ready: true, blurb: "The market in one screen" },
      { to: "/regimes", label: "Market regimes", icon: Activity, key: "r", ready: true, blurb: "Bull, sideways and bear phases from a hidden Markov model" },
      { to: "/states", label: "Market states", icon: Waves, key: "s", ready: true, blurb: "Calm, normal or stressed, and how long states tend to last" },
      { to: "/events", label: "Event lab", icon: CalendarRange, key: "e", ready: true, blurb: "What followed past signals, with confidence intervals" },
    ],
  },
  {
    group: "Volatility & risk",
    items: [
      { to: "/volatility", label: "Volatility & edge", icon: Sigma, key: "v", ready: true, blurb: "Realized vs implied, vol cone, GARCH and the volatility premium" },
      { to: "/surface", label: "Volatility surface", icon: Layers3, key: "f", ready: true, blurb: "Implied volatility across strikes and expiries" },
      { to: "/risk", label: "Risk monitor", icon: ShieldAlert, key: "k", ready: true, blurb: "Value at risk, tail risk and how tightly sectors move together" },
    ],
  },
  {
    group: "Global",
    items: [{ to: "/us", label: "US stocks", icon: Globe2, key: "u", ready: true, blurb: "Ratios, statements and charts for large US companies" }],
  },
  {
    group: "System",
    items: [{ to: "/status", label: "Data status", icon: RefreshCw, key: "d", ready: true, blurb: "When the data last refreshed and whether every job worked" }],
  },
];

export const ALL_NAV = NAV.flatMap((g) => g.items);
