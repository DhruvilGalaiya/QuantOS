// Friendly names for the generated OpenAPI schemas. Regenerate api.ts with `npm run gen:types` at the repo root.
import type { components } from "./api.ts";

type S = components["schemas"];

export type Meta = S["Meta"];
export type Freshness = S["Freshness"];
export type Today = S["Today"];
export type IndexBrief = S["IndexBrief"];
export type Market = S["Market"];
export type Regime = S["Regime"];
export type State = S["State"];
export type Vol = S["Vol"];
export type Implied = S["Implied"];
export type SeriesPoint = S["SeriesPoint"];
export type EventHit = S["EventHit"];
export type NewsEvent = S["NewsEvent"];

/** Routes the gateway serves under /api/v1 and their payloads (also the static JSON file names). */
export interface ApiRoutes {
  meta: Meta;
  today: Today;
  system: System;
}
export type RouteName = keyof ApiRoutes;
export type Risk = S["Risk"];

export type Named = S["Named"];
export type Regimes = S["Regimes"];
export type RegimePoint = S["RegimePoint"];
export type Volatility = S["Volatility"];
export type Vrp = S["Vrp"];
export type ConeRow = S["ConeRow"];
export type Garch = S["Garch"];
export type Decay = S["Decay"];
export type UsUniverse = S["UsUniverse"];
export type UsRow = S["UsRow"];
export type UsCompany = S["UsCompany"];
export type Ratio = S["Ratio"];
export type Statement = S["Statement"];
export type States = S["States"];
export type Events = S["Events"];
export type EventStudy = S["EventStudy"];
export type RiskMonitor = S["RiskMonitor"];
export type Surface = S["Surface"];
export type SurfaceExpiry = S["SurfaceExpiry"];
export type Quote = S["Quote"];
export type System = S["System"];
export type JobSummary = S["JobSummary"];
export type JobRun = S["JobRun"];
export type Ticks = S["Ticks"];
export type LiveQuote = S["LiveQuote"];
