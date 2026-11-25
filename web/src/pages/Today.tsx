import type { IndexBrief, NewsEvent, Today as TodayData } from "@quantos/types";
import { Link } from "react-router";
import { PriceCone } from "@/components/charts/PriceCone";
import { ErrorBox, Note, PageHeader, Panel, Pill, Rich, Section, Skeleton, StatList, type StatRow, TableFrame, Term } from "@/components/kit";
import { DATA_SOURCE, useRoute } from "@/lib/api";
import { DASH, day, level, pct, pts, rupees, share, stamp, tone } from "@/lib/format";
import { ALL_NAV } from "@/lib/nav";
import { cn } from "@/lib/utils";

const REGIME_TONE = { BULL: "up", SIDEWAYS: "normal", BEAR: "down" } as const;
const STATE_TONE = { CALM: "calm", NORMAL: "normal", STRESSED: "stressed" } as const;

function indexRows(i: IndexBrief): StatRow[] {
  const { market: m, vol: v, implied: imp } = i;
  const rows: StatRow[] = [
    {
      label: <Term def="How far the index is below its highest close ever.">From all-time high</Term>,
      value: m.drawdown < 0 ? pct(m.drawdown, 1, true) : "At the high",
      tone: m.drawdown < -0.1 ? "down" : undefined,
      sub: `High ${level(m.ath, 0)}`,
    },
    {
      label: <Term def="Where the last close sits between the lowest (0%) and highest (100%) close of the past year.">52-week range position</Term>,
      value: share(m.range52_pos),
      sub: `${level(m.lo52, 0)} – ${level(m.hi52, 0)}`,
    },
    { label: "1 week / 1 month", value: `${pct(m.week_pct, 1, true)} / ${pct(m.month_pct, 1, true)}` },
    {
      label: <Term def="How much the index actually moved over the last 21 sessions, as a yearly percentage (Yang-Zhang estimator).">Realized volatility, 21 sessions</Term>,
      value: pct(v.rv21),
      sub: v.rv21_level ? `${v.rv21_level}, ${v.rv21_ordinal}` : undefined,
    },
  ];
  if (v.vix != null) {
    rows.push(
      {
        label: <Term def="The yearly move NIFTY option prices imply over the next 30 days, published by NSE.">India VIX</Term>,
        value: level(v.vix * 100, 1),
        sub: v.vix_ordinal ? `${v.vix_ordinal} of the past year` : undefined,
      },
      {
        label: <Term def="VIX minus realized volatility. Positive means option buyers pay more than recent movement justifies: the seller's edge.">Volatility premium</Term>,
        value: pts(v.vrp_now),
        tone: tone(v.vrp_now),
        sub: v.vrp_ordinal ? `${v.vrp_ordinal} of history` : undefined,
      },
    );
  } else if (imp) {
    rows.push(
      { label: "Front at-the-money implied vol", value: pct(imp.atm_iv), sub: `expiry ${day(imp.expiry)}` },
      {
        label: <Term def="Option-implied volatility minus the GARCH model's 21-session volatility estimate.">Implied − GARCH</Term>,
        value: pts(v.iv_vs_garch),
        sub: `GARCH 21-session ${pct(v.garch_21d)}`,
      },
    );
  }
  return rows;
}

export function IndexPanel({ i, heading: H = "h2" }: { i: IndexBrief; heading?: "h2" | "h3" }) {
  const m = i.market;
  const t = tone(m.change_pct);
  return (
    <Panel className="min-w-0 p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <H className="text-[15px] font-medium text-muted-foreground">{i.name}</H>
        <div className="flex flex-wrap gap-2">
          {i.regime && (
            <Pill tone={REGIME_TONE[i.regime.regime as keyof typeof REGIME_TONE] ?? "muted"}>
              {i.regime.label} regime{i.regime.streak ? `, ${i.regime.streak} sessions` : ""}
            </Pill>
          )}
          {i.state && (
            <Pill tone={STATE_TONE[i.state.state as keyof typeof STATE_TONE] ?? "muted"}>
              {i.state.label}{i.state.streak ? ` for ${i.state.streak} sessions` : ""}
            </Pill>
          )}
        </div>
      </div>
      <p className="mt-1 flex items-baseline gap-3">
        <span className="num font-display text-[38px] font-semibold leading-none tracking-tight">{level(m.close)}</span>
        <span className={cn("num text-[15px] font-medium", t === "up" && "text-up", t === "down" && "text-down")}>{pct(m.change_pct, 2, true)}</span>
      </p>

      <div className="mt-5">
        <PriceCone history={i.history} implied={i.implied} label={i.name} />
        {i.implied && (
          <p className="mt-2 text-[12.5px] text-muted-foreground">
            <span className="mr-1.5 inline-block w-4 border-t border-dashed border-saffron align-middle" aria-hidden />
            Range option prices imply to {day(i.implied.expiry)}:{" "}
            <span className="num text-foreground">
              {level(i.implied.lower, 0)} – {level(i.implied.upper, 0)}
            </span>{" "}
            (±{level(i.implied.move_points, 0)} pts). By option pricing the index ends inside it about 2 times in 3.
          </p>
        )}
      </div>

      <StatList rows={indexRows(i)} className="mt-4 border-t border-line" />
    </Panel>
  );
}

function ExpectedMoves({ indices, }: { indices: IndexBrief[] }) {
  const rows = indices.filter((i) => i.implied);
  if (!rows.length) return null;
  const snap = rows[0].implied?.snapshot_ts;
  return (
    <Section
      title="How far the market is priced to move by expiry"
      lede="The price of the at-the-money call and put shows how much movement traders are paying to insure against. Option sellers profit if the index stays inside the range; buyers need a bigger move."
    >
      <TableFrame>
        <thead>
          <tr>
            <th>Index</th>
            <th className="r">Level</th>
            <th className="r">Range by expiry</th>
            <th className="r">Move</th>
            <th className="r">Expiry</th>
            <th className="r">Strike used</th>
            <th className="r">Call + put</th>
            <th className="r">Implied vol</th>
          </tr>
        </thead>
        <tbody className="num">
          {rows.map(({ symbol, name, implied: imp }) =>
            imp ? (
              <tr key={symbol}>
                <td className="font-medium">{name}</td>
                <td className="r">{level(imp.spot, 0)}</td>
                <td className="r font-medium">
                  {level(imp.lower, 0)} – {level(imp.upper, 0)}
                </td>
                <td className="r">
                  ±{level(imp.move_points, 0)} <span className="text-faint">(±{pct(imp.move_pct, 2)})</span>
                </td>
                <td className="r">{day(imp.expiry)}</td>
                <td className="r">{level(imp.atm_strike, 0)}</td>
                <td className="r">
                  {imp.straddle != null ? (
                    <>
                      {rupees(imp.atm_call)} + {rupees(imp.atm_put)} = <span className="font-medium">{rupees(imp.straddle)}</span>
                    </>
                  ) : (
                    DASH
                  )}
                </td>
                <td className="r">{pct(imp.atm_iv)}</td>
              </tr>
            ) : null,
          )}
        </tbody>
      </TableFrame>
      <Note>
        The call and put at the strike closest to the index on the nearest expiry. Their implied volatility (the yearly move they price in) is scaled to the time
        left: range = level ± level × implied vol × √(years to expiry). Prices are valued at that implied volatility, in index points (₹ per unit).
        {snap && ` Option prices as of ${stamp(snap)}.`}
      </Note>
    </Section>
  );
}

function Signals({ events }: { events: TodayData["events"] }) {
  return (
    <div>
      <h3 className="mb-3 font-medium">Price signals in the last five sessions</h3>
      {events.length ? (
        <>
          <TableFrame>
            <thead>
              <tr>
                <th>Signal</th>
                <th>Index</th>
                <th>Date</th>
                <th className="r">
                  <Term def="How many times this signal fired before, with a full month of data after it.">Past cases</Term>
                </th>
                <th className="r">Avg next month</th>
                <th className="r">
                  <Term def="The average one-month return over all days, for comparison.">Any day</Term>
                </th>
                <th className="r">
                  <Term def="Share of past cases where the index was higher a month later.">Up share</Term>
                </th>
                <th>History says</th>
              </tr>
            </thead>
            <tbody className="num">
              {events.map((e) => (
                <tr key={`${e.index}-${e.id}`}>
                  <td className="font-medium">{e.name}</td>
                  <td>{e.index}</td>
                  <td>{day(e.date)}</td>
                  <td className="r">{e.n ?? DASH}</td>
                  <td className={cn("r", { up: "text-up", down: "text-down", flat: "" }[tone(e.mean)])}>{pct(e.mean, 2, true)}</td>
                  <td className="r text-muted-foreground">{pct(e.baseline_mean, 2, true)}</td>
                  <td className="r">{share(e.hit_rate)}</td>
                  <td>{e.unusual ? <Pill tone="saffron">Unusual</Pill> : <span className="text-muted-foreground">Like a normal day</span>}</td>
                </tr>
              ))}
            </tbody>
          </TableFrame>
          <Note>“Unusual” means the average after the signal differs from an ordinary day with a p-value under 0.05 and at least 10 past cases. Full statistics and paths will be on the Event lab page.</Note>
        </>
      ) : (
        <p className="text-muted-foreground">No tracked price signals fired in the last five sessions.</p>
      )}
    </div>
  );
}

function News({ news }: { news: TodayData["news"] }) {
  const byDay = news.events.reduce<Record<string, NewsEvent[]>>((acc, e) => {
    (acc[e.date] ??= []).push(e);
    return acc;
  }, {});
  return (
    <div>
      <h3 className="mb-3 font-medium">Scheduled news, next 7 days</h3>
      {news.events.length ? (
        <Panel className="divide-y divide-line">
          {Object.entries(byDay).map(([d, evs]) => (
            <div key={d} className="grid gap-x-6 gap-y-1 px-4 py-3 sm:grid-cols-[8rem_1fr]">
              <p className="text-muted-foreground">{day(d)}</p>
              <ul className="space-y-3">
                {evs.map((e, k) => (
                  <li key={k}>
                    <p className="flex flex-wrap items-baseline gap-x-2">
                      <span className="font-medium">{e.title}</span>
                      <span className="text-[12.5px] text-muted-foreground">
                        {e.country}
                        {e.time ? `, ${e.time} IST` : ""}
                      </span>
                      <Pill tone={e.impact === "High" ? "down" : e.impact === "Medium" ? "normal" : "muted"}>{e.impact} impact</Pill>
                    </p>
                    <p className="mt-0.5 max-w-[72ch] text-[13px] text-muted-foreground">{e.why}</p>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </Panel>
      ) : (
        <p className="text-muted-foreground">No major scheduled events in the next 7 days.</p>
      )}
      {news.built_at && (
        <Note>
          Sources: Forex Factory (global releases), RBI and Federal Reserve meeting calendars, MoSPI release dates, the NSE corporate calendar (NIFTY 50 results) and the
          option chain (expiries). Updated {stamp(news.built_at)}.
        </Note>
      )}
    </div>
  );
}

function Loading() {
  return (
    <div aria-busy="true" aria-label="Loading today's brief">
      <Skeleton className="mb-8 h-14 max-w-xl" />
      <div className="grid gap-5 xl:grid-cols-2">
        <Skeleton className="h-[560px]" />
        <Skeleton className="h-[560px]" />
      </div>
    </div>
  );
}

export function TodayPage() {
  const { data, error, isPending, refetch } = useRoute("today");
  if (isPending) return <Loading />;
  if (error)
    return (
      <ErrorBox title="Today's brief could not be loaded">
        {DATA_SOURCE === "api" ? (
          <>
            {error.message}. Check that the gateway (<code>npm run dev:gateway</code>) and the quant service (<code>npm run dev:api</code>) are running, then{" "}
            <button className="text-saffron underline" onClick={() => refetch()}>
              try again
            </button>
            .
          </>
        ) : (
          <>{error.message}. The exported data file is missing; rebuild the site.</>
        )}
      </ErrorBox>
    );
  if (!data.indices.length)
    return (
      <ErrorBox title="No brief yet">
        The analytics job has not run. Build it with <code>python -m quantos_platform.jobs analytics</code>.
      </ErrorBox>
    );

  const deeper = ALL_NAV.filter((n) => n.to !== "/today");
  return (
    <>
      <PageHeader title="Today">
        Indian index markets at the close of {day(data.as_of, true)}: where prices sit, how much they move, and what option sellers are being paid. Measurements and base rates, not forecasts.
      </PageHeader>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {data.indices.map((i) => (
          <IndexPanel key={i.symbol} i={i} />
        ))}
      </div>

      <ExpectedMoves indices={data.indices} />

      {data.readings.length > 0 && (
        <Section title="Reading the numbers">
          <div className="max-w-[72ch] space-y-3 text-[15px] leading-relaxed text-muted-foreground">
            {data.readings.map((r, k) => (
              <p key={k}>
                <Rich text={r} />
              </p>
            ))}
          </div>
        </Section>
      )}

      <Section title="This week" lede="Signals that fired recently, with what followed them historically over the next month, and the scheduled news coming up.">
        <div className="space-y-8">
          <Signals events={data.events} />
          <News news={data.news} />
        </div>
      </Section>

      <Section title="Go deeper">
        <ul className="grid gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {deeper.map((n) => (
            <li key={n.to} className="bg-panel">
              <Link to={n.to} className="block h-full p-4 transition-colors hover:bg-panel-2">
                <span className="flex items-center gap-2 font-medium">
                  <n.icon className="size-4 text-muted-foreground" aria-hidden />
                  {n.label}
                </span>
                <span className="mt-1 block text-[13px] text-muted-foreground">{n.blurb}</span>
              </Link>
            </li>
          ))}
        </ul>
      </Section>
    </>
  );
}
