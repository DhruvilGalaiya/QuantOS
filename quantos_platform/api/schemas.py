"""Pydantic response models for the internal quant API. TypeScript types are generated from these (OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ------------------------------------------------------------------ meta
class Freshness(Model):
    dataset: str
    label: str
    as_of: str
    updated: str


class Meta(Model):
    version: str
    mode: str
    market_open: bool
    now: str
    freshness: list[Freshness]


# ------------------------------------------------------------------ today
class SeriesPoint(Model):
    time: str
    value: float


class Market(Model):
    as_of: str
    close: float
    change_pct: float
    week_pct: float
    month_pct: float
    drawdown: float
    ath: float
    hi52: float
    lo52: float
    range52_pos: float | None = None


class Regime(Model):
    regime: str
    label: str
    streak: int | None = None
    stickiness: str | None = None
    as_of: str | None = None


class State(Model):
    state: str
    label: str
    streak: int | None = None
    since: str | None = None
    still_after_5: float | None = None
    still_after_21: float | None = None


class Vol(Model):
    rv21: float | None = None
    rv21_percentile: float | None = None
    rv21_level: str | None = None
    rv21_ordinal: str | None = None
    garch_21d: float | None = None
    garch_long_run: float | None = None
    hurst: float | None = None
    vix: float | None = None
    vix_percentile_1y: float | None = None
    vix_level: str | None = None
    vix_ordinal: str | None = None
    vrp_now: float | None = None
    vrp_percentile: float | None = None
    vrp_ordinal: str | None = None
    vrp_share_positive: float | None = None
    vrp_vs_garch: float | None = None
    iv_vs_garch: float | None = None


class Implied(Model):
    expiry: str
    days: float | None = None
    atm_iv: float
    move_points: float
    move_pct: float
    snapshot_ts: str | None = None
    spot: float
    forward: float | None = None
    lower: float
    upper: float
    rr25: float | None = None
    bf25: float | None = None
    atm_strike: float | None = None
    atm_call: float | None = None
    atm_put: float | None = None
    straddle: float | None = None


class IndexBrief(Model):
    symbol: str
    name: str
    market: Market
    regime: Regime | None = None
    state: State | None = None
    vol: Vol
    implied: Implied | None = None
    history: list[SeriesPoint]


class EventHit(Model):
    id: str
    name: str
    index: str
    date: str
    n: int | None = None
    mean: float | None = None
    baseline_mean: float | None = None
    hit_rate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    unusual: bool


class NewsEvent(Model):
    date: str
    time: str | None = None
    title: str
    country: str
    impact: str
    why: str
    kind: str | None = None
    source: str | None = None


class News(Model):
    built_at: str | None = None
    start: str | None = None
    end: str | None = None
    events: list[NewsEvent]


class Risk(Model):
    var95: float
    cvar95: float
    sigma_next: float | None = None
    mc_paths: int
    mc_days: int
    mc_p05: float | None = None
    mc_p50: float | None = None
    mc_p95: float | None = None
    mc_hist: list[int]
    mc_edges: list[float]
    turbulence_pct: float | None = None
    avg_corr_pct: float | None = None


class Today(Model):
    as_of: str | None = None
    built: str | None = None
    indices: list[IndexBrief]
    readings: list[str]
    risk: Risk | None = None
    events: list[EventHit]
    news: News
    freshness: list[Freshness]


# ------------------------------------------------------------------ market regimes
class Named(Model):
    symbol: str
    name: str


class RegimeNow(Model):
    regime: str | None = None
    label: str | None = None
    streak: int | None = None
    persistence: float | None = None
    persistence_level: str | None = None
    stay_probability: float | None = None
    consistency: float | None = None
    close: float | None = None
    change_pct: float | None = None


class Forecast(Model):
    model: str
    target_date: str | None = None
    status: str | None = None
    pick: str | None = None
    confidence: float | None = None
    probs: dict[str, float]


class Scorecard(Model):
    model: str
    start: str | None = None
    end: str | None = None
    n: int
    direction_accuracy: float | None = None
    three_class_accuracy: float | None = None
    always_up: float | None = None
    walk_forward: bool


class Dynamics(Model):
    regime: str
    label: str
    stay: float
    lasts: float
    exits_to: str | None = None
    exits_share: float | None = None
    long_run: float | None = None


class RegimePoint(Model):
    time: str
    close: float
    regime: str | None = None


class OverviewRow(Model):
    symbol: str
    name: str
    close: float | None = None
    change_pct: float | None = None
    regime: str | None = None
    streak: int | None = None
    persistence: float | None = None
    next: str | None = None
    next_model: str | None = None


class LogRow(Model):
    date: str
    target: str
    regime_then: str | None = None
    forecast: str | None = None
    confidence: float | None = None
    move: float | None = None
    hit: bool | None = None


class PredictionLog(Model):
    model: str
    rows: list[LogRow]


class Regimes(Model):
    symbol: str
    name: str
    is_vix: bool
    states: list[str]
    labels: dict[str, str]
    instruments: list[Named]
    as_of: str | None = None
    current: RegimeNow | None = None
    forecast: Forecast | None = None
    scorecard: Scorecard | None = None
    matrix: list[list[float]] | None = None
    dynamics: list[Dynamics]
    timeline: list[RegimePoint]
    overview: list[OverviewRow]
    log: PredictionLog


# ------------------------------------------------------------------ volatility & edge
class Realized(Model):
    yang_zhang: float | None = None
    percentile: float | None = None
    ordinal: str | None = None
    estimators: dict[str, float]


class ImpliedNow(Model):
    atm_iv: float
    expiry: str | None = None


class VrpPoint(Model):
    time: str
    vix: float | None = None
    rv: float | None = None
    gap_avg: float | None = None


class VixBucket(Model):
    bucket: str
    vix_range: list[float]
    n: int | None = None
    share_implied_above: float
    mean_spread: float
    median_spread: float | None = None


class VrpYear(Model):
    year: int
    mean_spread: float
    share_implied_above: float | None = None
    mean_vix: float | None = None
    mean_rv: float | None = None


class Vrp(Model):
    since: str | None = None
    n_complete: int | None = None
    share_implied_above: float | None = None
    mean_spread: float | None = None
    median_spread: float | None = None
    worst_spread: float | None = None
    worst_date: str | None = None
    vix_now: float | None = None
    vix_percentile_1y: float | None = None
    premium_now: float | None = None
    premium_percentile: float | None = None
    premium_ordinal: str | None = None
    by_vix_level: list[VixBucket]
    yearly: list[VrpYear]
    series: list[VrpPoint]


class ConeRow(Model):
    horizon: int
    current: float
    percentile: float
    min: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: float


class TermPoint(Model):
    days: int
    vol: float


class GarchPoint(Model):
    time: str
    garch: float | None = None
    realized: float | None = None


class Garch(Model):
    alpha: float | None = None
    beta: float | None = None
    nu: float | None = None
    persistence: float | None = None
    half_life_days: float | None = None
    long_run_vol: float | None = None
    current_vol: float | None = None
    vol_21d: float | None = None
    fit_start: str | None = None
    n_obs: int | None = None
    term_structure: list[TermPoint] = []
    history: list[GarchPoint] = []


class DecayRow(Model):
    tdte: int
    nights: int
    win_rate: float
    avg_rupees: float
    median_per100: float
    worst_rupees: float
    verdict: str
    hist: list[int]


class Decay(Model):
    lot: int
    n: int | None = None
    start: str | None = None
    end: str | None = None
    edges: list[int]
    rows: list[DecayRow]


class HurstRoll(Model):
    end: str
    H: float


class Hurst(Model):
    H: float | None = None
    zeta: dict[str, float]
    rolling: list[HurstRoll]
    proxy: str | None = None


class Volatility(Model):
    symbol: str
    name: str
    indices: list[Named]
    as_of: str | None = None
    close: float | None = None
    realized: Realized | None = None
    implied: ImpliedNow | None = None
    vrp: Vrp | None = None
    cone: list[ConeRow] = []
    garch: Garch | None = None
    decay: Decay | None = None
    hurst: Hurst | None = None


# ------------------------------------------------------------------ US stocks
class UsRow(Model):
    ticker: str
    name: str
    sector: str
    market_cap: float | None = None
    sp500_rank: float | None = None
    nasdaq_rank: float | None = None
    close: float
    d1: float
    m1: float | None = None
    ytd: float | None = None
    y1: float | None = None
    vol20: float | None = None
    spark: list[float]


class UsUniverse(Model):
    as_of: str | None = None
    sectors: list[str]
    stocks: list[UsRow]


class Ratio(Model):
    key: str
    label: str
    value: float
    kind: str
    plain: str
    group: str


class Fact(Model):
    title: str
    text: str


class Summary(Model):
    lead: str | None = None
    rest: str | None = None


class StatementRow(Model):
    label: str
    kind: str
    bold: bool = False
    values: list[float | None]


class Statement(Model):
    columns: list[str] = []
    rows: list[StatementRow] = []


class Financials(Model):
    quarterly: Statement | None = None
    annual: Statement | None = None
    balance: Statement | None = None
    cashflow: Statement | None = None
    ratios: Statement | None = None


class Returns(Model):
    m1: float | None = None
    m6: float | None = None
    ytd: float | None = None
    y1: float | None = None
    y5: float | None = None


class UsCompany(Model):
    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    website: str | None = None
    currency: str
    as_of: str
    fundamentals_as_of: str | None = None
    price: float
    change_pct: float
    returns: Returns
    ratios: list[Ratio]
    facts: list[Fact]
    summary: Summary
    financials: Financials
    history: list[SeriesPoint]


# ------------------------------------------------------------------ market states
class StateNow(Model):
    state: str
    since: str | None = None
    streak: int | None = None
    probs: dict[str, float]
    still_after_5: float | None = None
    still_after_21: float | None = None


class StateBehaviour(Model):
    state: str
    share: float
    typical_day: float
    vol_annual: float
    big_down_share: float
    next_day_up: float | None = None
    fwd21_mean: float
    fwd21_p10: float
    fwd21_p90: float
    fwd21_up: float | None = None
    worst_day: float | None = None


class StateTransition(Model):
    p: list[list[float]]
    lo: list[list[float]]
    hi: list[list[float]]
    duration: list[float]
    dur_lo: list[float]
    dur_hi: list[float]


class StateEpisodes(Model):
    state: str
    n: int
    median_days: float | None = None
    longest: int | None = None
    next: dict[str, float | None]


class StatePoint(Model):
    time: str
    close: float
    state: str


class BaseRate(Model):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    from_: float = Field(alias="from")
    to: float
    n: int
    went_on: int
    share: float | None = None
    median_recover_days: float | None = None


class Fall(Model):
    peak: str
    trough: str
    depth: float
    days_down: int
    days_to_recover: float | None = None
    ongoing: bool


class Drawdown(Model):
    now: float
    stage: str
    peak: float
    base_rates: list[BaseRate]
    falls: list[Fall]
    series: list[SeriesPoint]


class States(Model):
    symbol: str
    name: str
    symbols: list[Named]
    labels: dict[str, str]
    states: list[str]
    as_of: str | None = None
    switch_prob: float | None = None
    current: StateNow | None = None
    behaviour: list[StateBehaviour] = []
    transition: StateTransition | None = None
    episodes: list[StateEpisodes] = []
    survival: dict[str, list[float]] = {}
    timeline: list[StatePoint] = []
    drawdown: Drawdown | None = None


# ------------------------------------------------------------------ event lab
class EventStat(Model):
    n: int | None = None
    mean: float | None = None
    median: float | None = None
    hit_rate: float | None = None
    worst: float | None = None
    best: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    baseline_mean: float | None = None
    baseline_hit_rate: float | None = None
    excess: float | None = None
    p_value: float | None = None
    unusual: bool = False


class EventPath(Model):
    offsets: list[int]
    mean: list[float | None]
    ci_low: list[float | None]
    ci_high: list[float | None]
    baseline: list[float | None]


class Occurrence(Model):
    date: str
    close: float | None = None
    h1: float | None = None
    h5: float | None = None
    h21: float | None = None
    h63: float | None = None


class EventStudy(Model):
    id: str
    name: str
    description: str
    group: str
    min_gap: int
    count: int
    last: str | None = None
    stats: dict[str, EventStat]
    path: EventPath
    occurrences: list[Occurrence]


class RecentEvent(Model):
    id: str
    name: str
    date: str
    today: bool | None = None


class Events(Model):
    symbol: str
    name: str
    symbols: list[Named]
    as_of: str | None = None
    since: str | None = None
    sessions: int | None = None
    horizons: list[int] = []
    n_boot: int | None = None
    recent: list[RecentEvent] = []
    events: list[EventStudy] = []
    alpha: float = 0.05
    min_n: int = 10


# ------------------------------------------------------------------ risk monitor
class RiskNow(Model):
    avg_corr: float
    avg_corr_pct: float
    absorption: float
    absorption_pct: float
    turbulence: float
    turbulence_pct: float
    turbulence_p95: float
    turbulent_days_21: int
    dispersion: float
    dispersion_pct: float
    avg_corr_level: str | None = None
    avg_corr_ordinal: str | None = None
    turbulence_ordinal: str | None = None
    dispersion_ordinal: str | None = None
    absorption_level: str | None = None


class TurbulenceBase(Model):
    n_turbulent: int
    n_all: int
    fwd_vol_turbulent: float
    fwd_vol_all: float
    fwd_ret_turbulent: float
    fwd_ret_all: float


class CorrMatrix(Model):
    names: list[str]
    matrix: list[list[float]]


class SectorRow(Model):
    symbol: str
    name: str
    w1: float | None = None
    m1: float
    m3: float | None = None
    m6: float | None = None
    y1: float | None = None
    range52: float | None = None
    off_high: float | None = None
    vol21: float | None = None


class VarLevel(Model):
    var: float
    cvar: float


class Backtest(Model):
    n: int
    start: str
    rate95: float
    rate99: float


class WorstDay(Model):
    date: str
    ret: float


class MonteCarlo(Model):
    p01: float | None = None
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    down5: float | None = None
    down10: float | None = None
    hist: list[int]
    edges: list[float]


class Tail(Model):
    symbol: str
    name: str
    close: float | None = None
    fhs: dict[str, VarLevel]
    hist: dict[str, VarLevel]
    backtest: Backtest
    hill_alpha: float | None = None
    worst_days: list[WorstDay]
    mc: MonteCarlo


class RiskPoint(Model):
    time: str
    avg_corr: float | None = None
    absorption: float | None = None
    turbulence: float | None = None
    p95: float | None = None


class Crash(Model):
    peak: str
    trough: str
    depth: float
    days_down: int
    dates: list[str]
    returns: dict[str, list[float]]


class RiskMonitor(Model):
    as_of: str | None = None
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    indices: dict[str, str] = {}
    now: RiskNow | None = None
    turbulence_base: TurbulenceBase | None = None
    corr_matrix: CorrMatrix | None = None
    scoreboard: list[SectorRow] = []
    tail: dict[str, Tail] = {}
    series: list[RiskPoint] = []
    crashes: list[Crash] = []


# ------------------------------------------------------------------ volatility surface
class SurfaceRef(Model):
    expiry: str
    days: float
    forward: float | None = None
    atm_iv: float | None = None
    rr25: float | None = None
    bf25: float | None = None


class SurfaceExpiry(Model):
    expiry: str
    days: float
    T: float
    forward: float | None = None
    n_points: int
    atm_iv: float | None = None
    rr25: float | None = None
    bf25: float | None = None
    svi_rmse: float | None = None
    sabr_rmse: float | None = None
    butterfly_ok: bool
    calendar_ok: bool | None = None
    sabr_rho: float | None = None
    sabr_nu: float | None = None
    svi_rho: float | None = None
    k_min: float | None = None
    k_max: float | None = None


class Quote(Model):
    expiry: str
    strike: float
    k: float | None = None
    iv: float
    iv_bid: float | None = None
    iv_ask: float | None = None
    svi: float | None = None
    sabr: float | None = None
    type: str | None = None
    used: bool
    oi: float | None = None


class SurfaceGrid(Model):
    k: list[float | None]
    moneyness: list[float | None]
    days: list[float | None]
    svi: list[list[float | None]]
    sabr: list[list[float | None]]


class Surface(Model):
    symbol: str
    name: str
    symbols: list[Named]
    snapshot_ts: str | None = None
    spot: float | None = None
    realized_20d: float | None = None
    ref: SurfaceRef | None = None
    arb_warnings: int = 0
    quotes: int = 0
    expiries: list[SurfaceExpiry] = []
    points: list[Quote] = []
    grid: SurfaceGrid | None = None
