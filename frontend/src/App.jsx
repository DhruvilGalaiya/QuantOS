import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronDown,
  CircleAlert,
  Gauge,
  Layers3,
  RefreshCw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/quantos/state";

function formatPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatDate(value) {
  if (!value) return "—";
  return String(value).slice(0, 10);
}

function RegimeBadge({ regime }) {
  const normalized = String(regime || "UNKNOWN").toUpperCase();

  return (
    <span className={`regime-badge ${normalized.toLowerCase()}`}>
      {normalized}
    </span>
  );
}

function MetricCard({ label, value, subtext, icon: Icon, tone = "" }) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-top">
        <span>{label}</span>
        {Icon && <Icon size={17} />}
      </div>

      <div className="metric-value">{value}</div>

      {subtext && <div className="metric-subtext">{subtext}</div>}
    </div>
  );
}

function ProbabilityBar({ label, value, className }) {
  return (
    <div className="probability-row">
      <div className="probability-label">
        <span>{label}</span>
        <strong>{formatPct(value, 1)}</strong>
      </div>

      <div className="probability-track">
        <div
          className={`probability-fill ${className}`}
          style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }}
        />
      </div>
    </div>
  );
}

function MonteCarloScenarioLab({ horizons, startingValue }) {
  const [selectedHorizon, setSelectedHorizon] = useState("1M");
  const [selectedModel, setSelectedModel] = useState("all");
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const horizonLabels = {
    "1M": "1 Month",
    "3M": "3 Months",
    "6M": "6 Months",
    "1Y": "1 Year",
  };

  const modelNames = {
    student_t: "Student-t",
    garch_t: "GARCH-t",
    garch_jump: "GARCH-t + Jumps",
    bootstrap: "Block Bootstrap",
    hmm: "HMM Regime Switching",
  };

  const models = horizons?.[selectedHorizon] || {};

  const modelOrder = [
    "student_t",
    "garch_t",
    "garch_jump",
    "bootstrap",
    "hmm",
  ];

  const selectedResult =
  selectedModel === "all"
    ? models.hmm || models.student_t || Object.values(models)[0]
    : models[selectedModel];

const selectedScenarioLabel =
  selectedModel === "all"
    ? "HMM primary scenario + model comparison"
    : modelNames[selectedModel];

  const visualization = selectedResult?.path_visualization;
  const percentilePaths = visualization?.percentile_paths || {};
  const paths = visualization?.paths || [];

  const days = visualization?.days || [];

  const chartWidth = 920;
  const chartHeight = 420;

  const padding = {
    top: 24,
    right: 28,
    bottom: 46,
    left: 78,
  };

  const plotWidth = chartWidth - padding.left - padding.right;
  const plotHeight = chartHeight - padding.top - padding.bottom;

  const allValues = [
    ...(percentilePaths.p05 || []),
    ...(percentilePaths.p95 || []),
    ...paths.flat(),
  ].filter((value) => Number.isFinite(Number(value)));

  let minValue = allValues.length ? Math.min(...allValues) : 0;
  let maxValue = allValues.length ? Math.max(...allValues) : 1;

  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }

  const range = maxValue - minValue;
  const chartMin = minValue - range * 0.08;
  const chartMax = maxValue + range * 0.08;

  function xScale(index) {
    const denominator = Math.max(days.length - 1, 1);
    return padding.left + (index / denominator) * plotWidth;
  }

  function yScale(value) {
    return (
      padding.top +
      ((chartMax - Number(value)) / (chartMax - chartMin)) * plotHeight
    );
  }

  function createPolyline(values) {
    if (!values?.length) return "";

    return values
      .map((value, index) => `${xScale(index)},${yScale(value)}`)
      .join(" ");
  }

  function createBandPolygon(lower, upper) {
    if (!lower?.length || !upper?.length) return "";

    const upperPoints = upper.map(
      (value, index) => `${xScale(index)},${yScale(value)}`
    );

    const lowerPoints = [...lower]
      .reverse()
      .map((value, reverseIndex) => {
        const index = lower.length - 1 - reverseIndex;
        return `${xScale(index)},${yScale(value)}`;
      });

    return [...upperPoints, ...lowerPoints].join(" ");
  }

  function formatPrice(value, digits = 0) {
    if (
      value === null ||
      value === undefined ||
      !Number.isFinite(Number(value))
    ) {
      return "—";
    }

    return `₹${Number(value).toLocaleString("en-IN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })}`;
  }

  function formatProbability(value) {
    if (
      value === null ||
      value === undefined ||
      !Number.isFinite(Number(value))
    ) {
      return "—";
    }

    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatRisk(value) {
    if (
      value === null ||
      value === undefined ||
      !Number.isFinite(Number(value))
    ) {
      return "—";
    }

    return `${(Number(value) * 100).toFixed(2)}%`;
  }

  function getSummaryValue(result, ...keys) {
    for (const key of keys) {
      const value = result?.summary?.[key];

      if (
        value !== null &&
        value !== undefined &&
        Number.isFinite(Number(value))
      ) {
        return Number(value);
      }
    }

    return null;
  }

  function getOutcomeValue(result, key) {
    const value = result?.possible_outcome_levels?.[key];

    return value !== undefined && value !== null ? Number(value) : null;
  }

  function getRiskProbability(result, key) {
    const value = result?.risk_probabilities?.[key];

    return value !== undefined && value !== null ? Number(value) : null;
  }

  const p05 = getOutcomeValue(selectedResult, "p05");
  const p25 = getOutcomeValue(selectedResult, "p25");
  const median = getOutcomeValue(selectedResult, "median");
  const p75 = getOutcomeValue(selectedResult, "p75");
  const p95 = getOutcomeValue(selectedResult, "p95");

  const belowStart = getRiskProbability(
    selectedResult,
    "finish_below_start"
  );

  const hitTarget = getRiskProbability(
    selectedResult,
    "hit_plus_10pct"
  );

  const hitStop = getRiskProbability(
    selectedResult,
    "hit_minus_10pct"
  );

  const var95 = getSummaryValue(
    selectedResult,
    "var_95",
    "VaR_95",
    "VaR95",
    "var95"
  );

  const cvar95 = getSummaryValue(
    selectedResult,
    "cvar_95",
    "CVaR_95",
    "CVaR95",
    "cvar95"
  );

  const terminalReturn = getSummaryValue(
    selectedResult,
    "median_terminal_return",
    "terminal_return_median"
  );

  const xTickIndexes =
    days.length > 1
      ? [
          0,
          Math.floor((days.length - 1) * 0.25),
          Math.floor((days.length - 1) * 0.5),
          Math.floor((days.length - 1) * 0.75),
          days.length - 1,
        ]
      : [0];

  function handlePointHover(event, index, value) {
    const rect = event.currentTarget.ownerSVGElement.getBoundingClientRect();

    setHoveredPoint({
      index,
      value,
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }

  return (
    <div className="mc-lab">
      <div className="mc-controls">
        <div className="mc-control-group">
          <label htmlFor="mc-horizon">Horizon</label>
          <div className="mc-select-wrapper">
            <select
              id="mc-horizon"
              value={selectedHorizon}
              onChange={(event) => {
                setSelectedHorizon(event.target.value);
                setHoveredPoint(null);
              }}
            >
              <option value="1M">1 Month</option>
              <option value="3M">3 Months</option>
              <option value="6M">6 Months</option>
              <option value="1Y">1 Year</option>
            </select>
            <ChevronDown size={16} />
          </div>
        </div>

        <div className="mc-control-group">
          <label htmlFor="mc-model">Model</label>
          <div className="mc-select-wrapper">
            <select
              id="mc-model"
              value={selectedModel}
              onChange={(event) => {
                setSelectedModel(event.target.value);
                setHoveredPoint(null);
              }}
            >
              <option value="all">All Models</option>

              {modelOrder.map((model) => (
                <option key={model} value={model}>
                  {modelNames[model]}
                </option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        </div>

        <div className="mc-control-info">
          <span>Scenario set</span>
          <strong>{selectedScenarioLabel}</strong>
        </div>
      </div>

      <div className="mc-chart-header">
        <div>
          <div className="mc-chart-title">
            NIFTY 50 simulated price paths
          </div>
          <div className="mc-chart-subtitle">
            {horizonLabels[selectedHorizon]} ·{" "}
            {visualization?.total_simulation_paths?.toLocaleString() ||
              "10,000"}{" "}
            simulations · 60 representative paths shown
          </div>
        </div>

        <div className="mc-scenario-label">
          SCENARIOS, NOT PREDICTIONS
        </div>
      </div>

      <div className="mc-visual-row">
        <div className="mc-chart-container">
          {days.length > 0 ? (
            <svg
              className="mc-chart"
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              role="img"
              aria-label={`Monte Carlo simulated NIFTY 50 price paths for ${horizonLabels[selectedHorizon]}`}
            >
              <rect
                x="0"
                y="0"
                width={chartWidth}
                height={chartHeight}
                className="mc-chart-background"
              />

              {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
                const value =
                  chartMax - fraction * (chartMax - chartMin);
                const y = yScale(value);

                return (
                  <g key={`y-grid-${fraction}`}>
                    <line
                      x1={padding.left}
                      x2={chartWidth - padding.right}
                      y1={y}
                      y2={y}
                      className="mc-grid-line"
                    />

                    <text
                      x={padding.left - 12}
                      y={y + 4}
                      textAnchor="end"
                      className="mc-axis-label"
                    >
                      {formatPrice(value)}
                    </text>
                  </g>
                );
              })}

              {xTickIndexes.map((index) => (
                <g key={`x-grid-${index}`}>
                  <line
                    x1={xScale(index)}
                    x2={xScale(index)}
                    y1={padding.top}
                    y2={chartHeight - padding.bottom}
                    className="mc-grid-line vertical"
                  />

                  <text
                    x={xScale(index)}
                    y={chartHeight - 18}
                    textAnchor="middle"
                    className="mc-axis-label"
                  >
                    {index === 0 ? "Today" : `Day ${days[index]}`}
                  </text>
                </g>
              ))}

              {percentilePaths.p05 && percentilePaths.p95 && (
                <polygon
                  points={createBandPolygon(
                    percentilePaths.p05,
                    percentilePaths.p95
                  )}
                  className="mc-band-outer"
                />
              )}

              {percentilePaths.p25 && percentilePaths.p75 && (
                <polygon
                  points={createBandPolygon(
                    percentilePaths.p25,
                    percentilePaths.p75
                  )}
                  className="mc-band-inner"
                />
              )}

              {paths.map((path, pathIndex) => {
                const pathColors = [
                  "#2563eb",
                  "#06b6d4",
                  "#10b981",
                  "#f59e0b",
                  "#ef4444",
                  "#8b5cf6",
                  "#ec4899",
                  "#14b8a6",
                ];

                return (
                  <polyline
                    key={`path-${pathIndex}`}
                    points={createPolyline(path)}
                    className="mc-simulation-path"
                    style={{
                      stroke: pathColors[pathIndex % pathColors.length],
                    }}
                  />
                );
              })}

              {percentilePaths.p05 && (
                <polyline
                  points={createPolyline(percentilePaths.p05)}
                  className="mc-percentile p05"
                />
              )}

              {percentilePaths.p95 && (
                <polyline
                  points={createPolyline(percentilePaths.p95)}
                  className="mc-percentile p95"
                />
              )}

              {percentilePaths.median && (
                <polyline
                  points={createPolyline(percentilePaths.median)}
                  className="mc-percentile median"
                />
              )}

              {percentilePaths.median?.map((value, index) => (
                <circle
                  key={`hover-point-${index}`}
                  cx={xScale(index)}
                  cy={yScale(value)}
                  r="9"
                  className="mc-hover-target"
                  onMouseEnter={(event) =>
                    handlePointHover(event, index, value)
                  }
                />
              ))}

              {hoveredPoint && (
                <g
                  className="mc-tooltip"
                  transform={`translate(${Math.min(
                    hoveredPoint.x + 12,
                    chartWidth - 150
                  )}, ${Math.max(hoveredPoint.y - 52, 8)})`}
                >
                  <rect width="138" height="44" rx="7" />

                  <text x="10" y="17">
                    Day {days[hoveredPoint.index]}
                  </text>

                  <text x="10" y="34">
                    Median {formatPrice(hoveredPoint.value, 2)}
                  </text>
                </g>
              )}
            </svg>
          ) : (
            <div className="mc-empty">
              Monte Carlo path visualization unavailable.
            </div>
          )}
        </div>

        <div className="mc-chart-sidebar">
          <div className="mc-sidebar-title">Line Legend</div>

          <div className="mc-legend-item">
            <div className="mc-legend-sample mc-sample-paths">
              <span />
              <span />
              <span />
            </div>

            <div>
              <strong>Individual simulated paths</strong>
              <p>
                Each coloured line represents one possible NIFTY 50
                price path. 60 representative paths are shown.
              </p>
            </div>
          </div>

          <div className="mc-legend-item">
            <div className="mc-legend-sample">
              <i className="mc-sample-median" />
            </div>

            <div>
              <strong>Median path (P50)</strong>
              <p>
                The middle outcome across all simulated paths.
              </p>
            </div>
          </div>

          <div className="mc-legend-item">
            <div className="mc-legend-sample">
              <i className="mc-sample-percentile" />
            </div>

            <div>
              <strong>5th percentile (P05)</strong>
              <p>
                Lower outcome boundary. Only 5% of simulated
                terminal outcomes are below this level.
              </p>
            </div>
          </div>

          <div className="mc-legend-item">
            <div className="mc-legend-sample">
              <i className="mc-sample-percentile" />
            </div>

            <div>
              <strong>95th percentile (P95)</strong>
              <p>
                Upper outcome boundary. Only 5% of simulated
                terminal outcomes are above this level.
              </p>
            </div>
          </div>

          <div className="mc-how-to-read">
            <div className="mc-info-icon">i</div>

            <div>
              <strong>How to read this chart</strong>

              <p>
                Each coloured line is one simulated scenario under
                the selected model.
              </p>

              <p>
                The thick blue line is the median simulated path.
                The dashed lines show the P05 and P95 boundaries,
                giving a sense of the range of possible outcomes.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mc-outcome-grid">
        <div>
          <span>P05</span>
          <strong>{formatPrice(p05, 0)}</strong>
        </div>

        <div>
          <span>P25</span>
          <strong>{formatPrice(p25, 0)}</strong>
        </div>

        <div className="mc-outcome-median">
          <span>Median</span>
          <strong>{formatPrice(median, 0)}</strong>
        </div>

        <div>
          <span>P75</span>
          <strong>{formatPrice(p75, 0)}</strong>
        </div>

        <div>
          <span>P95</span>
          <strong>{formatPrice(p95, 0)}</strong>
        </div>
      </div>

      <div className="mc-risk-grid">
        <div>
          <span>Probability below start</span>
          <strong>{formatProbability(belowStart)}</strong>
        </div>

        <div>
          <span>Probability +10% target</span>
          <strong>{formatProbability(hitTarget)}</strong>
        </div>

        <div>
          <span>Probability -10% stop</span>
          <strong>{formatProbability(hitStop)}</strong>
        </div>

        <div>
          <span>VaR 95%</span>
          <strong>{formatRisk(var95)}</strong>
        </div>

        <div>
          <span>CVaR 95%</span>
          <strong>{formatRisk(cvar95)}</strong>
        </div>
      </div>

      {selectedModel === "all" && (
        <div className="mc-model-comparison">
          <div className="mc-comparison-header">
            <div>
              <div className="mc-chart-title">Model comparison</div>
              <div className="mc-chart-subtitle">
                Terminal outcome distribution for the selected horizon
              </div>
            </div>
          </div>

          <div className="mc-comparison-list">
            {modelOrder.map((model) => {
              const result = models[model];

              if (!result) return null;

              const modelMedian = getOutcomeValue(result, "median");
              const modelP05 = getOutcomeValue(result, "p05");
              const modelP95 = getOutcomeValue(result, "p95");
              const modelLoss = getRiskProbability(
                result,
                "finish_below_start"
              );

              return (
                <div className="mc-comparison-row" key={model}>
                  <div className="mc-comparison-model">
                    {modelNames[model]}
                  </div>

                  <div>
                    <span>P05</span>
                    <strong>{formatPrice(modelP05)}</strong>
                  </div>

                  <div>
                    <span>Median</span>
                    <strong>{formatPrice(modelMedian)}</strong>
                  </div>

                  <div>
                    <span>P95</span>
                    <strong>{formatPrice(modelP95)}</strong>
                  </div>

                  <div>
                    <span>Below start</span>
                    <strong>{formatProbability(modelLoss)}</strong>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mc-method-note">
        <ShieldCheck size={15} />
        <span>
          Monte Carlo results represent simulated scenarios under the selected
          model assumptions. They are not forecasts or investment
          recommendations.
        </span>
      </div>

      {terminalReturn !== null && (
        <div className="mc-hidden-metric">
          Median terminal return: {formatPct(terminalReturn)}
        </div>
      )}
    </div>
  );
}

function App() {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  async function loadState() {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(API_URL);

      if (!response.ok) {
        throw new Error(`API returned HTTP ${response.status}`);
      }

      const data = await response.json();

      setState(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadState();
  }, []);

  const market = state?.market;
  const options = market?.options;
  const volatility = market?.realized_volatility;
  const forecast = market?.forecast_volatility;
  const riskState = market?.risk_state;
  const monteCarlo = state?.monte_carlo;

  const probability = useMemo(
    () => market?.regime_probability || {},
    [market]
  );

  if (loading) {
    return (
      <div className="app-shell loading-screen">
        <div className="loading-card">
          <BrainCircuit size={30} />
          <h1>QuantOS</h1>
          <p>Loading quantitative market state...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-shell loading-screen">
        <div className="error-card">
          <CircleAlert size={32} />
          <h1>QuantOS API unavailable</h1>

          <p>{error}</p>

          <code>http://127.0.0.1:8000/api/quantos/state</code>

          <button onClick={loadState}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={21} />
          </div>

          <div>
            <div className="brand-name">QuantOS</div>
            <div className="brand-subtitle">
              Quantitative Market Intelligence
            </div>
          </div>
        </div>

        <div className="topbar-right">
          <div className="status-dot">
            <span />
            LIVE STATE
          </div>

          <button className="refresh-button" onClick={loadState}>
            <RefreshCw size={15} />
            Refresh
          </button>
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <div>
            <div className="eyebrow">MARKET INTELLIGENCE</div>

            <h1>
              NIFTY 50
              <span> / Quantitative State</span>
            </h1>

            <p className="hero-description">
              A model-driven view of latent regime, volatility, options
              structure and simulated outcome distributions.
            </p>
          </div>

          <div className="freshness">
            <span>Underlying data</span>
            <strong>{formatDate(market?.hmm_date)}</strong>
          </div>
        </section>

        <section className="metrics-grid">
          <MetricCard
            label="Market Regime"
            value={<RegimeBadge regime={market?.regime} />}
            subtext="Causal HMM state"
            icon={BrainCircuit}
            tone="regime-card"
          />

          <MetricCard
            label="NIFTY 50"
            value={`₹${formatNumber(market?.market_data?.latest_close)}`}
            subtext={`Market date ${formatDate(market?.market_data?.volatility_model_date)}`}
            icon={Activity}
          />

          <MetricCard
            label="21D Realized Vol"
            value={formatPct(volatility?.vol_21d)}
            subtext={volatility?.volatility_level || "—"}
            icon={Gauge}
          />

          <MetricCard
            label="GARCH 21D"
            value={formatPct(forecast?.garch_21d)}
            subtext="Annualized forecast"
            icon={TrendingUp}
          />
        </section>

        <section className="main-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">REGIME ENGINE</div>
                <h2>Latent Market State</h2>
              </div>

              <BrainCircuit size={21} />
            </div>

            <div className="regime-summary">
              <div>
                <span className="muted-label">Current state</span>
                <div className="large-regime">
                  <RegimeBadge regime={market?.regime} />
                </div>
              </div>

              <div className="regime-confidence">
                <span className="muted-label">Dominant probability</span>
                <strong>{formatPct(probability.bear, 1)}</strong>
              </div>
            </div>

            <div className="probabilities">
              <ProbabilityBar
                label="BULL"
                value={probability.bull}
                className="bull"
              />

              <ProbabilityBar
                label="SIDE"
                value={probability.side}
                className="side"
              />

              <ProbabilityBar
                label="BEAR"
                value={probability.bear}
                className="bear"
              />
            </div>

            <div className="method-note">
              <ShieldCheck size={16} />
              <span>
                Probabilities shown here come from the synchronized HMM
                volatility-bridge state.
              </span>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">VOLATILITY INTELLIGENCE</div>
                <h2>Risk Environment</h2>
              </div>

              <Gauge size={21} />
            </div>

            <div className="vol-grid">
              <div>
                <span>5D realized</span>
                <strong>{formatPct(market?.hmm_current_volatility?.vol_5d)}</strong>
              </div>

              <div>
                <span>20D realized</span>
                <strong>{formatPct(market?.hmm_current_volatility?.vol_20d)}</strong>
              </div>

              <div>
                <span>60D realized</span>
                <strong>{formatPct(market?.hmm_current_volatility?.vol_60d)}</strong>
              </div>

              <div>
                <span>GARCH 1D</span>
                <strong>{formatPct(forecast?.garch_1d)}</strong>
              </div>

              <div>
                <span>GARCH 5D</span>
                <strong>{formatPct(forecast?.garch_5d)}</strong>
              </div>

              <div>
                <span>HAR 1D</span>
                <strong>{formatPct(forecast?.har_1d)}</strong>
              </div>
            </div>

            <div className="state-strip">
              <div>
                <span>Realized state</span>
                <strong>{riskState?.realized_vol_state || "—"}</strong>
              </div>

              <div>
                <span>IV vs realized</span>
                <strong>{riskState?.iv_vs_realized || "—"}</strong>
              </div>

              <div>
                <span>Term structure</span>
                <strong>{riskState?.term_structure_state || "—"}</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="panel options-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">OPTIONS MARKET</div>
              <h2>Implied Volatility Surface</h2>
            </div>

            <BarChart3 size={21} />
          </div>

          <div className="options-grid">
            <MetricCard
              label="Spot"
              value={`₹${formatNumber(options?.spot)}`}
              subtext={`ATM strike ₹${formatNumber(options?.atm_strike)}`}
              icon={Activity}
            />

            <MetricCard
              label="ATM IV"
              value={formatPct(options?.atm_iv_nearest)}
              subtext={`Expiry ${options?.nearest_expiry || "—"}`}
              icon={Gauge}
            />

            <MetricCard
              label="Next Expiry IV"
              value={formatPct(options?.atm_iv_next)}
              subtext={options?.next_expiry || "—"}
              icon={TrendingUp}
            />

            <MetricCard
              label="IV / Realized"
              value={formatNumber(options?.iv_realized_ratio, 2)}
              subtext="Market-implied / realized"
              icon={BarChart3}
            />
          </div>
        </section>

        <section className="panel monte-carlo-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">SCENARIO ENGINE</div>
              <h2>Monte Carlo Scenario Lab</h2>
            </div>

            <Layers3 size={21} />
          </div>

          <div className="mc-meta">
            <span>
              <strong>{monteCarlo?.metadata?.n_paths?.toLocaleString()}</strong>{" "}
              simulation paths
            </span>

            <span>
              Starting value{" "}
              <strong>
                ₹{formatNumber(monteCarlo?.metadata?.starting_value)}
              </strong>
            </span>

            <span>
              Regime <RegimeBadge regime={monteCarlo?.metadata?.current_regime} />
            </span>
          </div>

          <MonteCarloScenarioLab
  horizons={monteCarlo?.horizons}
  startingValue={monteCarlo?.metadata?.starting_value}
/>
        </section>

        <section className="bottom-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">REGIME-CONDITIONAL RISK</div>
                <h2>Forward Volatility</h2>
              </div>

              <TrendingDown size={21} />
            </div>

            <div className="forward-vol">
              <div className="forward-vol-number">
                {formatPct(riskState?.regime_conditioned_21d_vol)}
              </div>

              <p>
                Historical forward 21-day volatility associated with the
                current latent regime.
              </p>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">SYSTEM STATUS</div>
                <h2>Data Provenance</h2>
              </div>

              <ShieldCheck size={21} />
            </div>

            <div className="provenance">
              <div>
                <span>Market / HMM</span>
                <strong>{formatDate(market?.hmm_date)}</strong>
              </div>

              <div>
                <span>Options surface</span>
                <strong>{formatDate(options?.timestamp)}</strong>
              </div>

              <div>
                <span>Volatility model</span>
                <strong>{formatDate(market?.market_data?.volatility_model_date)}</strong>
              </div>

              <div>
                <span>Dashboard refresh</span>
                <strong>
                  {lastUpdated
                    ? lastUpdated.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "—"}
                </strong>
              </div>
            </div>
          </div>
        </section>

        <footer>
          <span>QuantOS · Quantitative Market Intelligence Platform</span>
          <span>Research system · Not investment advice</span>
        </footer>
      </main>
    </div>
  );
}

export default App;