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

function MonteCarloTable({ horizons }) {
  const horizonOrder = ["1M", "3M", "6M", "1Y"];

  const modelNames = {
    student_t: "Student-t",
    garch_t: "GARCH-t",
    garch_jump: "GARCH-t + Jumps",
    bootstrap: "Block Bootstrap",
    hmm: "HMM Regime Switching",
  };

  return (
    <div className="mc-table-wrapper">
      {horizonOrder.map((horizon) => {
        const models = horizons?.[horizon];

        if (!models) return null;

        return (
          <div className="mc-horizon" key={horizon}>
            <div className="mc-horizon-title">{horizon}</div>

            <table className="mc-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>P05</th>
                  <th>Median</th>
                  <th>P95</th>
                  <th>Below Start</th>
                </tr>
              </thead>

              <tbody>
                {Object.entries(models).map(([model, result]) => (
                  <tr key={model}>
                    <td>{modelNames[model] || model}</td>
                    <td className="negative">
                      {formatPct(result?.terminal_p05)}
                    </td>
                    <td>{formatPct(result?.terminal_median)}</td>
                    <td className="positive">
                      {formatPct(result?.terminal_p95)}
                    </td>
                    <td>{formatPct(result?.prob_finish_below_start)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
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
              <h2>Monte Carlo Outcome Distribution</h2>
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

          <MonteCarloTable horizons={monteCarlo?.horizons} />
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