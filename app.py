import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests.exceptions import RequestException
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, kpss, pacf
import streamlit as st

warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data import fetch_data, preprocess_data
from src.features import engineer_features
from src.models import train_arima, train_lstm
from src.evaluation import calculate_metrics

st.set_page_config(
    page_title="Stock Price Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --surface: rgba(11, 20, 36, 0.88);
            --surface-strong: rgba(14, 26, 45, 0.96);
            --card: rgba(18, 32, 55, 0.92);
            --border: rgba(135, 168, 255, 0.14);
            --text: #edf3ff;
            --muted: #96a7c8;
            --accent: #5aa9ff;
            --accent-2: #32d3a2;
            --warning: #f3bf3d;
            --danger: #ff7a7a;
        }

        html, body, [class*="css"] {
            font-family: "Avenir Next", "Inter", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(90, 169, 255, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(50, 211, 162, 0.14), transparent 28%),
                linear-gradient(180deg, #07111f 0%, #081726 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            display: none;
        }

        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 3rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 17, 31, 0.96) 0%, rgba(9, 19, 34, 0.96) 100%);
            border-right: 1px solid rgba(135, 168, 255, 0.12);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(18, 32, 55, 0.96), rgba(10, 18, 31, 0.9));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.8rem 1.9rem;
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.24);
            margin-bottom: 1.8rem;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.85rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #9ac0ff;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-size: clamp(2.1rem, 4vw, 3.45rem);
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: var(--text);
        }

        .hero-title span {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-copy {
            max-width: 68ch;
            color: var(--muted);
            font-size: 1.02rem;
            margin-top: 0.9rem;
            line-height: 1.6;
        }

        .info-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.52rem 0.92rem;
            border-radius: 999px;
            border: 1px solid rgba(90, 169, 255, 0.25);
            background: rgba(90, 169, 255, 0.12);
            color: #b9d9ff;
            font-size: 0.9rem;
            margin-top: 1rem;
        }

        .quick-facts {
            background: var(--surface-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
        }

        .quick-facts h4 {
            margin: 0 0 0.7rem 0;
            font-size: 0.88rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #a8bbdf;
        }

        .quick-facts .fact-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.55rem 0;
            border-bottom: 1px solid rgba(135, 168, 255, 0.08);
            color: var(--text);
            font-size: 0.94rem;
        }

        .quick-facts .fact-row:last-child {
            border-bottom: none;
        }

        .quick-facts .fact-label {
            color: var(--muted);
        }

        .section-shell {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 20px 56px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        div[data-testid="stTabs"] {
            margin-top: 0.45rem;
        }

        .insight-card {
            background: linear-gradient(180deg, rgba(18, 32, 55, 0.95), rgba(13, 24, 41, 0.92));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            color: var(--text);
            height: 100%;
        }

        .insight-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
        }

        .insight-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .callout {
            background: linear-gradient(135deg, rgba(90, 169, 255, 0.15), rgba(50, 211, 162, 0.12));
            border: 1px solid rgba(90, 169, 255, 0.18);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            color: var(--text);
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(90, 169, 255, 0.2);
            background: linear-gradient(135deg, #2a66d8, #1fb58b);
            color: white;
            font-weight: 700;
            padding: 0.72rem 1rem;
        }

        .stButton > button:hover {
            border-color: rgba(90, 169, 255, 0.32);
            transform: translateY(-1px);
        }

        div[data-testid="stMetric"] {
            background: rgba(14, 26, 45, 0.96);
            border: 1px solid rgba(135, 168, 255, 0.12);
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.16);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        button[role="tab"] {
            border-radius: 999px !important;
            border: 1px solid rgba(135, 168, 255, 0.15) !important;
            background: rgba(12, 22, 38, 0.7) !important;
        }

        button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(42, 102, 216, 0.35), rgba(31, 181, 139, 0.22)) !important;
            border-color: rgba(90, 169, 255, 0.35) !important;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.6rem;
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            .hero-shell {
                padding: 1.35rem 1.2rem;
                margin-bottom: 1.3rem;
            }

            .section-shell {
                padding: 0.95rem 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float) -> str:
    return f"{value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def ensure_series(values, index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index)
    return pd.Series(np.asarray(values), index=index)


def compute_stationarity(series: pd.Series) -> dict:
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {
            "state": "Insufficient data",
            "adf_p": None,
            "kpss_p": None,
            "summary": "Not enough observations to run stationarity tests reliably.",
        }

    adf_p = None
    kpss_p = None
    adf_stationary = None
    kpss_stationary = None

    try:
        adf_result = adfuller(clean_series, autolag="AIC")
        adf_p = float(adf_result[1])
        adf_stationary = adf_p <= 0.05
    except Exception:
        pass

    try:
        kpss_result = kpss(clean_series, regression="c", nlags="auto")
        kpss_p = float(kpss_result[1])
        kpss_stationary = kpss_p > 0.05
    except Exception:
        pass

    if adf_stationary is False and kpss_stationary is False:
        state = "Non-stationary"
        summary = "ADF rejects stationarity and KPSS also signals instability, so the price series is likely non-stationary."
    elif adf_stationary is True and kpss_stationary is True:
        state = "Stationary"
        summary = "Both tests point to a stationary series, which is uncommon for raw stock prices."
    else:
        state = "Mixed signals"
        summary = "The stationarity tests do not fully agree, so the series likely benefits from differencing or feature transformations."

    return {
        "state": state,
        "adf_p": adf_p,
        "kpss_p": kpss_p,
        "summary": summary,
    }


def build_price_chart(df_clean: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_clean.index,
            y=df_clean["price"],
            mode="lines",
            name="Price",
            line=dict(color="#5aa9ff", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        )
    )
    rolling_mean = df_clean["price"].rolling(20, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=df_clean.index,
            y=rolling_mean,
            mode="lines",
            name="20D Moving Avg",
            line=dict(color="#32d3a2", width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>20D MA: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Price History",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_returns_chart(df_clean: pd.DataFrame) -> go.Figure:
    returns = df_clean["price"].pct_change().dropna()
    volatility = returns.rolling(20, min_periods=1).std()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            name="Daily Return",
            mode="lines",
            line=dict(color="#5aa9ff", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.3%}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility,
            name="20D Volatility",
            mode="lines",
            line=dict(color="#32d3a2", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Volatility: %{y:.3%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_dark",
        title="Returns and Volatility",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Daily Return", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title_text="Rolling Volatility", tickformat=".1%", secondary_y=True)
    return fig


def build_decomposition_chart(series: pd.Series):
    clean_series = series.dropna()
    period = None
    if len(clean_series) >= 504:
        period = 252
    elif len(clean_series) >= 120:
        period = 30
    elif len(clean_series) >= 35:
        period = 7

    if period is None:
        return None, "Need a longer history to generate a stable decomposition view."

    try:
        decomposition = seasonal_decompose(clean_series, model="additive", period=period, extrapolate_trend="freq")
    except Exception:
        return None, "Decomposition failed on this date range. Try a wider window for a clearer seasonal view."

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )
    components = [
        (decomposition.observed, "#5aa9ff"),
        (decomposition.trend, "#32d3a2"),
        (decomposition.seasonal, "#f3bf3d"),
        (decomposition.resid, "#ff7a7a"),
    ]
    for row, (component, color) in enumerate(components, start=1):
        fig.add_trace(
            go.Scatter(
                x=component.index,
                y=component,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        title=f"Seasonal Decomposition (period={period})",
        margin=dict(l=10, r=10, t=60, b=10),
        height=900,
    )
    return fig, None


def build_acf_pacf_chart(series: pd.Series) -> go.Figure | None:
    clean_series = series.dropna()
    if len(clean_series) < 15:
        return None

    lag_limit = min(40, len(clean_series) - 2)
    if lag_limit < 1:
        return None

    acf_values = acf(clean_series, nlags=lag_limit, fft=True)
    pacf_values = pacf(clean_series, nlags=lag_limit, method="ywm")
    lags = np.arange(len(acf_values))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(
        go.Bar(x=lags, y=acf_values, marker_color="#5aa9ff", hovertemplate="Lag %{x}<br>ACF %{y:.3f}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=lags, y=pacf_values, marker_color="#32d3a2", hovertemplate="Lag %{x}<br>PACF %{y:.3f}<extra></extra>"),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        title="Autocorrelation Diagnostics",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


def build_forecast_chart(df_clean: pd.DataFrame, train_data: pd.DataFrame, test_data: pd.DataFrame, results: dict, best_only: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_data.index,
            y=train_data["price"],
            mode="lines",
            name="History",
            line=dict(color="#7da9ff", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>History: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data["price"],
            mode="lines",
            name="Actual",
            line=dict(color="#e5eefc", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>",
        )
    )

    model_colors = {"ARIMA": "#f3bf3d", "LSTM": "#32d3a2"}
    items = list(results.items())
    if best_only and items:
        best_model = min(items, key=lambda item: item[1]["metrics"]["rmse"])[0]
        items = [(best_model, results[best_model])]

    for name, result in items:
        if name == "ARIMA":
            forecast_series = ensure_series(result["forecast"], test_data.index)
        else:
            forecast_series = ensure_series(result["predictions"], pd.to_datetime(result["dates"]))
            forecast_series = forecast_series.reindex(test_data.index)
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=forecast_series,
                mode="lines",
                name=name,
                line=dict(color=model_colors.get(name, "#9c8cff"), width=2.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Actual vs Forecasted Prices",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_error_chart(actual: pd.Series, forecast: pd.Series, model_name: str) -> go.Figure:
    errors = actual - forecast
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(color="#e5eefc", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(color="#32d3a2" if model_name == "LSTM" else "#f3bf3d", width=2.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=errors.index,
            y=errors,
            name="Residual",
            marker_color="#5aa9ff",
            hovertemplate="%{x|%Y-%m-%d}<br>Error: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"{model_name} Forecast Details",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


def build_metrics_frame(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, result in results.items():
        rows.append(
            {
                "Model": model_name,
                "RMSE": result["metrics"]["rmse"],
                "MAE": result["metrics"]["mae"],
                "MAPE (%)": result["metrics"]["mape"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    return metrics_df


def build_insights(results: dict, stationarity: dict) -> list[str]:
    insights = []
    if len(results) >= 2:
        sorted_models = sorted(results.items(), key=lambda item: item[1]["metrics"]["rmse"])
        best_name, best_result = sorted_models[0]
        runner_up_name, runner_up_result = sorted_models[1]
        improvement = ((runner_up_result["metrics"]["rmse"] - best_result["metrics"]["rmse"]) / runner_up_result["metrics"]["rmse"] * 100)
        insights.append(
            f"{best_name} is currently the strongest model and is {improvement:.1f}% better on RMSE than {runner_up_name}."
        )
    elif len(results) == 1:
        only_name = next(iter(results))
        insights.append(f"{only_name} is the only selected model, so the dashboard focuses on its out-of-sample accuracy.")

    if stationarity["state"] == "Non-stationary":
        insights.append("The raw price series is non-stationary, which is normal for stock levels and supports feature engineering or differencing.")
    elif stationarity["state"] == "Stationary":
        insights.append("The series appears stationary, which can simplify short-horizon forecasting.")

    return insights


def make_report_text(config: dict, metrics_df: pd.DataFrame, stationarity: dict, insights: list[str]) -> str:
    lines = [
        f"# Forecast Report - {config['ticker']}",
        "",
        f"Period: {config['start_date']} to {config['end_date']}",
        f"Models: {', '.join(config['models'])}",
        f"Mode: {'API' if config['use_api'] else 'Local'}",
        "",
        "## Stationarity",
        f"- State: {stationarity['state']}",
        f"- Summary: {stationarity['summary']}",
        "",
        "## Metrics",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['Model']}: RMSE {row['RMSE']:.2f}, MAE {row['MAE']:.2f}, MAPE {row['MAPE (%)']:.2f}%"
        )

    if insights:
        lines.extend(["", "## Key Insights"])
        lines.extend([f"- {item}" for item in insights])

    return "\n".join(lines)


def build_hero(config: dict | None) -> None:
    if config is None:
        config = {
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "models": ["ARIMA", "LSTM"],
            "use_api": False,
        }

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Finance analytics dashboard</div>
            <h1 class="hero-title">Stock Price <span>Forecasting System</span></h1>
            <div class="hero-copy">
                Compare ARIMA and LSTM forecasts in a clean production-style dashboard with interactive charts,
                KPI cards, model insights, and exportable results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown(
            """
            <div class="callout">
                <strong>Workflow</strong><br>
                Configure a ticker, choose one or both models, then run the pipeline to inspect the data, compare
                forecasts, and download a concise report.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="quick-facts">
                <h4>Current setup</h4>
                <div class="fact-row"><span class="fact-label">Ticker</span><span>{config['ticker']}</span></div>
                <div class="fact-row"><span class="fact-label">Range</span><span>{config['start_date']} → {config['end_date']}</span></div>
                <div class="fact-row"><span class="fact-label">Models</span><span>{', '.join(config['models']) or 'None'}</span></div>
                <div class="fact-row"><span class="fact-label">Mode</span><span>{'API' if config['use_api'] else 'Local'}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> dict | None:
    with st.sidebar:
        st.markdown("## Configuration")
        st.caption("Set the forecasting inputs, then launch the pipeline from the button below.")

        with st.form("forecast_config"):
            ticker = st.text_input("Ticker symbol", value="AAPL", help="Enter a public ticker such as AAPL, MSFT, or TSLA.")

            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input(
                    "Start date",
                    value=datetime(2020, 1, 1),
                    help="Earlier history improves model stability and EDA quality.",
                )
            with date_col2:
                end_date = st.date_input(
                    "End date",
                    value=datetime.now(),
                    help="Choose a recent end date to analyze the latest market regime.",
                )

            st.markdown("---")
            models = st.multiselect(
                "Models",
                ["ARIMA", "LSTM"],
                default=["ARIMA", "LSTM"],
                help="Select one or both models to compare forecasts side by side.",
            )
            epochs = st.slider(
                "LSTM epochs",
                min_value=10,
                max_value=150,
                value=40,
                step=10,
                help="Higher values may improve accuracy but increase runtime.",
            )
            seq_length = st.slider(
                "Sequence length",
                min_value=20,
                max_value=120,
                value=60,
                step=10,
                help="Controls how many previous days the LSTM sees at once.",
            )

            st.markdown("---")
            use_api = st.toggle(
                "Use live API",
                value=False,
                help="Fetch predictions from a running backend instead of training locally.",
            )
            api_url = st.text_input(
                "API endpoint",
                value="http://localhost:8000",
                help="Used only when live API mode is enabled.",
                disabled=not use_api,
            )

            submitted = st.form_submit_button("Run Forecasting Pipeline", use_container_width=True)

    if submitted:
        ticker = ticker.strip().upper()
        if not ticker:
            st.sidebar.error("Please provide a ticker symbol.")
            return None
        if not models:
            st.sidebar.error("Select at least one model.")
            return None
        if start_date >= end_date:
            st.sidebar.error("Start date must be earlier than end date.")
            return None

        return {
            "ticker": ticker,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "models": models,
            "epochs": epochs,
            "seq_length": seq_length,
            "use_api": use_api,
            "api_url": api_url.strip().rstrip("/"),
        }

    return None


def run_pipeline(config: dict) -> dict:
    progress = st.progress(0)
    status = st.empty()

    def update(step_label: str, pct: int) -> None:
        status.markdown(f"**{step_label}**")
        progress.progress(pct)

    with st.spinner("Running the forecasting pipeline..."):
        update("Data · Loading market history", 8)
        df_raw = fetch_data(config["ticker"], config["start_date"], config["end_date"])

        update("Data · Cleaning and preprocessing", 20)
        df_clean = preprocess_data(df_raw)

        update("EDA · Engineering features and diagnostics", 38)
        df_features = engineer_features(df_clean)
        stationarity = compute_stationarity(df_clean["price"])

        train_size = int(len(df_features) * 0.8)
        train_data = df_features.iloc[:train_size]
        test_data = df_features.iloc[train_size:]

        if len(test_data) == 0:
            raise ValueError("Not enough data to create a test split. Use a wider date range.")

        results = {}

        update("Features · Preparing train/test split", 52)

        if config["use_api"]:
            update("Models · Requesting API predictions", 68)
            payload = {
                "ticker": config["ticker"],
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "seq_length": config["seq_length"],
            }
            response = requests.post(f"{config['api_url']}/predict", json=payload, timeout=120)
            response.raise_for_status()
            api_data = response.json()

            if "ARIMA" in config["models"] and api_data.get("model_versions", {}).get("arima"):
                arima_forecast = ensure_series(api_data["arima_forecast"], pd.to_datetime(api_data.get("forecast_index", test_data.index)))
                arima_forecast = arima_forecast.reindex(test_data.index)
                results["ARIMA"] = {
                    "forecast": arima_forecast,
                    "metrics": calculate_metrics(test_data["price"], arima_forecast, "ARIMA"),
                }

            if "LSTM" in config["models"] and api_data.get("model_versions", {}).get("lstm"):
                lstm_forecast = ensure_series(api_data["lstm_forecast"], pd.to_datetime(api_data.get("forecast_index", test_data.index)))
                lstm_forecast = lstm_forecast.reindex(test_data.index)
                results["LSTM"] = {
                    "predictions": lstm_forecast.values,
                    "actuals": test_data["price"].values,
                    "dates": test_data.index,
                    "metrics": calculate_metrics(test_data["price"].values, lstm_forecast.values, "LSTM"),
                }
        else:
            if "ARIMA" in config["models"]:
                update("Models · Training ARIMA", 68)
                arima_forecast = train_arima(train_data["price"], test_data["price"])
                results["ARIMA"] = {
                    "forecast": arima_forecast,
                    "metrics": calculate_metrics(test_data["price"], arima_forecast, "ARIMA"),
                }

            if "LSTM" in config["models"]:
                update("Models · Training LSTM", 84)
                _, _, _, lstm_predictions, lstm_actuals, lstm_dates = train_lstm(
                    df_features,
                    train_size,
                    seq_length=config["seq_length"],
                    epochs=config["epochs"],
                )
                results["LSTM"] = {
                    "predictions": lstm_predictions,
                    "actuals": lstm_actuals,
                    "dates": lstm_dates,
                    "metrics": calculate_metrics(lstm_actuals, lstm_predictions, "LSTM"),
                }

        if not results:
            raise ValueError("No model results were generated. Check the selected models or API availability.")

        update("Results · Assembling dashboard", 95)
        metrics_df = build_metrics_frame(results)
        insights = build_insights(results, stationarity)
        report_text = make_report_text(config, metrics_df, stationarity, insights)
        progress.progress(100)
        status.markdown("**Pipeline complete**")

    return {
        "config": config,
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_features": df_features,
        "train_data": train_data,
        "test_data": test_data,
        "results": results,
        "metrics_df": metrics_df,
        "stationarity": stationarity,
        "insights": insights,
        "report_text": report_text,
        "rows": len(df_clean),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def render_overview(bundle: dict) -> None:
    config = bundle["config"]
    metrics_df = bundle["metrics_df"]
    insights = bundle["insights"]
    stationarity = bundle["stationarity"]
    results = bundle["results"]

    top = st.container()
    with top:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Rows analyzed", f"{bundle['rows']:,}")
        with cols[1]:
            best_model = metrics_df.iloc[0]["Model"] if not metrics_df.empty else "-"
            st.metric("Best model", best_model)
        with cols[2]:
            if len(metrics_df) > 1:
                runner_up = metrics_df.iloc[1]
                improvement = ((runner_up["RMSE"] - metrics_df.iloc[0]["RMSE"]) / runner_up["RMSE"] * 100)
                st.metric("RMSE advantage", f"{improvement:.1f}%")
            else:
                st.metric("RMSE advantage", "-")
        with cols[3]:
            st.metric("Stationarity", stationarity["state"])

    st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            """
            <div class="section-shell">
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Executive summary")
        if not metrics_df.empty:
            st.dataframe(
                metrics_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}).highlight_min(
                    subset=["RMSE", "MAE", "MAPE (%)"], color="rgba(50, 211, 162, 0.2)"
                ),
                use_container_width=True,
                height=230,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            """
            <div class="section-shell">
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Takeaways")
        if insights:
            for item in insights:
                st.markdown(f"- {item}")
        else:
            st.write("Run one or both models to generate a ranked comparison.")
        st.info(stationarity["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    if results:
        st.markdown(
            """
            <div class="section-shell">
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Quick forecast view")
        focus_best = st.toggle("Highlight best model only", value=False, help="Show only the top-ranked model in the forecast overlay.")
        st.plotly_chart(
            build_forecast_chart(bundle["df_clean"], bundle["train_data"], bundle["test_data"], results, best_only=focus_best),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
            key="overview-forecast-chart",
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_eda_tab(bundle: dict) -> None:
    df_clean = bundle["df_clean"]
    stationarity = bundle["stationarity"]

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(build_price_chart(df_clean), use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="eda-price-chart")
    with col2:
        st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
        st.subheader("Stationarity")
        st.metric("Series state", stationarity["state"])
        if stationarity["adf_p"] is not None:
            st.metric("ADF p-value", f"{stationarity['adf_p']:.4f}")
        if stationarity["kpss_p"] is not None:
            st.metric("KPSS p-value", f"{stationarity['kpss_p']:.4f}")
        st.info(stationarity["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(build_returns_chart(df_clean), use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="eda-returns-chart")

    decomposition_fig, decomposition_note = build_decomposition_chart(df_clean["price"])
    if decomposition_fig is not None:
        st.plotly_chart(decomposition_fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="eda-decomposition-chart")
    else:
        st.warning(decomposition_note)

    acf_pacf_fig = build_acf_pacf_chart(df_clean["price"])
    if acf_pacf_fig is not None:
        st.plotly_chart(acf_pacf_fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="eda-acf-pacf-chart")

    with st.expander("Cleaned data preview", expanded=False):
        st.dataframe(df_clean.head(25), use_container_width=True)


def render_models_tab(bundle: dict) -> None:
    results = bundle["results"]
    test_data = bundle["test_data"]

    if not results:
        st.info("Run the pipeline to inspect model-specific forecast details.")
        return

    model_names = list(results.keys())
    if len(model_names) > 1:
        model_tabs = st.tabs(model_names)
        for tab, model_name in zip(model_tabs, model_names):
            with tab:
                if model_name == "ARIMA":
                    forecast = ensure_series(results[model_name]["forecast"], test_data.index)
                    actual = test_data["price"]
                else:
                    actual = pd.Series(results[model_name]["actuals"], index=pd.to_datetime(results[model_name]["dates"]))
                    forecast = pd.Series(results[model_name]["predictions"], index=pd.to_datetime(results[model_name]["dates"]))

                left, right = st.columns([1.3, 1])
                with left:
                    st.plotly_chart(
                        build_error_chart(actual, forecast, model_name),
                        use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True},
                        key=f"model-detail-chart-{model_name.lower()}",
                    )
                with right:
                    st.metric("RMSE", format_number(results[model_name]["metrics"]["rmse"]))
                    st.metric("MAE", format_number(results[model_name]["metrics"]["mae"]))
                    st.metric("MAPE", format_percent(results[model_name]["metrics"]["mape"]))
                    model_df = pd.DataFrame(
                        {
                            "Date": actual.index,
                            "Actual": actual.values,
                            "Forecast": forecast.values,
                            "Error": actual.values - forecast.values,
                        }
                    )
                    st.dataframe(model_df, use_container_width=True, height=300)
                    st.download_button(
                        f"Download {model_name} CSV",
                        model_df.to_csv(index=False),
                        file_name=f"{bundle['config']['ticker']}_{model_name.lower()}_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
    else:
        model_name = model_names[0]
        if model_name == "ARIMA":
            forecast = ensure_series(results[model_name]["forecast"], test_data.index)
            actual = test_data["price"]
        else:
            actual = pd.Series(results[model_name]["actuals"], index=pd.to_datetime(results[model_name]["dates"]))
            forecast = pd.Series(results[model_name]["predictions"], index=pd.to_datetime(results[model_name]["dates"]))

        st.plotly_chart(build_error_chart(actual, forecast, model_name), use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key=f"model-detail-chart-{model_name.lower()}")
        model_df = pd.DataFrame(
            {
                "Date": actual.index,
                "Actual": actual.values,
                "Forecast": forecast.values,
                "Error": actual.values - forecast.values,
            }
        )
        st.dataframe(model_df, use_container_width=True, height=320)


def render_comparison_tab(bundle: dict) -> None:
    results = bundle["results"]
    metrics_df = bundle["metrics_df"]

    if not results:
        st.info("Run the pipeline to compare ARIMA and LSTM.")
        return

    chart_col, control_col = st.columns([1.5, 1])
    with control_col:
        focus_best = st.toggle("Best model only", value=False, help="Show only the strongest model in the overlay.")
        st.markdown(
            """
            <div class="section-shell">
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Ranking")
        st.dataframe(
            metrics_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}).highlight_min(
                subset=["RMSE", "MAE", "MAPE (%)"], color="rgba(50, 211, 162, 0.2)"
            ),
            use_container_width=True,
            height=260,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col:
        st.plotly_chart(
            build_forecast_chart(bundle["df_clean"], bundle["train_data"], bundle["test_data"], results, best_only=focus_best),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
            key="comparison-forecast-chart",
        )

    if not metrics_df.empty:
        comparison_fig = make_subplots(rows=1, cols=3, subplot_titles=("RMSE", "MAE", "MAPE"))
        palette = {"ARIMA": "#f3bf3d", "LSTM": "#32d3a2"}
        for metric_name, col_idx in zip(["RMSE", "MAE", "MAPE (%)"], [1, 2, 3]):
            for _, row in metrics_df.iterrows():
                comparison_fig.add_trace(
                    go.Bar(
                        x=[row["Model"]],
                        y=[row[metric_name]],
                        name=row["Model"],
                        marker_color=palette.get(row["Model"], "#5aa9ff"),
                        showlegend=col_idx == 1,
                        hovertemplate=f"{metric_name}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )
        comparison_fig.update_layout(template="plotly_dark", title="Metric Comparison", margin=dict(l=10, r=10, t=60, b=10), height=420)
        st.plotly_chart(comparison_fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True}, key="comparison-bar-chart")


def render_downloads_tab(bundle: dict) -> None:
    metrics_df = bundle["metrics_df"]
    report_text = bundle["report_text"]
    results = bundle["results"]
    config = bundle["config"]

    st.markdown(
        """
        <div class="section-shell">
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Downloads")
    st.caption("Export the forecast summary and the model outputs for offline review.")

    if not metrics_df.empty:
        st.download_button(
            "Download metrics CSV",
            metrics_df.to_csv(index=False),
            file_name=f"{config['ticker']}_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        "Download summary report",
        report_text,
        file_name=f"{config['ticker']}_forecast_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    combined_rows = []
    for model_name, result in results.items():
        if model_name == "ARIMA":
            series = ensure_series(result["forecast"], bundle["test_data"].index)
            actual = bundle["test_data"]["price"]
        else:
            actual = pd.Series(result["actuals"], index=pd.to_datetime(result["dates"]))
            series = pd.Series(result["predictions"], index=pd.to_datetime(result["dates"]))

        frame = pd.DataFrame(
            {
                "Model": model_name,
                "Date": actual.index,
                "Actual": actual.values,
                "Forecast": series.values,
                "Error": actual.values - series.values,
            }
        )
        combined_rows.append(frame)

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    if not combined_df.empty:
        st.download_button(
            "Download combined predictions CSV",
            combined_df.to_csv(index=False),
            file_name=f"{config['ticker']}_combined_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(combined_df.head(50), use_container_width=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()
    st.session_state.setdefault("analysis_bundle", None)
    st.session_state.setdefault("last_config", None)

    config = render_sidebar()
    if config is not None:
        try:
            bundle = run_pipeline(config)
            st.session_state["analysis_bundle"] = bundle
            st.session_state["last_config"] = config
            st.success("Forecasting pipeline completed.")
        except RequestException as exc:
            st.error(f"Unable to reach the API: {exc}")
        except Exception as exc:
            st.error(f"An error occurred: {exc}")

    build_hero(st.session_state.get("last_config"))

    bundle = st.session_state.get("analysis_bundle")
    if bundle is None:
        st.markdown(
            """
            <div class="section-shell">
                <h3 style="margin-top:0;">Get started</h3>
                <p class="metric-note">Use the sidebar to configure a ticker, date range, and model mix. The dashboard will then populate with EDA, forecasts, and downloads.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Modern layout</h4>
                    <p>Hero header, structured tabs, and clean spacing make the dashboard feel production-ready.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Interactive analytics</h4>
                    <p>Plotly charts add zoom, hover details, and a much better reading experience for time series work.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Actionable output</h4>
                    <p>KPIs, insights, and downloadable reports help translate model output into an executive-friendly view.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    st.markdown('<div style="height: 0.35rem;"></div>', unsafe_allow_html=True)
    overview_tab, eda_tab, models_tab, comparison_tab, downloads_tab = st.tabs(["Overview", "EDA", "Models", "Comparison", "Downloads"])

    with overview_tab:
        render_overview(bundle)
    with eda_tab:
        render_eda_tab(bundle)
    with models_tab:
        render_models_tab(bundle)
    with comparison_tab:
        render_comparison_tab(bundle)
    with downloads_tab:
        render_downloads_tab(bundle)

    st.markdown(
        """
        <div class="footer-note" style="margin-top: 1rem;">
            This dashboard is for exploratory forecasting only. Always validate trading or investment decisions with additional research and risk controls.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
