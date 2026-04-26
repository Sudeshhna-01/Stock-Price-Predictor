import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import requests
from requests.exceptions import RequestException

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import fetch_data, preprocess_data
from src.eda import perform_eda
from src.features import engineer_features
from src.models import train_arima, train_lstm
from src.evaluation import calculate_metrics, plot_results

st.set_page_config(
    page_title="Crystal Ball - Stock Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'


def get_css(theme: str) -> str:
    if theme == 'dark':
        background = '#090d1a'
        surface = 'rgba(18, 30, 63, 0.88)'
        text = '#e9edf8'
        muted = '#8695b0'
        border = 'rgba(166, 184, 255, 0.08)'
    else:
        background = '#eef3ff'
        surface = 'rgba(255, 255, 255, 0.92)'
        text = '#1e293b'
        muted = '#64748b'
        border = 'rgba(15, 23, 42, 0.08)'

    return f"""
    <style>
    .main-header {{
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #4f8fff 0%, #a75cf5 100%);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.35rem;
    }}
    .subheader-text {{
        color: {text};
        font-size: 1.15rem;
        margin-bottom: 1.65rem;
        opacity: 0.88;
    }}
    .hero-panel {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 28px;
        padding: 2.2rem;
        box-shadow: 0 32px 70px rgba(8, 18, 53, 0.2);
        margin-bottom: 1.8rem;
        backdrop-filter: blur(18px);
    }}
    .sidebar-content {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 1.25rem;
        color: {text};
        box-shadow: 0 24px 60px rgba(10, 22, 55, 0.16);
        transition: transform 0.25s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
    }}
    .accent-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(79, 143, 255, 0.12);
        color: #98c9ff;
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }}
    .footer-note {{
        color: {muted};
        font-size: 0.95rem;
    }}
    .stButton>button {{
        border-radius: 20px;
        padding: 0.95rem 1.6rem;
        font-size: 1rem;
        font-weight: 700;
    }}
    .stApp {{
        background: {background};
    }}
    </style>
    """


def toggle_theme() -> None:
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'


def main() -> None:
    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('## 🎨 Theme')
        if st.button('🌙 Dark' if st.session_state.theme == 'light' else '☀️ Light', key='theme_toggle'):
            toggle_theme()
            st.rerun()

        st.markdown('---')
        st.markdown('## 📈 Stock Setup')
        ticker = st.text_input('Stock Ticker', value='AAPL', max_chars=10)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('From', value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input('To', value=datetime.now())

        st.markdown('---')
        st.markdown('## 🧠 Model Settings')
        epochs = st.slider('LSTM Epochs', min_value=10, max_value=100, value=30, step=10)
        seq_length = st.slider('Sequence Length', min_value=20, max_value=120, value=60, step=10)

        st.markdown('---')
        st.markdown('## ⚙️ Forecast Controls')
        run_arima = st.checkbox('ARIMA', value=True)
        run_lstm = st.checkbox('LSTM', value=True)
        use_api = st.checkbox('Use Live API', value=False)
        api_url = st.text_input('API Endpoint', value='http://localhost:8000') if use_api else ''

        st.markdown('---')
        run_forecasting = st.button('🔮 Reveal the Future', width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Crystal Ball</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-text">A polished stock forecasting dashboard with ARIMA + LSTM, real-time API support, and interactive visuals.</div>', unsafe_allow_html=True)
    row1, row2 = st.columns([3, 1])
    with row1:
        st.markdown("""
        <div class='accent-pill'>
            <span>🔍 Market intelligence, simplified.</span>
        </div>
        """, unsafe_allow_html=True)
        st.write('Run the full pipeline from data ingestion to prediction analysis with a single click.')
    with row2:
        st.metric('Active Mode', 'API' if use_api else 'Local')
        st.metric('Selected Models', 'ARIMA + LSTM' if run_arima and run_lstm else 'ARIMA' if run_arima else 'LSTM' if run_lstm else 'None')
    st.markdown('</div>', unsafe_allow_html=True)

    if run_forecasting:
        try:
            with st.spinner('Consulting the crystal ball...'):
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                progress = st.progress(0)
                status = st.empty()

                status.text('📊 Loading market data...')
                progress.progress(10)
                df_raw = fetch_data(ticker, start_str, end_str)

                status.text('🧹 Preprocessing...')
                progress.progress(25)
                df_clean = preprocess_data(df_raw)

                status.text('⚙️ Engineering features...')
                progress.progress(45)
                df_features = engineer_features(df_clean)

                train_size = int(len(df_features) * 0.8)
                train_data = df_features.iloc[:train_size]
                test_data = df_features.iloc[train_size:]
                results = {}

                if use_api:
                    status.text('🛰️ Getting predictions from API...')
                    progress.progress(60)
                    payload = {'ticker': ticker, 'start_date': start_str, 'end_date': end_str, 'seq_length': seq_length}
                    response = requests.post(f'{api_url}/predict', json=payload, timeout=120)
                    response.raise_for_status()
                    api_data = response.json()
                    if run_arima and api_data.get('model_versions', {}).get('arima'):
                        arima_forecast = pd.Series(api_data['arima_forecast'], index=pd.to_datetime(api_data.get('forecast_index', test_data.index)))
                        arima_forecast = arima_forecast.reindex(test_data.index)
                        results['ARIMA'] = {'forecast': arima_forecast, 'metrics': calculate_metrics(test_data['price'], arima_forecast, 'ARIMA')}
                    if run_lstm and api_data.get('model_versions', {}).get('lstm'):
                        lstm_forecast = pd.Series(api_data['lstm_forecast'], index=pd.to_datetime(api_data.get('forecast_index', test_data.index)))
                        lstm_forecast = lstm_forecast.reindex(test_data.index)
                        results['LSTM'] = {'predictions': lstm_forecast.values, 'actuals': test_data['price'].values, 'dates': test_data.index, 'metrics': calculate_metrics(test_data['price'].values, lstm_forecast.values, 'LSTM')}
                else:
                    if run_arima:
                        status.text('📈 Training ARIMA...')
                        progress.progress(65)
                        arima_forecast = train_arima(train_data['price'], test_data['price'])
                        results['ARIMA'] = {'forecast': arima_forecast, 'metrics': calculate_metrics(test_data['price'], arima_forecast, 'ARIMA')}
                    if run_lstm:
                        status.text('🧠 Training LSTM...')
                        progress.progress(80)
                        _, _, _, lstm_predictions, lstm_actuals, lstm_dates = train_lstm(df_features, train_size, seq_length=seq_length, epochs=epochs)
                        results['LSTM'] = {'predictions': lstm_predictions, 'actuals': lstm_actuals, 'dates': lstm_dates, 'metrics': calculate_metrics(lstm_actuals, lstm_predictions, 'LSTM')}

                status.text('✨ Finalizing results...')
                progress.progress(95)
                display_results(df_clean, results, ticker, start_str, end_str, train_data, test_data)
                progress.progress(100)

        except RequestException as exc:
            st.error(f'Unable to reach the API: {exc}')
        except Exception as exc:
            st.error(f'An error occurred: {exc}')

    else:
        st.markdown('#### Why Crystal Ball?')
        cards = st.columns(3)
        with cards[0]:
            st.markdown("""
            <div class='metric-card'>
                <h4>Modern UX</h4>
                <p>Clean layout, readable analytics, and polished visuals.</p>
            </div>
            """, unsafe_allow_html=True)
        with cards[1]:
            st.markdown("""
            <div class='metric-card'>
                <h4>Hybrid Models</h4>
                <p>Compare ARIMA and LSTM side by side for better insight.</p>
            </div>
            """, unsafe_allow_html=True)
        with cards[2]:
            st.markdown("""
            <div class='metric-card'>
                <h4>Fast Workflow</h4>
                <p>Run local or API-backed forecasting in one click.</p>
            </div>
            """, unsafe_allow_html=True)


def display_results(df_clean, results, ticker, start_date, end_date, train_data, test_data):
    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    st.subheader('Forecast Summary')
    st.write(f'**{ticker}** from **{start_date}** to **{end_date}** with **{len(df_clean):,}** rows analyzed.')
    st.markdown('</div>', unsafe_allow_html=True)

    if results:
        cols = st.columns(len(results))
        for idx, (name, model) in enumerate(results.items()):
            with cols[idx]:
                st.metric(f'{name} RMSE', f"{model['metrics']['rmse']:.2f}")
                st.metric(f'{name} MAE', f"{model['metrics']['mae']:.2f}")
                st.metric(f'{name} MAPE', f"{model['metrics']['mape']:.2f}%")

    if 'ARIMA' in results and 'LSTM' in results:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['price'], mode='lines', name='History', line=dict(color='#6ea8ff', width=2)))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['price'], mode='lines', name='Actual', line=dict(color='#e2e8f0', width=2)))
        fig.add_trace(go.Scatter(x=test_data.index, y=results['ARIMA']['forecast'], mode='lines', name='ARIMA', line=dict(color='#f6c23e', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=results['LSTM']['dates'], y=results['LSTM']['predictions'], mode='lines', name='LSTM', line=dict(color='#3dd1b1', width=2, dash='dash')))
        fig.update_layout(title=f'{ticker} Forecast Comparison', template='plotly_dark', hovermode='x unified')
        st.plotly_chart(fig, use_container_width='stretch')

    if 'ARIMA' in results:
        with st.expander('ARIMA Predictions', expanded=False):
            arima_df = pd.DataFrame({
                'Date': test_data.index,
                'Actual': test_data['price'].values,
                'Forecast': results['ARIMA']['forecast'].values,
                'Error': test_data['price'].values - results['ARIMA']['forecast'].values
            })
            st.dataframe(arima_df, width='stretch')
            st.download_button('Download ARIMA Results', arima_df.to_csv(index=False), file_name=f'{ticker}_arima.csv', mime='text/csv')

    if 'LSTM' in results:
        with st.expander('LSTM Predictions', expanded=False):
            lstm_df = pd.DataFrame({
                'Date': results['LSTM']['dates'],
                'Actual': results['LSTM']['actuals'],
                'Forecast': results['LSTM']['predictions'],
                'Error': results['LSTM']['actuals'] - results['LSTM']['predictions']
            })
            st.dataframe(lstm_df, width='stretch')
            st.download_button('Download LSTM Results', lstm_df.to_csv(index=False), file_name=f'{ticker}_lstm.csv', mime='text/csv')

    st.markdown('---')
    st.markdown("""
    <div class='footer-note'>
        This dashboard is for exploratory forecasting only. Always verify decisions with your own research.
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
