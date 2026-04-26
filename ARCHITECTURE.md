# Real-Time Pipeline Architecture

This architecture is designed for a production-grade stock forecasting system with real-time ingestion, model versioning, retraining, serving, and observability.

## Components

1. **Data Ingestion**
   - Source: Yahoo Finance API (`src.data.fetch_data`)
   - Process: periodic polling of market data, conversion to business-day frequency, calculation of returns and log returns
   - Output: cleaned historical price data stored in memory and optionally persisted for audit

2. **Feature Engineering**
   - Source: cleaned price data
   - Process: compute multivariate technical features such as RSI, MACD, ATR, OBV, momentum, and rolling volatility
   - Output: dataset suitable for ARIMA+LSTM training and serving

3. **Model Training and Versioning**
   - Module: `src.pipeline`
   - Process: train ARIMA and LSTM models on the latest dataset
   - Versioning: save the trained model, scaler, feature configuration, and metadata into timestamped directories under `models/arima/` and `models/lstm/`
   - Metadata: stores ticker, training window, sequence length, epochs, model version, and creation timestamp

4. **Automated Retraining**
   - Module: `retrain.py`
   - Scheduler: APScheduler runs retraining at a configurable cadence (e.g., daily at 02:00 UTC)
   - Behavior: fetch latest data, run training pipeline, publish new versioned models

5. **Prediction API**
   - Component: `api.py` (FastAPI)
   - Endpoints:
     - `/predict`: returns ARIMA and LSTM forecasts using latest saved models
     - `/retrain`: triggers manual retraining and version publication
     - `/health`: service liveness
     - `/metrics`: API request and retrain counters
   - Deployment: serve with Uvicorn for low-latency access

6. **Dashboard**
   - Component: `app.py` (Streamlit)
   - Purpose: user-facing analytics dashboard, model performance visualization, and optional API-driven predictions
   - Integration: can either run locally or connect to the FastAPI service for real-time predictions

7. **Logging and Monitoring**
   - Module: `src.logging_config.py`
   - Logging:
     - pipeline events
     - data ingestion operations
     - retraining jobs
     - API prediction and error metrics
   - Storage: rotating log files under `logs/pipeline.log`

## Data Flow

1. User requests a prediction via the Streamlit dashboard or FastAPI `/predict` endpoint.
2. The ingestion module fetches fresh market data and preprocesses it.
3. The feature engine computes real-time technical indicators and volatility signals.
4. The prediction service loads the latest versioned ARIMA and LSTM models.
5. Predictions are produced and returned to the dashboard or API client.
6. A scheduled retraining job refreshes the models periodically, publishing new versions.
7. Logs and metrics capture runtime behavior for observability.

## Deployment Topology

- `Streamlit Dashboard` connects to `FastAPI Prediction API`
- `FastAPI Prediction API` serves versioned models from `models/`
- `Retraining Scheduler` writes versioned models to `models/`
- `Logging` writes to `logs/pipeline.log`

## Real-Time Considerations

- Use a polling / streaming data source for continuous market updates.
- Keep the model store immutable by writing new versions to separate directories.
- Avoid data leakage by always using historical windows and separate train/test split logic.
- Monitor model drift and retraining frequency with service metrics and logs.
