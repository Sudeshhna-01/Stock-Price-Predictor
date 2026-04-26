import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pmdarima.arima import AutoARIMA

from src.data import fetch_data, preprocess_data
from src.features import engineer_features
from src.models import train_arima, train_lstm
from src.logging_config import logger

MODEL_STORE_DIR = Path("models")
ARIMA_DIR = MODEL_STORE_DIR / "arima"
LSTM_DIR = MODEL_STORE_DIR / "lstm"


def ensure_model_directories():
    MODEL_STORE_DIR.mkdir(exist_ok=True)
    ARIMA_DIR.mkdir(exist_ok=True)
    LSTM_DIR.mkdir(exist_ok=True)


def version_name():
    return datetime.utcnow().strftime("v%Y%m%d%H%M%S")


def latest_version_path(model_dir: Path):
    if not model_dir.exists():
        return None
    versions = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    return versions[-1] if versions else None


def save_metadata(model_dir: Path, metadata: dict):
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(model_dir: Path) -> dict:
    with open(model_dir / "metadata.json", "r") as f:
        return json.load(f)


def save_arima_model(model: AutoARIMA, metadata: dict):
    ensure_model_directories()
    version_dir = ARIMA_DIR / version_name()
    version_dir.mkdir(parents=True, exist_ok=True)
    model.save(version_dir / "arima_model.pkl")
    save_metadata(version_dir, metadata)
    logger.info(f"Saved ARIMA model version {version_dir.name}")
    return version_dir


def save_lstm_model(model: torch.nn.Module, scaler, feature_cols: list, metadata: dict):
    ensure_model_directories()
    version_dir = LSTM_DIR / version_name()
    version_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), version_dir / "lstm_state.pth")
    with open(version_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(version_dir / "features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    save_metadata(version_dir, metadata)
    logger.info(f"Saved LSTM model version {version_dir.name}")
    return version_dir


def load_arima_model(version_dir: Path = None):
    if version_dir is None:
        version_dir = latest_version_path(ARIMA_DIR)
    if version_dir is None:
        return None, None
    model = AutoARIMA.load(version_dir / "arima_model.pkl")
    metadata = load_metadata(version_dir)
    return model, metadata


def load_lstm_model(version_dir: Path = None, device: str = "cpu"):
    if version_dir is None:
        version_dir = latest_version_path(LSTM_DIR)
    if version_dir is None:
        return None, None, None, None
    metadata = load_metadata(version_dir)
    feature_cols = pickle.loads((version_dir / "features.pkl").read_bytes())
    with open(version_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model_state = torch.load(version_dir / "lstm_state.pth", map_location=device)
    return model_state, scaler, feature_cols, metadata


def train_and_version_models(ticker: str, start_date: str, end_date: str, seq_length: int = 60, epochs: int = 50):
    logger.info(f"Retraining models for {ticker} from {start_date} to {end_date}")
    df_raw = fetch_data(ticker, start_date, end_date)
    df_clean = preprocess_data(df_raw)
    df_features = engineer_features(df_clean)

    train_size = int(len(df_features) * 0.8)
    train_data = df_features.iloc[:train_size]
    test_data = df_features.iloc[train_size:]

    arima_model, _ = train_arima(
        train_data['price'],
        test_data['price'],
        train_exog=train_data.drop(columns=['price']),
        test_exog=test_data.drop(columns=['price']),
        return_model=True
    )

    lstm_model, scaler, feature_cols, _, _, _ = train_lstm(
        df_features,
        train_size,
        seq_length=seq_length,
        epochs=epochs
    )

    metadata = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'seq_length': seq_length,
        'epochs': epochs,
        'features': feature_cols,
    }

    arima_dir = save_arima_model(arima_model, metadata)
    lstm_dir = save_lstm_model(lstm_model, scaler, feature_cols, metadata)

    return {
        'arima_version': arima_dir.name,
        'lstm_version': lstm_dir.name,
        'metadata': metadata,
    }


def get_latest_model_versions():
    return {
        'arima': latest_version_path(ARIMA_DIR).name if latest_version_path(ARIMA_DIR) else None,
        'lstm': latest_version_path(LSTM_DIR).name if latest_version_path(LSTM_DIR) else None,
    }


def create_lstm_sequences(df: pd.DataFrame, seq_length: int, feature_cols: list):
    data = df[feature_cols].values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def predict_with_latest_models(ticker: str, start_date: str, end_date: str, seq_length: int = 60):
    logger.info(f"Generating predictions for {ticker} using latest model versions")
    df_raw = fetch_data(ticker, start_date, end_date)
    df_clean = preprocess_data(df_raw)
    df_features = engineer_features(df_clean)

    train_size = int(len(df_features) * 0.8)
    test_data = df_features.iloc[train_size:]

    arima_model, arima_metadata = load_arima_model()
    if arima_model is None:
        raise RuntimeError("No saved ARIMA model found")
    arima_forecast = arima_model.predict(n_periods=len(test_data), exogenous=test_data.drop(columns=['price']))
    arima_forecast = pd.Series(arima_forecast, index=test_data.index)

    lstm_state, scaler, feature_cols, lstm_metadata = load_lstm_model()
    if lstm_state is None:
        raise RuntimeError("No saved LSTM model found")

    from src.models import LSTMModel
    model = LSTMModel(input_size=len(feature_cols), hidden_size=128, num_layers=2, output_size=1)
    model.load_state_dict(lstm_state)
    model.eval()

    X, y = create_lstm_sequences(df_features, seq_length, feature_cols)
    if len(X) == 0:
        raise ValueError("Not enough data for LSTM sequence generation")

    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).numpy().flatten()

    extended_preds = np.zeros((len(preds), len(feature_cols)))
    extended_preds[:, -1] = preds
    predictions_inv = scaler.inverse_transform(extended_preds)[:, -1]

    dates = df_features.index[seq_length:]
    prediction_dates = dates[train_size - seq_length:]
    predicted_series = pd.Series(predictions_inv[train_size - seq_length:], index=prediction_dates)

    return {
        'model_versions': get_latest_model_versions(),
        'arima': arima_forecast,
        'lstm': predicted_series,
        'test_actuals': test_data['price'],
    }
