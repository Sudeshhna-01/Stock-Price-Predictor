import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from src.models import train_arima, train_lstm


def calculate_metrics(y_true, y_pred, model_name=None):
    """Calculates regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    if model_name:
        print(f"--- {model_name} Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%\n")
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    returns = returns.dropna()
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / (annual_volatility + 1e-9)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'cumulative_return': cumulative.iloc[-1]
    }


def compute_feature_importance(df: pd.DataFrame, train_size: int) -> pd.Series:
    features = df.drop(columns=['price'])
    target = df['price']
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(features.iloc[:train_size], target.iloc[:train_size])
    importances = pd.Series(model.feature_importances_, index=features.columns)
    return importances.sort_values(ascending=False)


def backtest_strategy(actual: pd.Series, predicted: pd.Series, signal_type: str = 'long_only') -> dict:
    actual = actual.copy()
    predicted = predicted.copy()
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual, index=predicted.index)
    if not isinstance(predicted, pd.Series):
        predicted = pd.Series(predicted, index=actual.index)

    returns = actual.pct_change()
    if signal_type == 'long_only':
        signal = (predicted > actual).astype(int)
    elif signal_type == 'long_short':
        signal = np.where(predicted > actual, 1, -1)
    else:
        raise ValueError("signal_type must be 'long_only' or 'long_short'")

    signal = pd.Series(signal, index=actual.index)
    strategy_returns = returns.shift(-1) * signal
    strategy_returns = strategy_returns.dropna()
    metrics = calculate_risk_metrics(strategy_returns)
    metrics['strategy_returns'] = strategy_returns
    metrics['cumulative_returns'] = (1 + strategy_returns).cumprod()
    return metrics


def walk_forward_validation(df: pd.DataFrame, seq_length: int = 60, epochs: int = 50, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    arima_fold_metrics = []
    lstm_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if len(train) < seq_length or len(test) == 0:
            continue

        print(f"Walk-forward fold {fold}: train={len(train)}, test={len(test)}")

        arima_forecast = train_arima(
            train['price'],
            test['price'],
            train_exog=train.drop(columns=['price']),
            test_exog=test.drop(columns=['price'])
        )
        arima_metrics = calculate_metrics(test['price'], arima_forecast, f"ARIMA Fold {fold}")
        arima_fold_metrics.append(arima_metrics)

        _, _, _, lstm_preds, lstm_actuals, lstm_dates = train_lstm(
            df,
            train_size=len(train),
            seq_length=seq_length,
            epochs=epochs
        )
        lstm_series = pd.Series(lstm_preds, index=lstm_dates)
        lstm_actual_series = pd.Series(lstm_actuals, index=lstm_dates)
        lstm_fold_metrics.append(calculate_metrics(lstm_actual_series, lstm_series, f"LSTM Fold {fold}"))

    return {
        'arima': arima_fold_metrics,
        'lstm': lstm_fold_metrics
    }


def plot_results(train_data, test_data, arima_forecast, lstm_predictions, lstm_dates, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(16, 8))
    plt.plot(train_data.index, train_data['price'], label='Train Data', color='blue')
    plt.plot(test_data.index, test_data['price'], label='Actual Test Data', color='black')
    plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.plot(lstm_dates, lstm_predictions, label='LSTM Forecast', color='green')
    plt.title('Stock Price Prediction: ARIMA vs LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'forecast_comparison.png')
    plt.savefig(plot_path)
    print(f"Plot saved successfully to {plot_path}")
    plt.close()
