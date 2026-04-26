import os
import argparse
import pandas as pd
from src.data import fetch_data, preprocess_data
from src.eda import perform_eda
from src.features import engineer_features
from src.models import train_arima, train_lstm, train_garch
from src.evaluation import (
    calculate_metrics,
    plot_results,
    compute_feature_importance,
    backtest_strategy,
    walk_forward_validation,
)


def main():
    parser = argparse.ArgumentParser(description="Finance-grade ARIMA + LSTM forecasting system")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2018-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for LSTM training')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for LSTM')
    parser.add_argument('--walk_splits', type=int, default=5, help='Number of walk-forward folds')
    args = parser.parse_args()

    print("=== Finance-grade Stock Forecasting System ===")
    print(f"Ticker: {args.ticker}")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"LSTM epochs: {args.epochs}, Sequence length: {args.seq_length}")
    print()

    print("1. Dataset Selection & Ingestion")
    df_raw = fetch_data(args.ticker, args.start, args.end)

    print("\n2. Data Preprocessing")
    df_clean = preprocess_data(df_raw)

    print("\n3. Exploratory Data Analysis")
    perform_eda(df_clean)

    print("\n4. Feature Engineering")
    df_features = engineer_features(df_clean)

    train_size = int(len(df_features) * 0.8)
    train_data = df_features.iloc[:train_size]
    test_data = df_features.iloc[train_size:]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    print("\n5. ARIMA Model Training")
    arima_forecast = train_arima(
        train_data['price'],
        test_data['price'],
        train_exog=train_data.drop(columns=['price']),
        test_exog=test_data.drop(columns=['price'])
    )

    print("\n6. LSTM Model Training")
    lstm_model, lstm_scaler, lstm_features, lstm_predictions, lstm_test_actuals, lstm_dates = train_lstm(
        df_features,
        train_size,
        seq_length=args.seq_length,
        epochs=args.epochs,
    )

    print("\n7. Feature Importance")
    feature_importance = compute_feature_importance(df_features, train_size)
    print(feature_importance.head(10).to_string())

    print("\n8. Volatility Modeling")
    volatility_forecast = train_garch(df_clean['log_returns'].iloc[:train_size], horizon=5)
    print(f"GARCH 5-day volatility forecast: {volatility_forecast}")

    print("\n9. Walk-forward Validation")
    wf_results = walk_forward_validation(df_features, seq_length=args.seq_length, epochs=args.epochs, n_splits=args.walk_splits)
    print(f"Completed {len(wf_results['arima'])} ARIMA folds and {len(wf_results['lstm'])} LSTM folds.")

    print("\n10. Final Evaluation")
    arima_metrics = calculate_metrics(test_data['price'], arima_forecast, "ARIMA")
    lstm_metrics = calculate_metrics(pd.Series(lstm_test_actuals, index=lstm_dates), pd.Series(lstm_predictions, index=lstm_dates), "LSTM")

    print("\n11. Backtesting")
    arima_backtest = backtest_strategy(test_data['price'], arima_forecast, signal_type='long_short')
    lstm_backtest = backtest_strategy(pd.Series(lstm_test_actuals, index=lstm_dates), pd.Series(lstm_predictions, index=lstm_dates), signal_type='long_short')

    print("\nARIMA Backtest Metrics:")
    print(f"Sharpe Ratio: {arima_backtest['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {arima_backtest['max_drawdown']:.4f}")
    print(f"Cumulative Return: {arima_backtest['cumulative_return']:.4f}")

    print("\nLSTM Backtest Metrics:")
    print(f"Sharpe Ratio: {lstm_backtest['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {lstm_backtest['max_drawdown']:.4f}")
    print(f"Cumulative Return: {lstm_backtest['cumulative_return']:.4f}")

    print("\n12. Model Comparison")
    print(f"RMSE - ARIMA: {arima_metrics['rmse']:.4f}, LSTM: {lstm_metrics['rmse']:.4f}")
    print(f"MAE - ARIMA: {arima_metrics['mae']:.4f}, LSTM: {lstm_metrics['mae']:.4f}")
    print(f"MAPE - ARIMA: {arima_metrics['mape']:.2f}%, LSTM: {lstm_metrics['mape']:.2f}%")

    if arima_metrics['rmse'] < lstm_metrics['rmse']:
        print("ARIMA performed better in terms of RMSE.")
    else:
        print("LSTM performed better in terms of RMSE.")

    print("\n13. Visualization")
    plot_results(train_data, test_data, arima_forecast, lstm_predictions, lstm_dates)

    print("\n=== Pipeline execution completed successfully ===")


if __name__ == "__main__":
    main()
