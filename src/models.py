import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


def train_arima(train_data: pd.Series, test_data: pd.Series, train_exog: pd.DataFrame = None, test_exog: pd.DataFrame = None, return_model: bool = False):
    """Trains ARIMA/SARIMAX with optional exogenous variables."""
    print("Training ARIMA/SARIMAX with auto parameter selection...")
    if train_exog is not None and test_exog is not None:
        model = pm.auto_arima(
            train_data,
            exogenous=train_exog,
            start_p=0, start_q=0,
            max_p=5, max_q=5, max_d=2,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        forecast = model.predict(n_periods=len(test_data), exogenous=test_exog)
    else:
        model = pm.auto_arima(
            train_data,
            start_p=0, start_q=0,
            max_p=5, max_q=5, max_d=2,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        forecast = model.predict(n_periods=len(test_data))

    print(f"Selected ARIMA order: {model.order}")
    model.fit(train_data, exogenous=train_exog)
    forecast = pd.Series(forecast, index=test_data.index)

    if return_model:
        return model, forecast
    return forecast


def train_garch(returns: pd.Series, horizon: int = 5):
    """Fits a GARCH(1,1) volatility model and forecasts forward variance."""
    print("Training GARCH volatility model...")
    am = arch_model(returns.dropna() * 100, vol='Garch', p=1, q=1, dist='normal')
    fitted = am.fit(disp='off')
    forecast = fitted.forecast(horizon=horizon, reindex=False)
    var_forecast = forecast.variance.values[-1, :] / 10000.0
    vol_forecast = np.sqrt(var_forecast)
    return vol_forecast


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def create_sequences(data: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def train_lstm(df: pd.DataFrame, train_size: int, seq_length: int = 60, epochs: int = 50, batch_size: int = 32, hidden_size: int = 128):
    """Trains a multivariate LSTM model and returns prediction results."""
    print("Training LSTM...")
    feature_cols = [col for col in df.columns if col != 'price'] + ['price']
    print(f"Using features: {feature_cols}")

    data = df[feature_cols].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, seq_length)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    lstm_train_size = train_size - seq_length
    if lstm_train_size <= 0:
        raise ValueError("Train size too small for the given sequence length.")

    X_train, X_test = X[:lstm_train_size], X[lstm_train_size:]
    y_train, y_test = y[:lstm_train_size], y[lstm_train_size:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(X_train):.6f}')

    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch)
            predictions.extend(preds.cpu().numpy().flatten())

    predictions = np.array(predictions)
    reversed_preds = np.zeros((predictions.shape[0], len(feature_cols)))
    reversed_preds[:, -1] = predictions
    predictions_inv = scaler.inverse_transform(reversed_preds)[:, -1]

    actuals = df['price'].values[train_size:]
    dates = df.index[train_size:]
    return model, scaler, feature_cols, predictions_inv, actuals, dates
