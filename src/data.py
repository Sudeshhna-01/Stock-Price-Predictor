import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches price and volume data from Yahoo Finance."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    optional_cols = ['Adj Close']
    available_cols = [col for col in required_cols + optional_cols if col in data.columns]

    df = data[available_cols].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the dataframe by handling missing values and generating return/volatility series."""
    print("Preprocessing data...")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    df = df.resample('B').ffill()
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])

    df.dropna(inplace=True)
    print(f"Dataset shape after preprocessing: {df.shape}")
    return df
