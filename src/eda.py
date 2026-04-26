import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import warnings
warnings.filterwarnings("ignore")

def perform_eda(df: pd.DataFrame, output_dir="output"):
    """Performs Exploratory Data Analysis on the time series data."""
    print("Performing Exploratory Data Analysis...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Basic statistics
    print("Basic Statistics:")
    print(df.describe())

    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['price'], label='Stock Price')
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"{output_dir}/time_series.png")
    plt.close()

    # Decomposition
    try:
        decomposition = seasonal_decompose(df['price'], model='additive', period=252)  # 252 trading days in a year
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/decomposition.png")
        plt.close()
    except:
        print("Decomposition failed - possibly insufficient data or non-seasonal pattern")

    # Stationarity tests
    perform_stationarity_tests(df['price'], output_dir)

    # ACF and PACF plots
    plot_acf_pacf(df['price'], output_dir)

def perform_stationarity_tests(series: pd.Series, output_dir):
    """Performs ADF and KPSS tests for stationarity."""
    print("\nStationarity Tests:")

    # ADF Test
    result_adf = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result_adf[0]}')
    print(f'p-value: {result_adf[1]}')
    print(f'Critical Values:')
    for key, value in result_adf[4].items():
        print(f'\t{key}: {value}')
    if result_adf[1] <= 0.05:
        print("ADF: Series is stationary")
    else:
        print("ADF: Series is non-stationary")

    # KPSS Test
    result_kpss = kpss(series, regression='c', nlags="auto")
    print(f'\nKPSS Statistic: {result_kpss[0]}')
    print(f'p-value: {result_kpss[1]}')
    print(f'Critical Values:')
    for key, value in result_kpss[3].items():
        print(f'\t{key}: {value}')
    if result_kpss[1] <= 0.05:
        print("KPSS: Series is non-stationary")
    else:
        print("KPSS: Series is stationary")

def plot_acf_pacf(series: pd.Series, output_dir):
    """Plots ACF and PACF for ARIMA parameter selection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series, ax=ax1, lags=40)
    ax1.set_title('Autocorrelation Function (ACF)')
    plot_pacf(series, ax=ax2, lags=40)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f"{output_dir}/acf_pacf.png")
    plt.close()