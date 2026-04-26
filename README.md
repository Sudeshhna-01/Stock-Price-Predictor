# Stock Price Forecasting System

A production-grade time series forecasting system that compares ARIMA and LSTM models for stock price prediction.

## Features

- **Data Collection**: Automated stock data fetching from Yahoo Finance
- **Data Preprocessing**: Handling missing values, datetime indexing, and business day resampling
- **Exploratory Data Analysis**: Trend analysis, seasonality decomposition, stationarity tests (ADF, KPSS), ACF/PACF plots
- **Feature Engineering**: Lag features, rolling statistics, exponential moving averages, volatility measures
- **ARIMA Modeling**: Automatic parameter selection using pmdarima
- **LSTM Modeling**: Bidirectional LSTM with multiple features and early stopping
- **Model Evaluation**: RMSE, MAE, MAPE metrics with visualization
- **Model Comparison**: Side-by-side performance analysis

## Project Structure

```
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── src/
│   ├── data.py            # Data ingestion and preprocessing
│   ├── eda.py             # Exploratory data analysis
│   ├── features.py        # Feature engineering
│   ├── models.py          # ARIMA and LSTM model training
│   └── evaluation.py      # Model evaluation and visualization
├── output/                # Generated plots and results
└── venv/                  # Virtual environment
```

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface
```bash
python main.py
```

### Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to access the interactive web interface.

### Custom Parameters (Command Line)
```bash
python main.py --ticker TSLA --start 2020-01-01 --end 2024-01-01 --epochs 100 --seq_length 30
```

### Parameters
- `--ticker`: Stock ticker symbol (default: AAPL)
- `--start`: Start date (YYYY-MM-DD, default: 2018-01-01)
- `--end`: End date (YYYY-MM-DD, default: 2023-01-01)
- `--epochs`: Number of LSTM training epochs (default: 50)
- `--seq_length`: Sequence length for LSTM (default: 60)

## Streamlit Web Interface

The interactive web interface provides an intuitive way to use the forecasting system:

### Features
- **Interactive Parameter Selection**: Choose stock ticker, date range, and model parameters
- **Model Selection**: Run ARIMA, LSTM, or both models
- **Real-time Results**: View model performance metrics and comparisons
- **Interactive Visualizations**: Explore forecasts, time series analysis, and decomposition
- **Download Results**: Export prediction data as CSV files
- **Progress Tracking**: Monitor the analysis pipeline in real-time

### Interface Sections
1. **Configuration Sidebar**: Set all parameters and run the analysis
2. **Results Dashboard**: View metrics, comparisons, and key insights
3. **Visualization Gallery**: Interactive plots and charts
4. **Data Export**: Download raw predictions and analysis results

### Running the Web App
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.

## Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to `app.py`
5. Click Deploy

### Other Deployment Options
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment guides including:
- Heroku deployment
- Docker deployment
- Local server setup
- Production optimizations

## Methodology

### 1. Dataset Selection
- Uses Yahoo Finance API for reliable, real-time stock data
- Supports any publicly traded stock ticker
- Handles business day resampling and missing data

### 2. Data Preprocessing
- Datetime indexing with business day frequency
- Forward-fill missing values
- Outlier handling through statistical methods

### 3. Exploratory Data Analysis
- **Time Series Plot**: Visual inspection of price trends
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Stationarity Tests**:
  - Augmented Dickey-Fuller (ADF) test
  - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
- **ACF/PACF Analysis**: Autocorrelation and partial autocorrelation plots

### 4. Feature Engineering
- **Lag Features**: Price lags (1, 2, 3, 7, 14, 30 days)
- **Rolling Statistics**: Mean, std, min, max over multiple windows
- **Exponential Moving Averages**: EMA with 7 and 14-day spans
- **Percentage Changes**: Daily and weekly returns
- **Volatility Measures**: Rolling standard deviation of returns

### 5. ARIMA Model
- **Automatic Parameter Selection**: Uses pmdarima for optimal (p,d,q) parameters
- **Parameter Range**: p=0-5, d=0-2, q=0-5
- **Non-seasonal Model**: Focuses on short-term forecasting

### 6. LSTM Model
- **Architecture**: Bidirectional LSTM with dropout regularization
- **Sequence Length**: Configurable (default: 60 days)
- **Multiple Features**: Uses all engineered features as input
- **Early Stopping**: Prevents overfitting
- **Bidirectional Processing**: Captures both forward and backward dependencies

### 7. Model Evaluation
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
- **Visualization**: Actual vs predicted plots

### 8. Model Comparison
- Side-by-side performance metrics
- Insights into model strengths and weaknesses
- Recommendations based on results

## Output Files

All visualizations and results are saved in the `output/` directory:
- `time_series.png`: Original time series plot
- `decomposition.png`: Seasonal decomposition
- `acf_pacf.png`: Autocorrelation plots
- `forecast_comparison.png`: Model comparison visualization

## Dependencies

- yfinance: Stock data fetching
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Plotting
- seaborn: Statistical visualization
- statsmodels: Statistical modeling and tests
- scikit-learn: Machine learning utilities
- tensorflow: Deep learning framework
- pmdarima: ARIMA parameter selection

## Production Considerations

- **Scalability**: Modular design allows easy extension to multiple stocks
- **Error Handling**: Robust error handling for API failures and data issues
- **Logging**: Comprehensive logging for debugging and monitoring
- **Configuration**: Command-line arguments for easy parameter tuning
- **Reproducibility**: Fixed random seeds and deterministic preprocessing

## Future Enhancements

- **Seasonal ARIMA (SARIMA)**: For stocks with strong seasonal patterns
- **Prophet Model**: Facebook's forecasting model for comparison
- **Ensemble Methods**: Combining multiple models for better performance
- **Real-time Prediction**: Live prediction capabilities
- **Web Dashboard**: Interactive visualization interface
- **Backtesting Framework**: Comprehensive model validation

## License

This project is for educational and research purposes. Please respect API usage limits and terms of service.