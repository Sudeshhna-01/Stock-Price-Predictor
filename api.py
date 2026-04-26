from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.logging_config import logger
from src.pipeline import (
    get_latest_model_versions,
    predict_with_latest_models,
    train_and_version_models,
)

app = FastAPI(
    title="Stock Price Forecasting API",
    description="Real-time prediction API for finance-grade ARIMA + LSTM models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_request_count = 0
prediction_count = 0
retrain_count = 0


class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    start_date: str = Field(..., example="2018-01-01")
    end_date: str = Field(..., example="2023-12-31")
    seq_length: int = Field(60, gt=10, lt=180, example=60)


class RetrainRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    start_date: str = Field(..., example="2018-01-01")
    end_date: str = Field(..., example="2023-12-31")
    seq_length: int = Field(60, gt=10, lt=180, example=60)
    epochs: int = Field(50, gt=1, lt=200, example=50)


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/version")
def version():
    return {"api_version": "1.0.0", "model_versions": get_latest_model_versions()}


@app.post("/predict")
def predict(request: PredictionRequest):
    global api_request_count, prediction_count
    api_request_count += 1

    try:
        prediction_data = predict_with_latest_models(
            request.ticker,
            request.start_date,
            request.end_date,
            seq_length=request.seq_length,
        )
    except Exception as exc:
        logger.error("Prediction failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))

    prediction_count += 1
    return {
        "model_versions": prediction_data["model_versions"],
        "arima_forecast": prediction_data["arima"].to_dict(),
        "lstm_forecast": prediction_data["lstm"].to_dict(),
        "test_actuals": prediction_data["test_actuals"].to_dict(),
    }


@app.post("/retrain")
def retrain(request: RetrainRequest):
    global retrain_count
    retrain_count += 1

    try:
        result = train_and_version_models(
            request.ticker,
            request.start_date,
            request.end_date,
            seq_length=request.seq_length,
            epochs=request.epochs,
        )
    except Exception as exc:
        logger.error("Retraining failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "completed", "model_versions": result}


@app.get("/metrics")
def metrics():
    return {
        "api_request_count": api_request_count,
        "prediction_count": prediction_count,
        "retrain_count": retrain_count,
        "model_versions": get_latest_model_versions(),
    }
