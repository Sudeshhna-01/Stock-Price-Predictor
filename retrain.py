import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from src.logging_config import logger
from src.pipeline import train_and_version_models

TICKER = os.getenv("RETRAIN_TICKER", "AAPL")
START_DATE = os.getenv("RETRAIN_START", "2018-01-01")
END_DATE = os.getenv("RETRAIN_END", "2023-12-31")
SEQ_LENGTH = int(os.getenv("RETRAIN_SEQ_LENGTH", "60"))
EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "50"))
RETRAIN_HOUR = int(os.getenv("RETRAIN_HOUR", "2"))


def retrain_pipeline():
    logger.info("Starting scheduled retraining pipeline")
    try:
        result = train_and_version_models(
            TICKER,
            START_DATE,
            END_DATE,
            seq_length=SEQ_LENGTH,
            epochs=EPOCHS,
        )
        logger.info(f"Retraining completed: {result}")
    except Exception as exc:
        logger.exception("Retraining pipeline failed")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(retrain_pipeline, "cron", hour=RETRAIN_HOUR)
    logger.info(f"Scheduled retraining job at hour {RETRAIN_HOUR} UTC")
    retrain_pipeline()
    scheduler.start()
