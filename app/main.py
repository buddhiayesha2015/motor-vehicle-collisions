from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.collisions.data_cleaning import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.collisions.settings import MLFLOW_SETTINGS, PATHS
from .schemas import CollisionRecord, PredictionResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Collision Injury Prediction API")

MODEL_URIS = {
    "model1": "models:/collisions_random_forest/Production",
    "model2": "models:/collisions_gradient_boosting/Production",
    "model3": "models:/collisions_elastic_net/Production",
}


def to_dataframe(records: List[CollisionRecord]) -> pd.DataFrame:
    data = []
    for record in records:
        row = {
            "CRASH DATE": record.crash_date.strftime("%Y-%m-%d"),
            "CRASH TIME": record.crash_time.strftime("%H:%M"),
            "BOROUGH": (record.borough or "").title(),
            "ZIP CODE": record.zip_code or "",
            "LATITUDE": record.latitude,
            "LONGITUDE": record.longitude,
            "ON STREET NAME": (record.on_street_name or "").title(),
            "CROSS STREET NAME": (record.cross_street_name or "").title(),
            "OFF STREET NAME": (record.off_street_name or "").title(),
            "CONTRIBUTING FACTOR VEHICLE 1": (record.contributing_factor_vehicle_1 or "").title(),
            "VEHICLE TYPE CODE 1": (record.vehicle_type_code_1 or "").title(),
            "NUMBER OF PERSONS KILLED": record.number_of_persons_killed,
            "NUMBER OF PEDESTRIANS INJURED": record.number_of_pedestrians_injured,
            "NUMBER OF PEDESTRIANS KILLED": record.number_of_pedestrians_killed,
            "NUMBER OF CYCLIST INJURED": record.number_of_cyclist_injured,
            "NUMBER OF CYCLIST KILLED": record.number_of_cyclist_killed,
            "NUMBER OF MOTORIST INJURED": record.number_of_motorist_injured,
            "NUMBER OF MOTORIST KILLED": record.number_of_motorist_killed,
        }
        df_row = pd.DataFrame([row])
        df_row["CRASH_DATETIME"] = pd.to_datetime(df_row["CRASH DATE"] + " " + df_row["CRASH TIME"], errors="coerce")
        df_row["CRASH_HOUR"] = df_row["CRASH_DATETIME"].dt.hour
        df_row["CRASH_MONTH"] = df_row["CRASH_DATETIME"].dt.month
        df_row["CRASH_YEAR"] = df_row["CRASH_DATETIME"].dt.year
        data.append(df_row.drop(columns=["CRASH_DATETIME"]))
    return pd.concat(data, ignore_index=True)


def load_model(uri: str):
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    if MLFLOW_SETTINGS.registry_uri:
        mlflow.set_registry_uri(MLFLOW_SETTINGS.registry_uri)
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unable to load model %s", uri)
        raise HTTPException(status_code=500, detail=str(exc))


MODELS_CACHE = {key: load_model(uri) for key, uri in MODEL_URIS.items()}


def predict(records: List[CollisionRecord], model_key: str) -> PredictionResponse:
    df = to_dataframe(records)
    model = MODELS_CACHE[model_key]
    predictions = model.predict(df).tolist()
    timestamp = datetime.utcnow().isoformat()
    logger.info("Model %s generated %d predictions", model_key, len(predictions))
    return PredictionResponse(
        model_name=model_key,
        model_version=getattr(model, "version", None),
        timestamp=timestamp,
        predictions=predictions,
    )


@app.post("/predict_model1", response_model=PredictionResponse)
async def predict_model1(records: List[CollisionRecord]):
    return predict(records, "model1")


@app.post("/predict_model2", response_model=PredictionResponse)
async def predict_model2(records: List[CollisionRecord]):
    return predict(records, "model2")


@app.post("/predict_model3", response_model=PredictionResponse)
async def predict_model3(records: List[CollisionRecord]):
    return predict(records, "model3")
