from __future__ import annotations

import csv
import logging
from numbers import Number
from datetime import datetime
from pathlib import Path
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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

# Mapping used to locate run-based artifacts when the Model Registry lookup fails
MODEL_NAMES = {
    "model1": "collisions_random_forest",
    "model2": "collisions_gradient_boosting",
    "model3": "collisions_elastic_net",
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


def _lookup_run_uri(model_key: str) -> str | None:
    """Return a runs:/ URI from artifacts/model_comparison.csv if available."""

    comparison_path = Path("artifacts/model_comparison.csv")
    if not comparison_path.exists():
        return None

    model_name = MODEL_NAMES[model_key]
    with comparison_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") == model_name and row.get("run_id"):
                return f"runs:/{row['run_id']}/model"

    return None


def load_model(model_key: str):
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    if MLFLOW_SETTINGS.registry_uri:
        mlflow.set_registry_uri(MLFLOW_SETTINGS.registry_uri)

    primary_uri = MODEL_URIS[model_key]
    try:
        return mlflow.sklearn.load_model(primary_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model registry lookup failed for %s: %s", primary_uri, exc)

        fallback_uri = _lookup_run_uri(model_key)
        if fallback_uri:
            try:
                logger.info("Loading model %s from local artifacts at %s", model_key, fallback_uri)
                return mlflow.sklearn.load_model(fallback_uri)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load fallback model at %s", fallback_uri)

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load model. Ensure the MLflow Model Registry is reachable or "
                "artifacts/model_comparison.csv contains a valid run_id for the model."
            ),
        )


MODELS_CACHE: dict[str, object] = {}


def get_model(model_key: str):
    """Lazily load and cache models to avoid startup failures.

    When running the API locally without a populated MLflow Model Registry,
    attempting to eagerly load models at import time raises an exception and
    prevents the application from starting. By loading the model only when an
    endpoint is called, we can surface a clear HTTP error while still allowing
    the service to start for development and testing.
    """

    if model_key not in MODEL_URIS:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model_key}")

    if model_key not in MODELS_CACHE:
        MODELS_CACHE[model_key] = load_model(model_key)

    return MODELS_CACHE[model_key]


def predict(records: List[CollisionRecord], model_key: str) -> PredictionResponse:
    df = to_dataframe(records)
    model = get_model(model_key)
    predictions = model.predict(df).tolist()
    timestamp = datetime.utcnow().isoformat()
    logger.info("Model %s generated %d predictions", model_key, len(predictions))
    if predictions:
        preview = ", ".join(
            f"{p:.3f}" if isinstance(p, Number) else str(p) for p in predictions[:3]
        )
        if len(predictions) > 3:
            preview += "..."
        logger.info("Prediction preview: %s", preview)
    return PredictionResponse(
        model_name=model_key,
        model_version=getattr(model, "version", None),
        timestamp=timestamp,
        predictions=predictions,
    )


@app.get("/", summary="API status")
async def root():
    """Simple landing endpoint to avoid 404s at the root path."""

    return {
        "message": "Collision Injury Prediction API",
        "available_endpoints": {
            "predict_model1": "/predict_model1",
            "predict_model2": "/predict_model2",
            "predict_model3": "/predict_model3",
        },
    }


@app.get("/demo", response_class=HTMLResponse, summary="Interactive web demo")
async def demo_page() -> HTMLResponse:
    """Serve a small web UI that can call the prediction endpoints."""

    demo_html = (Path(__file__).with_name("demo.html")).read_text(encoding="utf-8")
    return HTMLResponse(content=demo_html)


@app.post("/predict_model1", response_model=PredictionResponse)
async def predict_model1(records: List[CollisionRecord]):
    return predict(records, "model1")


@app.post("/predict_model2", response_model=PredictionResponse)
async def predict_model2(records: List[CollisionRecord]):
    return predict(records, "model2")


@app.post("/predict_model3", response_model=PredictionResponse)
async def predict_model3(records: List[CollisionRecord]):
    return predict(records, "model3")
