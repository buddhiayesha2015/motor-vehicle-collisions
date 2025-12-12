from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_cleaning import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from .settings import MLFLOW_SETTINGS, PATHS


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_split(PATHS.train), load_split(PATHS.validate), load_split(PATHS.test)


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_features = NUMERIC_COLUMNS + ["CRASH_HOUR", "CRASH_MONTH", "CRASH_YEAR"]
    categorical_features = CATEGORICAL_COLUMNS

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_regression(model, X, y, split_name: str) -> Dict[str, float]:
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    metrics = {f"{split_name}_rmse": rmse, f"{split_name}_mae": mae, f"{split_name}_r2": r2}
    return metrics


def log_feature_importance(model, feature_names: list[str], artifact_path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1])
        plt.xlabel("Importance")
        plt.tight_layout()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(artifact_path)
        plt.close()


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    if MLFLOW_SETTINGS.registry_uri:
        mlflow.set_registry_uri(MLFLOW_SETTINGS.registry_uri)
    mlflow.set_experiment(MLFLOW_SETTINGS.experiment_name)


def register_model(model_uri: str, name: str) -> str:
    result = mlflow.register_model(model_uri=model_uri, name=name)
    return result.version


def save_feature_metadata(feature_names: list[str]) -> None:
    metadata = json.loads(PATHS.feature_metadata.read_text()) if PATHS.feature_metadata.exists() else {}
    metadata["model_feature_names"] = feature_names
    PATHS.feature_metadata.write_text(json.dumps(metadata, indent=2))
