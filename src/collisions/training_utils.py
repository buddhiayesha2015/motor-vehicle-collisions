from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
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
    return load_split(PATHS.train), load_split(PATHS.validation), load_split(PATHS.test)


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categories into an "Other" bucket to cap OHE size."""

    def __init__(self, top_n: int = 50, min_frequency: int | float | None = None):
        self.top_n = top_n
        self.min_frequency = min_frequency
        self.frequent_categories_: dict[str, set] = {}
        self.columns_: list[str] = []

    def fit(self, X, y=None):  # noqa: ANN001
        df = self._to_frame(X)
        self.columns_ = list(df.columns)
        for col in self.columns_:
            counts = df[col].value_counts(dropna=False)
            top = counts
            if self.min_frequency is not None:
                threshold = self.min_frequency
                if isinstance(threshold, float):
                    threshold = max(int(len(df) * threshold), 1)
                top = top[top >= threshold]
            top = top.head(self.top_n)
            self.frequent_categories_[col] = set(top.index.tolist())
        return self

    def transform(self, X):  # noqa: ANN001
        df = self._to_frame(X)
        for col in self.columns_:
            frequent = self.frequent_categories_.get(col, set())
            df[col] = df[col].where(df[col].isin(frequent), other="Other")
        return df

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:  # noqa: ANN001, N802
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def build_preprocessor(one_hot_sparse: bool = True, max_categories: int = 50) -> ColumnTransformer:
    numeric_features = [
        col for col in NUMERIC_COLUMNS if col != TARGET_COLUMN
    ] + ["CRASH_HOUR", "CRASH_MONTH", "CRASH_YEAR"]
    categorical_features = CATEGORICAL_COLUMNS

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ohe_kwargs = {"handle_unknown": "ignore"}
    # Support both new (sparse_output) and legacy (sparse) kwargs for OneHotEncoder
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        ohe_kwargs["sparse_output"] = one_hot_sparse
    else:
        ohe_kwargs["sparse"] = one_hot_sparse

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare_grouper", RareCategoryGrouper(top_n=max_categories)),
        ("encoder", OneHotEncoder(**ohe_kwargs)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )
    return preprocessor


def evaluate_regression(model, X, y, split_name: str) -> Dict[str, float]:
    preds = model.predict(X)

    if "squared" in mean_squared_error.__code__.co_varnames:
        rmse = mean_squared_error(y, preds, squared=False)
    else:
        rmse = float(np.sqrt(mean_squared_error(y, preds)))

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
