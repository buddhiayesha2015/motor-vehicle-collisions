from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from nannyml.data_quality.missing import MissingValuesCalculator
from nannyml.data_quality.range import NumericalRangeCalculator
from nannyml.data_quality.unseen import UnseenValuesCalculator
from nannyml.drift import UnivariateDriftCalculator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collisions.data_cleaning import (
    TARGET_COLUMN,
    clamp_coordinates,
    convert_datetime,
    encode_categoricals,
    handle_missing_values,
)
from src.collisions.settings import MLFLOW_SETTINGS, PATHS

REPORT_PATH = Path("artifacts/drift_report.md")


MODEL_NAME = "collisions_random_forest"


def _load_latest_registered_model(client: MlflowClient):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None

    latest = max(versions, key=lambda v: int(v.version))
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")


def load_model():
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    if MLFLOW_SETTINGS.resolved_registry_uri:
        mlflow.set_registry_uri(MLFLOW_SETTINGS.resolved_registry_uri)

    client = MlflowClient()

    return _load_latest_registered_model(client)


def load_full_dataset() -> pd.DataFrame:
    df = pd.read_csv(PATHS.raw_data)
    df = convert_datetime(df)
    df = clamp_coordinates(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = df.drop_duplicates(subset=["COLLISION_ID"])
    df = df.sort_values("CRASH_DATETIME").reset_index(drop=True)
    return df


def split_reference_production(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = df[(df["CRASH_YEAR"] >= 2012) & (df["CRASH_YEAR"] <= 2022)].copy()
    production = df[(df["CRASH_YEAR"] >= 2023) & (df["CRASH_YEAR"] <= 2025)].copy()

    if reference.empty or production.empty:
        raise ValueError("Both reference (2012-2022) and production (2023-2025) splits must contain data.")

    return reference, production


def main() -> None:
    df = load_full_dataset()
    reference, production = split_reference_production(df)

    model = load_model()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    reference["prediction"] = model.predict(reference[feature_cols])
    production["prediction"] = model.predict(production[feature_cols])

    # Using explicit square root keeps compatibility with older scikit-learn versions
    rmse_ref = sqrt(mean_squared_error(reference[TARGET_COLUMN], reference["prediction"]))
    rmse_prod = sqrt(mean_squared_error(production[TARGET_COLUMN], production["prediction"]))

    # DataQualityCalculator is exposed via the data_quality module in current NannyML versions
    cat_cols = reference[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    mv_res = MissingValuesCalculator(column_names=feature_cols).fit(reference).calculate(production)

    uv_res = None
    if cat_cols:
        uv_res = UnseenValuesCalculator(column_names=cat_cols).fit(reference).calculate(production)

    rng_res = None
    if num_cols:
        rng_res = NumericalRangeCalculator(column_names=num_cols).fit(reference).calculate(production)

    drift_calc = UnivariateDriftCalculator(
        column_names=feature_cols,
        timestamp_column_name="CRASH_DATETIME",
        treat_as_categorical=cat_cols,
        continuous_methods=["kolmogorov_smirnov"],
        categorical_methods=["chi2"],
    )
    drift_results = drift_calc.fit(reference).calculate(production)

    performance_section = [
        "## Model Performance",
        f"Reference RMSE: {rmse_ref:.3f}",
        f"Production RMSE: {rmse_prod:.3f}",
        f"Reference MAE: {mean_absolute_error(reference[TARGET_COLUMN], reference['prediction']):.3f}",
        f"Production MAE: {mean_absolute_error(production[TARGET_COLUMN], production['prediction']):.3f}",
        f"Reference R2: {r2_score(reference[TARGET_COLUMN], reference['prediction']):.3f}",
        f"Production R2: {r2_score(production[TARGET_COLUMN], production['prediction']):.3f}",
    ]

    sections = [
        "# Drift Analysis",
        "## Data Windows",
        f"Reference period: 2012-01-01 to 2022-12-31 ({len(reference)} rows)",
        f"Production period: 2023-01-01 to 2025-12-31 ({len(production)} rows)",
        *performance_section,
        "## Data Quality - Missing Values",
        mv_res.to_df().to_markdown(),
    ]

    if uv_res is not None:
        sections += ["## Data Quality - Unseen Values", uv_res.to_df().to_markdown()]
    if rng_res is not None:
        sections += ["## Data Quality - Numerical Ranges", rng_res.to_df().to_markdown()]

    sections += [
        "## Data Drift (Univariate)",
        drift_results.to_df().to_markdown(),
    ]

    REPORT_PATH.write_text("\n".join(sections))


if __name__ == "__main__":
    main()
