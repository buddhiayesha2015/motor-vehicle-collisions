from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from nannyml.data_quality.missing import MissingValuesCalculator
from nannyml.data_quality.unseen import UnseenValuesCalculator
from nannyml.data_quality.range import NumericalRangeCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import PATHS, MLFLOW_SETTINGS

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


def main() -> None:
    reference = pd.read_csv(PATHS.cleaned_data)
    production = pd.read_csv(PATHS.drift_cleaned_data)

    reference["CRASH_TIMESTAMP"] = pd.to_datetime(reference["CRASH_DATETIME"])
    production["CRASH_TIMESTAMP"] = pd.to_datetime(production["CRASH_DATETIME"])

    model = load_model()
    feature_cols = [
        c for c in reference.columns if c not in {TARGET_COLUMN, "CRASH_TIMESTAMP"}
    ]
    reference["prediction"] = model.predict(reference[feature_cols])
    production["prediction"] = model.predict(production[feature_cols])

    # Using explicit square root keeps compatibility with older scikit-learn versions
    rmse_ref = sqrt(mean_squared_error(reference[TARGET_COLUMN], reference["prediction"]))
    rmse_prod = sqrt(mean_squared_error(production[TARGET_COLUMN], production["prediction"]))
    mae_prod = mean_absolute_error(production[TARGET_COLUMN], production["prediction"])
    r2_prod = r2_score(production[TARGET_COLUMN], production["prediction"])

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
        timestamp_column_name="CRASH_TIMESTAMP",
        column_names=feature_cols,
        categorical_drift_method="chi2",
        continuous_drift_method="kolmogorov_smirnov",
        chunk_size=5000,
    ).fit(reference)
    drift_res = drift_calc.calculate(production)

    sections = [
        "## Data Quality - Missing Values",
        mv_res.to_df().to_markdown(),
    ]
    if uv_res is not None:
        sections += ["## Data Quality - Unseen Values", uv_res.to_df().to_markdown()]
    if rng_res is not None:
        sections += ["## Data Quality - Numerical Ranges", rng_res.to_df().to_markdown()]
    sections += ["## Univariate Drift", drift_res.to_df().to_markdown()]

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Drift Analysis",
                "Reference window: 2012-2022",
                "Production window: 2023-2025",
                f"Reference RMSE: {rmse_ref:.3f}",
                f"Production RMSE: {rmse_prod:.3f}",
                f"Production MAE: {mae_prod:.3f}",
                f"Production R2: {r2_prod:.3f}",
                *sections,
            ]
        )
    )


if __name__ == "__main__":
    main()
