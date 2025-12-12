from __future__ import annotations

from pathlib import Path

import mlflow
import nannyml as nml
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import PATHS, MLFLOW_SETTINGS

MODEL_URI = "models:/collisions_random_forest/Production"
REPORT_PATH = Path("artifacts/drift_report.md")


def load_model():
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    if MLFLOW_SETTINGS.registry_uri:
        mlflow.set_registry_uri(MLFLOW_SETTINGS.registry_uri)
    return mlflow.sklearn.load_model(MODEL_URI)


def main() -> None:
    df = pd.read_csv(PATHS.cleaned_data)
    cutoff = int(len(df) * 0.85)
    reference = df.iloc[:cutoff].copy()
    production = df.iloc[cutoff:].copy()

    model = load_model()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    reference["prediction"] = model.predict(reference[feature_cols])
    production["prediction"] = model.predict(production[feature_cols])

    rmse_ref = mean_squared_error(reference[TARGET_COLUMN], reference["prediction"], squared=False)
    rmse_prod = mean_squared_error(production[TARGET_COLUMN], production["prediction"], squared=False)

    data_drift = nml.DataQualityCalculator(column_names=feature_cols)
    data_drift = data_drift.fit(reference)
    drift_report = data_drift.calculate(production)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Drift Analysis",
                f"Reference RMSE: {rmse_ref:.3f}",
                f"Production RMSE: {rmse_prod:.3f}",
                "\n## Data Quality Summary",
                drift_report.to_markdown(),
            ]
        )
    )


if __name__ == "__main__":
    main()
