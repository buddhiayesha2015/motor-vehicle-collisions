from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import nannyml as nml
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import PATHS, MLFLOW_SETTINGS

REPORT_PATH = Path("artifacts/drift_report.md")


def load_model():
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)

    df = pd.read_csv("artifacts/model_comparison.csv")
    run_id = df.loc[df["model"] == "collisions_random_forest", "run_id"].iloc[0]

    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


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
