from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from nannyml.data_quality.missing import MissingValuesCalculator
from nannyml.data_quality.unseen import UnseenValuesCalculator
from nannyml.data_quality.range import NumericalRangeCalculator
from math import sqrt
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

    # Using explicit square root keeps compatibility with older scikit-learn versions
    rmse_ref = sqrt(mean_squared_error(reference[TARGET_COLUMN], reference["prediction"]))
    rmse_prod = sqrt(mean_squared_error(production[TARGET_COLUMN], production["prediction"]))

    # DataQualityCalculator is exposed via the data_quality module in current NannyML versions
    cat_cols = reference[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    mv_res = MissingValuesCalculator(column_names=feature_cols).fit(reference).calculate(production)

    uv_res = None
    if cat_cols:
        uv_res = UnseenValuesCalculator(column_names=cat_cols).fit(reference).calculate(
            production)  # categorical only :contentReference[oaicite:4]{index=4}

    rng_res = None
    if num_cols:
        rng_res = NumericalRangeCalculator(column_names=num_cols).fit(reference).calculate(production)

    sections = [
        "## Data Quality - Missing Values",
        mv_res.to_df().to_markdown(),
    ]
    if uv_res is not None:
        sections += ["## Data Quality - Unseen Values", uv_res.to_df().to_markdown()]
    if rng_res is not None:
        sections += ["## Data Quality - Numerical Ranges", rng_res.to_df().to_markdown()]

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Drift Analysis",
                f"Reference RMSE: {rmse_ref:.3f}",
                f"Production RMSE: {rmse_prod:.3f}",
                *sections,
            ]
        )
    )


if __name__ == "__main__":
    main()
