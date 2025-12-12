from __future__ import annotations

import mlflow
import pandas as pd

from src.collisions.settings import MLFLOW_SETTINGS

MODELS = [
    "collisions_random_forest",
    "collisions_gradient_boosting",
    "collisions_elastic_net",
]


def fetch_metrics(model_name: str) -> dict:
    client = mlflow.tracking.MlflowClient()
    experiments = client.get_experiment_by_name(MLFLOW_SETTINGS.experiment_name)
    if not experiments:
        return {"model": model_name}
    runs = client.search_runs(experiments.experiment_id, filter_string=f"tags.mlflow.log-model.history.model_name = '{model_name}'")
    if not runs:
        return {"model": model_name}
    best = sorted(runs, key=lambda r: r.data.metrics.get("val_rmse", float("inf")))[0]
    return {
        "model": model_name,
        "run_id": best.info.run_id,
        "val_rmse": best.data.metrics.get("val_rmse"),
        "val_mae": best.data.metrics.get("val_mae"),
        "val_r2": best.data.metrics.get("val_r2"),
        "test_rmse": best.data.metrics.get("test_rmse"),
        "test_mae": best.data.metrics.get("test_mae"),
        "test_r2": best.data.metrics.get("test_r2"),
    }


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    rows = [fetch_metrics(name) for name in MODELS]
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("artifacts/model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
