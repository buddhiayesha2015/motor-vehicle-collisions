from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h2o
import mlflow
import mlflow.h2o  # noqa: F401
import pandas as pd
from h2o.automl import H2OAutoML

from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import PATHS
from src.collisions.training_utils import setup_mlflow


def _pip_version_or_none() -> str | None:
    """Best-effort retrieval of the installed pip version for MLflow env specs."""

    try:
        return metadata.version("pip")
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    setup_mlflow()

    h2o.init()
    train_df = pd.read_csv(PATHS.train)
    validate_df = pd.read_csv(PATHS.validation)

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    train_h2o = h2o.H2OFrame(train_df)
    valid_h2o = h2o.H2OFrame(validate_df)

    automl_params = {"max_models": 10, "seed": 7, "sort_metric": "RMSE"}

    pip_version = _pip_version_or_none()
    conda_env = (
        mlflow.h2o.get_default_conda_env(pip_version=pip_version)
        if pip_version
        else None
    )

    def log_h2o_model(model, name: str) -> None:
        if conda_env:
            mlflow.h2o.log_model(model, name=name, conda_env=conda_env)
        else:
            mlflow.h2o.log_model(model, name=name)

    with mlflow.start_run(run_name="h2o_automl"):
        mlflow.log_params({"automl_" + k: v for k, v in automl_params.items()})
        aml = H2OAutoML(**automl_params)
        aml.train(x=feature_cols, y=TARGET_COLUMN, training_frame=train_h2o, validation_frame=valid_h2o)

        leaderboard = aml.leaderboard.as_data_frame()
        PATHS.automl_leaderboard.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(PATHS.automl_leaderboard, index=False)
        mlflow.log_artifact(PATHS.automl_leaderboard, artifact_path="automl")

        metric_columns = [c for c in leaderboard.columns if c != "model_id"]

        top_algos = leaderboard["model_id"].head(3).tolist()
        algo_types = []
        for model_id in top_algos:
            model = h2o.get_model(model_id)
            algo_types.append(model.algo)
        mlflow.log_text("\n".join(algo_types), artifact_file="automl/top_algos.txt")

        for _, row in leaderboard.iterrows():
            model_id = row["model_id"]
            model = h2o.get_model(model_id)
            with mlflow.start_run(
                run_name=f"{model.algo}_{model_id}", nested=True
            ):
                mlflow.log_params({"model_id": model_id, "algorithm": model.algo})
                metrics = {
                    col: float(row[col])
                    for col in metric_columns
                    if pd.notnull(row[col])
                }
                if metrics:
                    mlflow.log_metrics(metrics)
                log_h2o_model(model, name=f"automl_model_{model_id}")

        perf = aml.leader.model_performance(valid_h2o)
        mlflow.log_metrics({
            "leader_rmse": perf.rmse(),
            "leader_mae": perf.mae(),
            "leader_r2": perf.r2(),
        })
        log_h2o_model(aml.leader, name="automl_leader")
        print("Top three model types:", algo_types)


if __name__ == "__main__":
    main()
