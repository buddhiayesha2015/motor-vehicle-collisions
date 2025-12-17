from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collisions import training_utils
from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import MLFLOW_SETTINGS


MODEL_NAME = "collisions_random_forest"


def main() -> None:
    training_utils.setup_mlflow()
    train_df, validate_df, test_df = training_utils.load_datasets()
    X_train, y_train = training_utils.feature_target_split(train_df)
    X_val, y_val = training_utils.feature_target_split(validate_df)
    X_test, y_test = training_utils.feature_target_split(test_df)

    preprocessor = training_utils.build_preprocessor()
    model = RandomForestRegressor(n_estimators=200, random_state=7, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({"n_estimators": model.n_estimators, "random_state": model.random_state})
        pipeline.fit(X_train, y_train)

        val_metrics = training_utils.evaluate_regression(pipeline, X_val, y_val, "val")
        test_metrics = training_utils.evaluate_regression(pipeline, X_test, y_test, "test")
        for k, v in {**val_metrics, **test_metrics}.items():
            mlflow.log_metric(k, v)

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        training_utils.save_feature_metadata(feature_names.tolist())
        importance_path = Path("artifacts/random_forest_importance.png")
        training_utils.log_feature_importance(model, feature_names.tolist(), importance_path)
        if importance_path.exists():
            mlflow.log_artifact(str(importance_path))

        mlflow.sklearn.log_model(sk_model=pipeline, name="random_forest", registered_model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
