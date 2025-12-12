from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collisions import training_utils

MODEL_NAME = "collisions_gradient_boosting"


def main() -> None:
    training_utils.setup_mlflow()
    train_df, validate_df, test_df = training_utils.load_datasets()
    X_train, y_train = training_utils.feature_target_split(train_df)
    X_val, y_val = training_utils.feature_target_split(validate_df)
    X_test, y_test = training_utils.feature_target_split(test_df)

    preprocessor = training_utils.build_preprocessor()
    model = HistGradientBoostingRegressor(max_depth=10, learning_rate=0.05, random_state=7)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "random_state": model.random_state,
        })
        pipeline.fit(X_train, y_train)

        val_metrics = training_utils.evaluate_regression(pipeline, X_val, y_val, "val")
        test_metrics = training_utils.evaluate_regression(pipeline, X_test, y_test, "test")
        for k, v in {**val_metrics, **test_metrics}.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=MODEL_NAME)
        run = mlflow.active_run()
        if run:
            model_uri = f"runs:/{run.info.run_id}/model"
            version = training_utils.register_model(model_uri, MODEL_NAME)
            mlflow.log_param("registered_version", version)


if __name__ == "__main__":
    main()
