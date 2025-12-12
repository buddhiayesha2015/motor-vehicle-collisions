from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h2o
import pandas as pd
from h2o.automl import H2OAutoML

from src.collisions.data_cleaning import TARGET_COLUMN
from src.collisions.settings import PATHS


def main() -> None:
    h2o.init()
    train_df = pd.read_csv(PATHS.train)
    validate_df = pd.read_csv(PATHS.validation)

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    train_h2o = h2o.H2OFrame(train_df)
    valid_h2o = h2o.H2OFrame(validate_df)

    aml = H2OAutoML(max_models=10, seed=7, sort_metric="RMSE")
    aml.train(x=feature_cols, y=TARGET_COLUMN, training_frame=train_h2o, validation_frame=valid_h2o)

    leaderboard = aml.leaderboard.as_data_frame()
    PATHS.automl_leaderboard.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(PATHS.automl_leaderboard, index=False)

    top_algos = leaderboard["model_id"].head(3).tolist()
    algo_types = []
    for model_id in top_algos:
        model = h2o.get_model(model_id)
        algo_types.append(model.algo)
    print("Top three model types:", algo_types)


if __name__ == "__main__":
    main()
