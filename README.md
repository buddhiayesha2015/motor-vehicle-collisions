# Motor Vehicle Collisions Modeling

This project cleans the NYC Motor Vehicle Collisions dataset, performs time-aware splits, trains multiple models with MLflow tracking, and exposes predictions through FastAPI.

## Workflows
- `python -m src.collisions.data_cleaning` cleans data, normalizes fields, encodes categoricals, and saves train/validate/test splits in `data/processed/`.
- `scripts/run_automl.py` runs H2O AutoML on the training set and records the leaderboard.
- `scripts/train_random_forest.py`, `scripts/train_gradient_boosting.py`, and `scripts/train_elastic_net.py` train three models, evaluate them, log to MLflow, and register them in the Model Registry.
- `scripts/drift_analysis.py` evaluates data and performance drift using a newer time window.
- `app/main.py` starts a FastAPI app with three prediction endpoints that load models from the registry.
- `scripts/demo_requests.py` exercises the FastAPI endpoints with a GUI-friendly JSON payload for quick smoke tests.

## MLflow Tracking
A `docker-compose.yml` file spins up PostgreSQL and an MLflow tracking server:
```
docker compose up
```
Point the training scripts to the tracking server using environment variables:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_REGISTRY_URI=http://localhost:5000
```

## Serving
```
uvicorn app.main:app --reload
```

Use the JSON schema in `app/schemas.py` for request formatting.
