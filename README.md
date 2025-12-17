# Motor Vehicle Collisions Modeling

This project cleans the NYC Motor Vehicle Collisions dataset, performs time-aware splits, trains multiple models with MLflow tracking, and exposes predictions through FastAPI.

## Quickstart (Windows 11, PowerShell)
This workflow assumes MLflow is hosted remotely (EC2 + Neon + S3) and your local scripts log to that server.

1. **Clone and enter the repo**
   ```powershell
   git clone <repo-url>
   cd motor-vehicle-collisions
   ```

2. **Create your `.env`**
   Copy `.env.example` to `.env` and keep it out of version control. Update the client-side values to point at the remote MLflow server (tracking/registry URIs default to `http://34.206.23.119:5000`).

3. **Install `uv` and sync the environment**
   ```powershell
   pip install uv
   uv sync
   ```
   *This creates/updates the `.venv` directory and installs all project dependencies declared in `pyproject.toml`.*

4. **Activate the virtual environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. **Download the data (place provided CSV/XLSX under `data/`)**
   - `data/Motor_Vehicle_Collisions-Crashes_20251211.csv`
   - `data/MVCollisionsDataDictionary_20190813_ERD.xlsx`

6. **Clean and split the data**
   ```powershell
   python -m src.collisions.data_cleaning
   ```
   Outputs go to `data/processed/` (train/validate/test) and `artifacts/feature_metadata.json`.

7. **Run H2O AutoML** (on the cleaned training split; logs to remote MLflow)
   ```powershell
   python scripts\run_automl.py
   ```

8. **Train the three manual models**
   ```powershell
   python scripts\train_random_forest.py
   python scripts\train_gradient_boosting.py
   python scripts\train_elastic_net.py
   ```
   Each script logs metrics/artifacts to the same remote MLflow experiment and registers a model version in the remote registry.

9. **Compare and visualize**
   ```powershell
   python scripts\compare_models.py
   python scripts\drift_analysis.py
   ```

10. **Serve predictions with FastAPI**
    ```powershell
    uvicorn app.main:app --reload
    ```
    Endpoints: `/predict_model1`, `/predict_model2`, `/predict_model3` (payload schema in `app/schemas.py`).

11. **Smoke-test the API with a browser**
    - Visit <http://localhost:8000/demo> for a web-based JSON client that calls the prediction endpoints without writing code.
    - The server logs will show how many predictions were generated plus a short preview for each request.
    - The legacy CLI smoke-test remains available via `python scripts\demo_requests.py` if you prefer a script.

## Workflow reference
- `python -m src.collisions.data_cleaning` cleans data, normalizes fields, encodes categoricals, and saves time-aware train/validate/test splits in `data/processed/`.
- `scripts/run_automl.py` runs H2O AutoML on the training set and logs the run, leaderboard artifact, and leader model to the remote MLflow experiment/registry.
- `scripts/train_random_forest.py`, `scripts/train_gradient_boosting.py`, and `scripts/train_elastic_net.py` train three models, evaluate them, log to MLflow, and register them in the Model Registry.
- `scripts/drift_analysis.py` evaluates data and performance drift using a newer time window.
- `app/main.py` starts a FastAPI app with three prediction endpoints that load models from the registry.
- `scripts/demo_requests.py` exercises the FastAPI endpoints with a GUI-friendly JSON payload for quick smoke tests.

## Remote MLflow Tracking (AWS EC2 + Neon + S3)
**Server side (run on the EC2 instance at `34.206.23.119`):**
- Export secrets with environment variables (or an `.env` kept on the VM):
  ```bash
  export MLFLOW_BACKEND_STORE_URI="postgresql://<neon-user>:<password>@<neon-host>:<port>/<database>?sslmode=require"
  export MLFLOW_ARTIFACT_ROOT="s3://mlflow-artifacts-buddhi"
  export AWS_ACCESS_KEY_ID="<aws-access-key-id>"
  export AWS_SECRET_ACCESS_KEY="<aws-secret-access-key>"
  export AWS_DEFAULT_REGION="us-east-1"
  ```
- Start the tracking server (installed on EC2):
  ```bash
  mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT"
  ```

**Client side (Windows 11 for training and serving):**
- Copy `.env.example` to `.env` and set `MLFLOW_TRACKING_URI`/`MLFLOW_REGISTRY_URI` to `http://34.206.23.119:5000` (or your load balancer/hostname). The Pydantic settings loader automatically reads `.env`, so you do not need to export variables manually.
- Keep database passwords, AWS keys, and the Neon connection string out of source control—store them only in the remote `.env` or environment variables on EC2.

## Serving
```powershell
uvicorn app.main:app --reload
```

Use the JSON schema in `app/schemas.py` for request formatting.
