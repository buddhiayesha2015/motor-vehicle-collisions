# Motor Vehicle Collisions Modeling

This project cleans the NYC Motor Vehicle Collisions dataset, performs time-aware splits, trains multiple models with MLflow tracking, and exposes predictions through FastAPI.

## Quickstart (Windows 11, PowerShell)
1. **Clone and enter the repo**
   ```powershell
   git clone <repo-url>
   cd motor-vehicle-collisions
   ```

2. **Install `uv` and sync the environment**
   ```powershell
   pip install uv
   uv sync
   ```
   *This creates/updates the `.venv` directory and installs all project dependencies declared in `pyproject.toml`.*

3. **Activate the virtual environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Download the data (place provided CSV/XLSX under `data/`)**
   - `data/Motor_Vehicle_Collisions-Crashes_20251211.csv`
   - `data/MVCollisionsDataDictionary_20190813_ERD.xlsx`

5. **Clean and split the data**
   ```powershell
   python -m src.collisions.data_cleaning
   ```
   Outputs go to `data/processed/` (train/validate/test) and `artifacts/feature_metadata.json`.

6. **Start MLflow + PostgreSQL via Docker Compose (local option)**
   ```powershell
   docker compose up
   ```
   MLflow UI: <http://localhost:5000> (by default). PostgreSQL runs on port 5432.

7. **Point scripts to MLflow tracking** (new PowerShell session or after activating the venv)
   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
   $env:MLFLOW_REGISTRY_URI = "http://localhost:5000"
   ```

8. **Run H2O AutoML** (on the cleaned training split)
   ```powershell
   python scripts\run_automl.py
   ```

9. **Train the three manual models**
   ```powershell
   python scripts\train_random_forest.py
   python scripts\train_gradient_boosting.py
   python scripts\train_elastic_net.py
   ```
   Each script logs metrics/artifacts to the same MLflow experiment and registers a model version.

10. **Compare and visualize**
    ```powershell
    python scripts\compare_models.py
    python scripts\drift_analysis.py
    ```

11. **Serve predictions with FastAPI**
    ```powershell
    uvicorn app.main:app --reload
    ```
    Endpoints: `/predict_model1`, `/predict_model2`, `/predict_model3` (payload schema in `app/schemas.py`).

12. **Smoke-test the API with a browser**
    - Visit <http://localhost:8000/demo> for a web-based JSON client that calls the prediction endpoints without writing code.
    - The server logs will show how many predictions were generated plus a short preview for each request.
    - The legacy CLI smoke-test remains available via `python scripts\demo_requests.py` if you prefer a script.

## Workflow reference
- `python -m src.collisions.data_cleaning` cleans data, normalizes fields, encodes categoricals, and saves time-aware train/validate/test splits in `data/processed/`.
- `scripts/run_automl.py` runs H2O AutoML on the training set and records the leaderboard.
- `scripts/train_random_forest.py`, `scripts/train_gradient_boosting.py`, and `scripts/train_elastic_net.py` train three models, evaluate them, log to MLflow, and register them in the Model Registry.
- `scripts/drift_analysis.py` evaluates data and performance drift using a newer time window.
- `app/main.py` starts a FastAPI app with three prediction endpoints that load models from the registry.
- `scripts/demo_requests.py` exercises the FastAPI endpoints with a GUI-friendly JSON payload for quick smoke tests.

## MLflow Tracking
A `docker-compose.yml` file spins up PostgreSQL and an MLflow tracking server:
```powershell
docker compose up
```
Point the training scripts to the tracking server using environment variables:
```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_REGISTRY_URI = "http://localhost:5000"
```

### Remote MLflow (EC2 + S3 + Neon Postgres)
To log runs to the hosted stack (EC2 MLflow server, S3 artifacts, Neon Postgres backend), create a `.env` from the template and fill in your secrets locally:

```powershell
Copy-Item .env.example .env
notepad .env   # fill in AWS + Neon credentials, keep the file out of version control
```

Key entries in `.env`:
- `MLFLOW_TRACKING_URI` / `MLFLOW_REGISTRY_URI`: `http://34.206.23.119:5000`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`: IAM credentials for `mlflow-artifacts-buddhi`
- `MLFLOW_S3_BUCKET`: `mlflow-artifacts-buddhi`
- `POSTGRES_CONNECTION_URI`: Neon SQLAlchemy URI (e.g., `postgresql+psycopg2://mlflow_buddhi:<YOUR_NEON_PASSWORD>@ep-lively-band-a4ojyx5g-pooler.us-east-1.aws.neon.tech/mlflow?sslmode=require&channel_binding=require`)

Then point your PowerShell session at the remote MLflow instance before running training scripts:

```powershell
$env:MLFLOW_TRACKING_URI = "http://34.206.23.119:5000"
$env:MLFLOW_REGISTRY_URI = "http://34.206.23.119:5000"

# Load AWS creds from .env into the session (simple example, adjust if you prefer a secrets manager)
Get-Content .env | Where-Object { $_ -match "^AWS_" } | ForEach-Object {
  $name, $value = $_.Split('='); Set-Item -Path "Env:$name" -Value $value
}
```

When the server on EC2 is started with the Neon connection string and the `mlflow-artifacts-buddhi` S3 bucket, all artifacts and metadata from local training runs will be persisted remotely.

## Serving
```powershell
uvicorn app.main:app --reload
```

Use the JSON schema in `app/schemas.py` for request formatting.
