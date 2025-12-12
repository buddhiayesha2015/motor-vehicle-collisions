from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlflowSettings(BaseSettings):
    """Configuration for MLflow tracking and registry."""

    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="collision_regression")
    registry_uri: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env")


class DatabaseSettings(BaseSettings):
    """Settings for the Postgres backend store for MLflow."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="mlflow")
    password: str = Field(default="mlflow")
    database: str = Field(default="mlflow")

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env")

    @property
    def uri(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class Paths(BaseModel):
    """Project paths used across scripts."""

    raw_data: Path = Path("data/Motor_Vehicle_Collisions-Crashes_20251211.csv")
    processed_dir: Path = Path("data/processed")
    cleaned_data: Path = processed_dir / "cleaned_collisions.csv"
    train: Path = processed_dir / "train.csv"
    validate: Path = processed_dir / "validate.csv"
    test: Path = processed_dir / "test.csv"
    automl_leaderboard: Path = Path("artifacts/automl_leaderboard.csv")
    feature_metadata: Path = Path("artifacts/feature_metadata.json")


MLFLOW_SETTINGS = MlflowSettings()
DB_SETTINGS = DatabaseSettings()
PATHS = Paths()
