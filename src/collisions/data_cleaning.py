from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import PATHS

TARGET_COLUMN = "NUMBER OF PERSONS INJURED"


NUMERIC_COLUMNS = [
    "LATITUDE",
    "LONGITUDE",
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
]

CATEGORICAL_COLUMNS = [
    "BOROUGH",
    "ZIP CODE",
    "ON STREET NAME",
    "CROSS STREET NAME",
    "OFF STREET NAME",
    "CONTRIBUTING FACTOR VEHICLE 1",
    "VEHICLE TYPE CODE 1",
]

DATETIME_COLS = ["CRASH DATE", "CRASH TIME"]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.title()


def clamp_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[~df["LATITUDE"].between(40, 41), "LATITUDE"] = np.nan
    df.loc[~df["LONGITUDE"].between(-75, -71), "LONGITUDE"] = np.nan
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        df[col] = normalize_text(df[col]).replace({"": "Unknown"})
    return df


def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["CRASH TIME"] = df["CRASH TIME"].astype(str).str.zfill(5)
    df["CRASH_DATETIME"] = pd.to_datetime(
        df["CRASH DATE"] + " " + df["CRASH TIME"], errors="coerce", infer_datetime_format=True
    )
    df = df.dropna(subset=["CRASH_DATETIME"])
    df["CRASH_DATE_ISO"] = df["CRASH_DATETIME"].dt.strftime("%Y-%m-%d")
    df["CRASH_HOUR"] = df["CRASH_DATETIME"].dt.hour
    df["CRASH_MONTH"] = df["CRASH_DATETIME"].dt.month
    df["CRASH_YEAR"] = df["CRASH_DATETIME"].dt.year
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLUMNS:
        df[f"{col}_ENCODED"] = df[col].astype("category").cat.codes
    return df


def clean_dataset(raw_path: Path = PATHS.raw_data, output_path: Path = PATHS.cleaned_data) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = convert_datetime(df)
    df = df.sort_values("CRASH_DATETIME").reset_index(drop=True)
    df = clamp_coordinates(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = df.drop_duplicates(subset=["COLLISION_ID"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    metadata = {
        "target": TARGET_COLUMN,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_columns": [f"{c}_ENCODED" for c in CATEGORICAL_COLUMNS],
    }
    PATHS.feature_metadata.write_text(json.dumps(metadata, indent=2))
    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * 0.35)
    validate_end = int(n_rows * 0.70)
    train_df = df.iloc[:train_end].copy()
    validate_df = df.iloc[train_end:validate_end].copy()
    test_df = df.iloc[validate_end:].copy()
    return train_df, validate_df, test_df


def split_and_save(df: pd.DataFrame) -> None:
    train_df, validate_df, test_df = chronological_split(df)
    PATHS.train.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PATHS.train, index=False)
    validate_df.to_csv(PATHS.validate, index=False)
    test_df.to_csv(PATHS.test, index=False)


def main() -> None:
    df = clean_dataset()
    split_and_save(df)


if __name__ == "__main__":
    main()
