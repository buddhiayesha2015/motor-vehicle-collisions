from __future__ import annotations

from datetime import date, time
from typing import Optional

from pydantic import BaseModel, Field


class CollisionRecord(BaseModel):
    crash_date: date = Field(..., alias="CRASH DATE")
    crash_time: time = Field(..., alias="CRASH TIME")
    borough: Optional[str] = Field(None, alias="BOROUGH")
    zip_code: Optional[str] = Field(None, alias="ZIP CODE")
    latitude: Optional[float] = Field(None, alias="LATITUDE")
    longitude: Optional[float] = Field(None, alias="LONGITUDE")
    on_street_name: Optional[str] = Field(None, alias="ON STREET NAME")
    cross_street_name: Optional[str] = Field(None, alias="CROSS STREET NAME")
    off_street_name: Optional[str] = Field(None, alias="OFF STREET NAME")
    contributing_factor_vehicle_1: Optional[str] = Field(None, alias="CONTRIBUTING FACTOR VEHICLE 1")
    vehicle_type_code_1: Optional[str] = Field(None, alias="VEHICLE TYPE CODE 1")
    number_of_persons_killed: int = Field(0, alias="NUMBER OF PERSONS KILLED")
    number_of_pedestrians_injured: int = Field(0, alias="NUMBER OF PEDESTRIANS INJURED")
    number_of_pedestrians_killed: int = Field(0, alias="NUMBER OF PEDESTRIANS KILLED")
    number_of_cyclist_injured: int = Field(0, alias="NUMBER OF CYCLIST INJURED")
    number_of_cyclist_killed: int = Field(0, alias="NUMBER OF CYCLIST KILLED")
    number_of_motorist_injured: int = Field(0, alias="NUMBER OF MOTORIST INJURED")
    number_of_motorist_killed: int = Field(0, alias="NUMBER OF MOTORIST KILLED")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str | None
    timestamp: str
    predictions: list[float]
