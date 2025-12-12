from __future__ import annotations

import json
from datetime import date, time

import requests

sample_payload = [
    {
        "CRASH DATE": date(2024, 1, 15).isoformat(),
        "CRASH TIME": time(8, 30).strftime("%H:%M"),
        "BOROUGH": "Manhattan",
        "ZIP CODE": "10001",
        "LATITUDE": 40.748,
        "LONGITUDE": -73.9857,
        "ON STREET NAME": "West 34 Street",
        "CROSS STREET NAME": "7 Avenue",
        "OFF STREET NAME": "",
        "CONTRIBUTING FACTOR VEHICLE 1": "Unspecified",
        "VEHICLE TYPE CODE 1": "Sedan",
        "NUMBER OF PERSONS KILLED": 0,
        "NUMBER OF PEDESTRIANS INJURED": 0,
        "NUMBER OF PEDESTRIANS KILLED": 0,
        "NUMBER OF CYCLIST INJURED": 0,
        "NUMBER OF CYCLIST KILLED": 0,
        "NUMBER OF MOTORIST INJURED": 1,
        "NUMBER OF MOTORIST KILLED": 0,
    }
]

for endpoint in ["predict_model1", "predict_model2", "predict_model3"]:
    resp = requests.post(f"http://localhost:8000/{endpoint}", json=sample_payload, timeout=30)
    print(endpoint, resp.status_code, resp.json())
