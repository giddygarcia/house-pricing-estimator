import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="House Pricing Prediction API")

MODEL_PATH = Path(__file__).resolve().parent / "lgbm_model.pkl"
model = joblib.load(MODEL_PATH)


class PropertyFeatures(BaseModel):
    District: str | None = "Faro"
    City: str | None = "Loulé"
    Town: str | None = "São Clemente"
    TotalArea: float | None = 226.0
    LivingArea: float | None = 184.0
    ConstructionYear: float | None = 1958
    TotalRooms: float | None = 3
    NumberOfBedrooms: float | None = 2
    NumberOfBathrooms: float | None = 2
    Parking: float | None = 0
    Garage: float | None = 0
    Elevator: float | None = 0
    ElectricCarsCharging: float | None = 0
    EnergyCertNum: float | None = 4


class PredictRequest(BaseModel):
    features: PropertyFeatures


class PredictResponse(BaseModel):
    predicted_price: str


@app.get("/")
def root():
    return {"service": "House Pricing Prediction API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        df = pd.DataFrame([body.features.model_dump()])
        price = np.expm1(model.predict(df)[0])
        return PredictResponse(predicted_price=f"€{float(price):,.2f}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
