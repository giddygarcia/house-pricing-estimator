from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="House Pricing Prediction API")

model = joblib.load("lgbm_model.pkl")

class PredictRequest(BaseModel):
    features: dict = Field(example={
        "District": "Faro",
        "City": "Loulé",
        "Town": "São Clemente",
        "TotalArea": 226.0,
        "LivingArea": 184.0,
        "ConstructionYear": 1958,
        "TotalRooms": 3,
        "NumberOfBedrooms": 2,
        "NumberOfBathrooms": 2,
        "Parking": 0,
        "Garage": 0,
        "Elevator": 0,
        "ElectricCarsCharging": 0,
        "EnergyCertNum": 4
    })

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
        df = pd.DataFrame([body.features])
        price = np.expm1(model.predict(df)[0])
        return PredictResponse(predicted_price=f"€{float(price):,.2f}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))