from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_model.joblib"
FEATURES_PATH = BASE_DIR / "house_features.joblib"

app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Model files not found in {BASE_DIR}. "
        "Make sure house_model.joblib and house_features.joblib are present in the same folder as main.py."
    ) from exc


class HouseFeature(BaseModel):
    MedInc: float = Field(gt=0, description="median income in the block")
    HouseAge: float = Field(gt=0, description="median house age in the block")
    AveRooms: float = Field(gt=0, description="average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="average number of bedrooms per household")
    Population: float = Field(gt=0, description="block population")
    AveOccup: float = Field(gt=0, description="average number of household members")
    Latitude: float = Field(ge=32, le=42, description="block latitude")
    Longitude: float = Field(ge=-125, le=-114, description="block longitude")


@app.get("/")
def home():
    return {
        "message": "Welcome to the California House Price Prediction API. Use the /predict endpoint to get predictions based on house features.",
        "status": "success",
        "endpoints": ["send POST request to /predict"],
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000",
    }


@app.post("/predict")
def predict(house: HouseFeature):
    try:
        input_data = pd.DataFrame(
            [{
                "MedInc": house.MedInc,
                "HouseAge": house.HouseAge,
                "AveRooms": house.AveRooms,
                "AveBedrms": house.AveBedrms,
                "Population": house.Population,
                "AveOccup": house.AveOccup,
                "Latitude": house.Latitude,
                "Longitude": house.Longitude,
            }]
        )

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.2f}",
            "predicted_price_short": f"${predicted * 100000:,.2f} (approx. {predicted:.2f} hundred thousand)",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}",
        ) from e