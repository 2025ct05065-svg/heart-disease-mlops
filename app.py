import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from patient features.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    sex: int
    cp: int
    fbs: int
    restecg: int
    exang: int
    slope: int
    ca: float
    thal: float


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        data = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        return {
            "prediction": prediction,
            "label": (
                "Heart disease detected"
                if prediction == 1
                else "No heart disease detected"
            ),
            "probability": round(probability, 4),
            "note": "This is a model prediction, not a medical diagnosis.",
        }
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid prediction input: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc!s}",
        ) from exc
