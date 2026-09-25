from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import os

app = FastAPI(title="MediVision-AI Backend", description="API for predicting chronic disease risks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
models = {}
features = []

@app.on_event("startup")
def load_models():
    global models, features
    try:
        models = joblib.load(MODEL_DIR / "trained_models.joblib")
        features = joblib.load(MODEL_DIR / "selected_features.joblib")
        print("Models successfully loaded!")
    except Exception as e:
        print(f"Error loading models: {e}. Please run the Jupyter notebook to train and save the models.")

class PatientData(BaseModel):
    age: float
    gender: int
    ethnicity: int
    bmi: float
    waist_cm: float
    weight_kg: float
    height_cm: float
    hba1c: float
    fasting_glucose: float
    total_cholesterol: float
    hdl: float
    triglycerides: float
    ldl: float
    ever_smoked: int
    sedentary_minutes: float
    education: int
    income_poverty_ratio: float

@app.post("/predict")
def predict(data: PatientData):
    if not models or not features:
        return {"error": "Models are not loaded on the server. Please check the backend logs."}
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Ensure column order matches exactly what the model expects
    df = df[features]
    
    results = {}
    for disease, model in models.items():
        disease_name = disease.replace('target_', '').replace('_', ' ').title()
        
        # Get probability of the positive class (index 1)
        prob = float(model.predict_proba(df)[0][1])
        
        results[disease_name] = {
            "risk_score": prob,
            "risk_percentage": f"{prob * 100:.1f}%",
            "prediction": "AT RISK" if prob > 0.5 else "Low Risk",
            "is_at_risk": bool(prob > 0.5)
        }
        
    return {
        "patient_profile": {"age": data.age, "bmi": data.bmi}, 
        "predictions": results
    }

@app.get("/")
def root():
    return {"message": "MediVision-AI API is running. Use /docs to view the API documentation."}

# Trigger reload

# Reload after age update

# Revert age update
