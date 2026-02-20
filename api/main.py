from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import AgePredictor
import yaml
import uvicorn

app = FastAPI(title="NHANES Age Intelligence API")

# Load config to get feature names
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
FEATURES = config['data']['features']

class Biomarkers(BaseModel):
    BMXWT: float
    BMXHT: float
    BMXBMI: float
    BMXWAIST: float
    LBXSTP: float
    LBXSTR: float
    LBXSCH: float
    RIAGENDR: float

@app.get("/")
def read_root():
    return {"message": "Welcome to NHANES Age API"}

@app.post("/predict")
def predict_age(data: Biomarkers):
    try:
        predictor = AgePredictor()
        input_df = pd.DataFrame([data.dict()])
        prediction = predictor.predict(input_df)
        return {"predicted_age": round(float(prediction[0]), 2)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
