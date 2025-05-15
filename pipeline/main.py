from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd


# Load the trained models at startup
health_model = joblib.load("../models/healthkit/healthkit_rf_model.pkl")

app = FastAPI()

# CORS setup so the Swift app can talk to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Labels (reverse mapping)
label_map = {0: "No Risk", 1: "Possible Risk"}
@app.post("/send_all_metrics")
async def receive_all_metrics(request: Request):
    try:
        data = await request.json()
        print(f"Received 4-metrics: {data}")

        # Convert feature names to lowercase to match the health_model
        features = {
            "hr":  data.get("HR"),
            "hrv": data.get("HRV"),
            "rhr": data.get("RHR"),
            "hhr": data.get("HHR"),
        }
        df = pd.DataFrame([features])
        prediction = health_model.predict(df)[0]
        prediction_label = label_map.get(prediction, "Unknown")

        print(f"HealthKit Model Prediction: {prediction_label}")
        return {"status": "success", "initialPrediction": prediction_label}

    except Exception as e:
        print(f"Error in /send_all_metrics: {e}")
        return {"status": "error", "message": str(e)}


# Send raw ECG recording JSON here
@app.post("/send_ecg")
def send_ecg(request: Request):
    try:
        from ecg_model import predict_from_json
        json_path = "../data/mockhealthkit/high_risk/137334.json"
        model_path = "../models/ecg/bilstm_model_multiclass.pth"

        results = predict_from_json(json_path, model_path)
        if results.count("High") == 1:
            return {"status":"success","finalPrediction": "Symptoms of Cardiac Arrest, Monitor"}
        elif results.count("Low") == len(results):
            return {"status":"success","finalPrediction": "False Alarm"}
        elif results.count("High") >= 2:
            return {"status":"success","finalPrediction": "High Risk, Contact EMS"}

    except Exception as e:
        print(f"Error in /send_ecg: {e}")
        return {"status":"error","message":str(e)}
    
    


