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

# Labels 
label_map = {0: "No Risk", 1: "Possible Risk"}

@app.post("/send_all_metrics")
async def receive_all_metrics(request: Request):
    try:
        data = await request.json()
        print(f"Received 4-metrics: {data}")

        features = {
            "hr":  data.get("HR"),
            "hrv": data.get("HRV"),
            "rhr": data.get("RHR"),
            "hhr": data.get("HHR"),
        }
        df = pd.DataFrame([features])
        prediction = health_model.predict(df)[0]
        prediction_label = label_map.get(prediction)

        return {"status": "success", "initialPrediction": prediction_label}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/send_test_ecg")
async def send_test_ecg(request: Request):
    try:
        from ecg_model import predict_from_json
        json_path = "../data/mockhealthkit/high_risk/137334.json"
        model_path = "../models/ecg/bilstm_model_multiclass.pth"

        print("You are currently in the demo mode")

        results = predict_from_json(json_path, model_path)
        return determine_risk(results)

    except Exception as e:
        return {"status":"error","message":str(e)}

@app.post("/send_ecg")
async def send_ecg_data(request: Request):
    try:
        from ecg_model import predict_from_json
        data = await request.json()
        json_path = data.get("json_path")
        model_path = "../models/ecg/bilstm_model_multiclass.pth"

        results = predict_from_json(json_path, model_path)
        return determine_risk(results)

    except Exception as e:
        return {"status":"error","message":str(e)}

# Helper function to determine risk level
def determine_risk(results):
    if results.count("High") >= 2:
        return {"status":"success","finalPrediction": "High Risk, Contact EMS"}
    elif results.count("High") == 1:
        return {"status":"success","finalPrediction": "Symptoms of Cardiac Arrest, Monitor"}
    elif results.count("Low") == len(results):
        return {"status":"success","finalPrediction": "False Alarm"}
    else:
        return {"status":"error","message":"Unable to determine risk."}

