from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

# load trained model at startup
health_model = joblib.load("../models/healthkit/healthkit_rf_model.pkl")

app = FastAPI()

# set up cors so the swift app can communicate to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# map numeric labels to messages
label_map = {0: "No Risk", 1: "Possible Risk"}

@app.post("/send_all_metrics")
async def receive_all_metrics(request: Request):
    try:
        # parse incoming json payload
        data = await request.json()
        print(f"Received 4 metrics {data}")

        # build dataframe for prediction
        features = {
            "hr": data.get("HR"),
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
        # import ecg predictor from model file
        from ecg_model import predict_from_json

        # use demo file and model for testing
        json_path = "../data/mockhealthkit/high_risk/137334.json"
        model_path = "../models/ecg/bilstm_model_multiclass.pth"

        print("You are in demo mode, This result is based on Mock ECG Data")
        results = predict_from_json(json_path, model_path)
        return determine_risk(results)

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/send_ecg")
async def send_ecg_data(request: Request):
    try:
        # import ecg predictor from model file
        from ecg_model import predict_from_json

        # read path to json from client payload
        data = await request.json()
        json_path = data.get("json_path")
        model_path = "../models/ecg/bilstm_model_multiclass.pth"

        results = predict_from_json(json_path, model_path)
        return determine_risk(results)

    except Exception as e:
        return {"status": "error", "message": str(e)}

# helper function to determine risk from ecg results
def determine_risk(results):
    if results.count("High") >= 2:
        return {"status": "success", "finalPrediction": "High Risk Contact EMS"}
    if results.count("High") == 1:
        return {"status": "success", "finalPrediction": "Symptoms of Cardiac Arrest Monitor"}
    if results.count("Low") == len(results):
        return {"status": "success", "finalPrediction": "False Alarm"}
    return {"status": "error", "message": "unable to determine risk"}
