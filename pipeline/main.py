from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import wfdb
from joblib import load
import os

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models and Globals ---
ensemble4_model = None
ensemble5_model = None

# Dummy scalers and models
hr_threshold = 160  # fallback HR threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model (must match what was trained)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Load ECG model
ecg_model = LSTMModel().to(device)
ecg_model.load_state_dict(torch.load('./models/ecg/lstm_model.pth', map_location=device))
ecg_model.eval()

# --- Data model ---
class ECGData(BaseModel):
    ecg_signal: list  # ECG segment data (1D list)

# --- Startup: load everything ---
@app.on_event("startup")
def startup_event():
    global ensemble4_model, ensemble5_model

    print("🔵 Loading Ensemble Models...")
    ensemble4_model = load('./models/ensemble/ensemble4_model.pkl')
    ensemble5_model = load('./models/ensemble/ensemble5_model.pkl')
    print("✅ Loaded ensemble models!")

    print("🧪 Running MITDB benchmark...")
    benchmark_mitdb()

# --- Helper Functions ---
def benchmark_mitdb():
    """Evaluates both ensemble4 and ensemble5 on MITDB records"""
    mitdb_records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
           list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
           list(range(228, 229)) + list(range(230, 235))]
    ]

    results_4 = []
    results_5 = []

    for record in mitdb_records:
        try:
            signal, fields = wfdb.rdsamp(f'../CardioVision/data/mitdb/{record}')
            ecg_signal = signal[:, 0]

            ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                out = ecg_model(ecg_tensor)
                ecg_pred = (torch.sigmoid(out.squeeze()) > 0.5).float().item()

            features = np.array([1, 1, 1, 1])  # Assume dummy HR, HRV, RHR, HHR features
            risk4 = ensemble4_model.predict(features.reshape(1, -1))[0]
            risk5 = ensemble5_model.predict(np.append(features, ecg_pred).reshape(1, -1))[0]

            results_4.append(risk4)
            results_5.append(risk5)

        except Exception as e:
            print(f"[⚠️] Skipped {record}: {e}")

    print("✅ Benchmark Results:")
    print(f"4-Ensemble Predictions: {results_4}")
    print(f"5-Ensemble Predictions: {results_5}")

# --- API Endpoints ---
@app.post("/heartbeat")
def heartbeat(data: ECGData):
    ecg_signal = np.array(data.ecg_signal)

    if ecg_signal.size == 0:
        raise HTTPException(status_code=400, detail="ECG signal empty.")

    # Step 1: Dummy feature generation for HR, HRV, RHR, HHR
    features4 = np.array([1, 1, 1, 1]).reshape(1, -1)  # Assume good metrics

    risk4 = ensemble4_model.predict(features4)[0]

    if risk4 == 0:
        return {"message": "✅ Heartbeat normal. No ECG needed.", "risk_level": "Low"}

    # Step 2: Since risk detected → Analyze full ECG
    ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        out = ecg_model(ecg_tensor)
        ecg_pred = (torch.sigmoid(out.squeeze()) > 0.5).float().item()

    # Step 3: Final ensemble prediction
    features5 = np.append(features4.flatten(), ecg_pred).reshape(1, -1)
    risk5 = ensemble5_model.predict(features5)[0]

    if risk5 == 0:
        risk_label = "Low Risk"
    elif risk5 == 1:
        risk_label = "High Risk"
    else:
        risk_label = "Medium Risk"

    return {"message": "✅ Heartbeat analyzed.", "final_risk": risk_label}

