from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from models.train_lstm import LSTMModel

app = FastAPI()

# Updated input model to be extensible
class ECGInput(BaseModel):
    voltage: list[float]
    sampling_frequency: float | None = None  # Optional, future use

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1)
model.load_state_dict(torch.load("models/lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict_ecg")
def predict_ecg(data: ECGInput):
    try:
        # Take the first 250 values (matching training window)
        ecg_signal = np.array(data.voltage[:250], dtype=np.float32)

        # Normalize the signal: center and scale
        ecg_signal -= np.mean(ecg_signal)
        max_abs = np.max(np.abs(ecg_signal)) or 1.0  # avoid divide-by-zero
        ecg_signal /= max_abs

        # Prepare input for model
        tensor_input = torch.tensor(ecg_signal).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            output = model(tensor_input)
            prediction = torch.sigmoid(output.squeeze()).item()

        risk = "arrhythmia" if prediction > 0.5 else "normal"

        return {
            "risk_level": risk,
            "probability": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
