from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from scripts.train.train_ecg import LSTMModel

app = FastAPI()

class ECGInput(BaseModel):
    voltage: list[float]  # Ensure frontend sends key "voltage"
    sampling_frequency: float | None = None

# Load trained model
model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1)
model.load_state_dict(torch.load("models/lstm_model.pth", map_location=torch.device('cpu')))
model.eval()
print("[MODEL] LSTM model loaded and ready.")

@app.post("/predict_ecg")
def predict_ecg(data: ECGInput):
    try:
        print(f"[RECEIVED] ECG voltage list of size {len(data.voltage)}")

        ecg_signal = np.array(data.voltage[:250], dtype=np.float32)
        print(f"[PREPROCESS] First 5 raw values: {ecg_signal[:5]}")

        # Normalize signal
        ecg_signal -= np.mean(ecg_signal)
        max_abs = np.max(np.abs(ecg_signal)) or 1.0
        ecg_signal /= max_abs
        print(f"[NORMALIZED] First 5 values: {ecg_signal[:5]}")

        tensor_input = torch.tensor(ecg_signal).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            output = model(tensor_input)
            prediction = torch.sigmoid(output.squeeze()).item()

        risk = "arrhythmia" if prediction > 0.5 else "normal"
        print(f"[RESULT] Risk: {risk}, Probability: {prediction:.4f}")

        return {
            "risk_level": risk,
            "probability": prediction
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
