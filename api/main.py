from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from models.train_lstm import LSTMModel

app = FastAPI()

class ECGInput(BaseModel):
    ecg: list  # List of voltage values

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1)
model.load_state_dict(torch.load("models/lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict_ecg")
def predict_ecg(data: ECGInput):
    try:
        ecg_signal = np.array(data.ecg[:250])  # Take first 250 if longer
        tensor_input = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            output = model(tensor_input)
            prediction = torch.sigmoid(output.squeeze()).item()
        
        return {"risk": "arrhythmia" if prediction > 0.5 else "normal", "probability": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
