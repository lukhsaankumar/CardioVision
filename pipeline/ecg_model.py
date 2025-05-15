# ecg_model.py
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

def preprocess_single_json(json_path, window_size=250, stride=125, target_fs=250):
    with open(json_path, 'r') as f:
        data = json.load(f)

    voltages = np.array(data['voltages'], dtype=np.float32)
    if len(voltages) < window_size:
        return None

    segments = []
    scaler = StandardScaler()
    for start in range(0, len(voltages) - window_size + 1, stride):
        segment = voltages[start:start + window_size]
        segment = resample(segment, window_size)
        segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
        if np.any(np.isnan(segment)) or np.any(np.isinf(segment)) or np.std(segment) == 0:
            continue
        segments.append(segment)

    if not segments:
        return None

    segments = np.array(segments, dtype=np.float32)
    return torch.tensor(segments).unsqueeze(-1)  # shape: [num_segments, 250, 1]

def load_model(model_path, device):
    model = BiLSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_from_json(json_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    inputs = preprocess_single_json(json_path)

    if inputs is None:
        return "❌ No valid segments to process."

    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    label_map = {0: "Low", 1: "Med", 2: "High"}
    pred_labels = [label_map[p] for p in preds]
    return pred_labels
