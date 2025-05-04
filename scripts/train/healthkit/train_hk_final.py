import os
import numpy as np
import wfdb
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load, dump

# Load your submodels
hrv_model = load('../CardioVision/models/heartratevariability/xgb_hrv_model.pkl')
hrv_scaler = load('../CardioVision/models/heartratevariability/scaler.pkl')

rhr_model = load('../CardioVision/models/restingheartrate/rhr_model.pkl')
rhr_scaler = load('../CardioVision/models/restingheartrate/scaler.pkl')

hhr_model = load('../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl')
hhr_scaler = load('../CardioVision/models/highheartrateevents/scaler.pkl')

# HR model (threshold-based fallback)
import json
with open('../CardioVision/models/heartrate/hr_model2.json', 'r') as f:
    hr_threshold_model = json.load(f)

# ECG LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ecg_model = LSTMModel().to(device)
ecg_model.load_state_dict(torch.load('../CardioVision/models/ecg/lstm_model.pth', map_location=device))
ecg_model.eval()

# -----------------------------------------------------
# Helper functions

def extract_features(record):
    """Extracts all features for ensemble: HR, HRV, RHR, HHR, ECG"""
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    signal = rec.p_signal[:, 0]
    r_peaks = ann.sample

    # Compute mean HR
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # ms
    hr_series = 60000 / rr_intervals
    mean_hr = np.mean(hr_series)

    # HRV features (dummy shape to match scaler)
    hrv_feats = np.random.randn(1, 9)
    hrv_feats_scaled = hrv_scaler.transform(hrv_feats)
    hrv_pred = hrv_model.predict(hrv_feats_scaled)[0]

    # RHR feature (dummy, just use mean_hr)
    rhr_feats = np.array([[mean_hr]])
    rhr_feats_scaled = rhr_scaler.transform(rhr_feats)
    rhr_pred = rhr_model.predict(rhr_feats_scaled)[0]

    # HHR features (dummy shape to match scaler)
    hhr_feats = np.random.randn(1, 8)
    hhr_feats_scaled = hhr_scaler.transform(hhr_feats)
    hhr_pred = hhr_model.predict(hhr_feats_scaled)[0]

    # HR threshold check
    hr_pred = 1 if mean_hr > hr_threshold_model['threshold'] else 0

    # ECG beat prediction
    ecg_features = []
    for beat in r_peaks:
        start = beat - 125
        end = beat + 125
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            out = ecg_model(segment)
            pred = (torch.sigmoid(out.squeeze()) > 0.5).float().item()
            ecg_features.append(pred)

    ecg_pred = int(np.round(np.mean(ecg_features))) if ecg_features else 0

    return [hr_pred, hrv_pred, rhr_pred, hhr_pred, ecg_pred]

# -----------------------------------------------------
# Load data
def load_data(records):
    X = []
    y = []
    for record in records:
        try:
            features = extract_features(record)
            X.append(features)

            # Label assignment:
            # If ECG detects arrhythmia (1), assume Medium/High Risk
            label = 1 if features[4] == 1 else 0  # ECG-based label
            y.append(label)

            print(f"[✅] {record} - Features: {features}, Label: {label}")

        except Exception as e:
            print(f"[⚠️] Skip {record}: {e}")

    return np.array(X), np.array(y)

# -----------------------------------------------------
# Training logic
def train_ensemble5():
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
           list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
           list(range(228, 229)) + list(range(230, 235))]
    ]

    print("🔎 Loading data from MITDB...")
    X, y = load_data(records)

    print(f"✅ Dataset shape: {X.shape}")

    # Train a logistic regression
    ensemble_model = LogisticRegression()
    ensemble_model.fit(X, y)

    os.makedirs('../CardioVision/models/ensemble', exist_ok=True)
    dump(ensemble_model, '../CardioVision/models/ensemble/ensemble5_model.pkl')
    print("✅ Saved 5-Layer Ensemble model to models/ensemble/ensemble5_model.pkl")

if __name__ == "__main__":
    train_ensemble5()
