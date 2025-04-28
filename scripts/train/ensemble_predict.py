import os
import json
import torch
import numpy as np
import wfdb
from joblib import load
from sklearn.preprocessing import StandardScaler

# =============================
# Load Models
# =============================

# HR Model (Threshold Based)
def load_hr_model(path='../CardioVision/models/heartrate/hr_model2.json'):
    with open(path, 'r') as f:
        return json.load(f)

# HRV Model (XGBoost)
def load_hrv_model(path='../CardioVision/models/heartratevariability/xgb_hrv_model.pkl'):
    return load(path)

# RHR Model (Logistic Regression)
def load_rhr_model(path='../CardioVision/models/restingheartrate/rhr_model.pkl'):
    return load(path)

# HHR Model (Random Forest)
def load_hhr_model(path='../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl'):
    return load(path)

# ECG Model (LSTM)
class ECG_LSTM(torch.nn.Module):
    def __init__(self):
        super(ECG_LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1, hidden_size=128, num_layers=3, batch_first=True, bidirectional=False
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return torch.sigmoid(x)

def load_ecg_model(path='../CardioVision/models/ecg/lstm_model.pth', device='cpu'):
    model = ECG_LSTM()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# =============================
# Feature Extraction
# =============================

def load_ecg_signal(record, base_path='../CardioVision/data/mitdb'):
    rec = wfdb.rdrecord(os.path.join(base_path, record))
    signal = rec.p_signal[:, 0]  # Only use MLII
    return signal, rec.fs

def extract_rr_intervals(signal, fs):
    r_peaks = np.where(np.diff(np.sign(signal - np.mean(signal))) > 0)[0]
    rr_intervals = np.diff(r_peaks) / fs * 1000  # milliseconds
    return rr_intervals

def compute_hr_features(rr_intervals):
    hr_series = 60000 / rr_intervals
    mean_hr = np.mean(hr_series)
    return mean_hr

def compute_hrv_features(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0
    mean_rr = np.mean(rr_intervals)
    return np.array([rmssd, sdnn, nn50, pnn50, mean_rr])

def extract_beats(signal, r_peaks, window_size=250):
    beats = []
    for peak in r_peaks:
        start = peak - window_size // 2
        end = peak + window_size // 2
        if start >= 0 and end < len(signal):
            beats.append(signal[start:end])
    return np.array(beats)

# =============================
# Ensemble Prediction
# =============================

def ensemble_predict(record, use_ecg=False):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all models
    hr_model = load_hr_model()
    hrv_model = load_hrv_model()
    rhr_model = load_rhr_model()
    hhr_model = load_hhr_model()
    ecg_model = load_ecg_model(device=device) if use_ecg else None

    # Load signal
    signal, fs = load_ecg_signal(record)
    rr_intervals = extract_rr_intervals(signal, fs)

    if len(rr_intervals) < 2:
        print(f"[Skip] Record {record}: Not enough RR intervals.")
        return

    mean_hr = compute_hr_features(rr_intervals)
    hrv_feats = compute_hrv_features(rr_intervals).reshape(1, -1)

    # Load scalers
    hrv_scaler = load('../CardioVision/models/heartratevariability/scaler.pkl')
    hhr_scaler = load('../CardioVision/models/highheartrateevents/scaler.pkl')

    # HR Risk
    hr_risk = int(mean_hr > hr_model['threshold'])

    # HRV Risk
    # Pad HRV features to 9 dimensions if necessary
    if hrv_feats.shape[1] < 9:
        padding = np.zeros((hrv_feats.shape[0], 9 - hrv_feats.shape[1]))
        hrv_feats = np.hstack((hrv_feats, padding))
    hrv_feats_scaled = hrv_scaler.transform(hrv_feats)
    hrv_risk = int(hrv_model.predict(hrv_feats_scaled)[0])

    # RHR Risk (Use mean HR for RHR model)
    rhr_risk = int(rhr_model.predict([[mean_hr]])[0])

    # HHR Risk (Spike features placeholder)
    spike_features = np.array([[0, 0, 0, 0, 0, 0, 0]])  # Simplified placeholder
    # Pad HHR features to 8 dimensions if necessary
    if spike_features.shape[1] < 8:
        padding = np.zeros((spike_features.shape[0], 8 - spike_features.shape[1]))
        spike_features = np.hstack((spike_features, padding))

    spike_features_scaled = hhr_scaler.transform(spike_features)
    hhr_risk = int(hhr_model.predict(spike_features_scaled)[0])

    # ECG Risk
    ecg_risk = 0
    if use_ecg and ecg_model is not None:
        r_peaks = np.where(np.diff(np.sign(signal - np.mean(signal))) > 0)[0]
        beats = extract_beats(signal, r_peaks)
        if beats.shape[0] > 0:
            beats_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(-1).to(device)
            with torch.no_grad():
                outputs = ecg_model(beats_tensor)
                preds = (outputs.squeeze() > 0.5).float()
                ecg_risk = int(preds.mean() > 0.5)

    # ================================
    # Combine Ensemble Decision
    # ================================
    metrics = [hr_risk, hrv_risk, rhr_risk, hhr_risk]
    if use_ecg:
        metrics.append(ecg_risk)

    total_risk_score = sum(metrics)
    
    if total_risk_score >= (4 if use_ecg else 3):
        risk_level = 'High Risk'
    elif total_risk_score >= (2 if use_ecg else 2):
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'

    print(f"\n✅ [Ensemble] Record {record}")
    print(f"Metrics - HR: {hr_risk}, HRV: {hrv_risk}, RHR: {rhr_risk}, HHR: {hhr_risk}, ECG: {ecg_risk if use_ecg else 'N/A'}")
    print(f"➡️ Final Predicted Risk Level: {risk_level}")

# =============================
# Main Testing
# =============================

if __name__ == '__main__':
    # Example run
    test_records = ['100', '101', '102']  # MIT-BIH records
    for record in test_records:
        ensemble_predict(record, use_ecg=True)  # Toggle use_ecg=True/False
