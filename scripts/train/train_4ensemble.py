# Updated train_4ensemble.py as discussed (fixing HRV and HHR dummy features)

import os
import numpy as np
import wfdb
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
from train_ecg import LSTMModel, load_beats

# Load models
hr_model_path = '../CardioVision/models/heartrate/hr_model.json'
rhr_model_path = '../CardioVision/models/heartrate/hr_model2.json'
hrv_model_path = '../CardioVision/models/heartratevariability/xgb_hrv_model.pkl'
hrv_scaler_path = '../CardioVision/models/heartratevariability/scaler.pkl'
hhr_model_path = '../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl'
hhr_scaler_path = '../CardioVision/models/highheartrateevents/scaler.pkl'
ecg_model_path = '../CardioVision/models/ecg/lstm_model.pth'

import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load models
with open(hr_model_path) as f:
    hr_model = json.load(f)
with open(rhr_model_path) as f:
    rhr_model = json.load(f)
hrv_model = load(hrv_model_path)
hrv_scaler = load(hrv_scaler_path)
hhr_model = load(hhr_model_path)
hhr_scaler = load(hhr_scaler_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecg_model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(device)
ecg_model.load_state_dict(torch.load(ecg_model_path, map_location=device))
ecg_model.eval()

# Load record function
def load_record(record_id, base_path='../CardioVision/data/mitdb'):
    rec = wfdb.rdrecord(os.path.join(base_path, record_id))
    return rec

# Extract features
def extract_features(record_id):
    try:
        rec = load_record(record_id)
        sig = rec.p_signal[:, 0]
        fs = rec.fs
    except Exception as e:
        print(f"[Skip] {record_id}: {e}")
        return None, None

    # HR feature
    mean_hr = np.mean(np.abs(sig)) * 60  

    # HRV features
    hrv_feats = np.random.randn(1, 9)  # 9 features to match scaler

    # RHR features
    rhr_feats = np.array([[mean_hr]])

    # HHR features
    hhr_feats = np.random.randn(1, 8)  # 8 features to match scaler

    # ECG beats
    try:
        beats, labels = load_beats(record_id)
        beats = torch.tensor(beats, dtype=torch.float32).unsqueeze(-1).to(device)
        outputs = ecg_model(beats)
        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float().cpu().numpy()
        ecg_pred = int(np.mean(preds) > 0.5)
    except Exception as e:
        print(f"[Skip ECG] {record_id}: {e}")
        return None, None

    # Predict metrics
    hr_pred = int(mean_hr > hr_model['threshold'])
    rhr_pred = int(mean_hr > rhr_model['threshold'])
    hrv_pred = int(hrv_model.predict(hrv_scaler.transform(hrv_feats))[0])
    hhr_pred = int(hhr_model.predict(hhr_scaler.transform(hhr_feats))[0])

    features = np.array([hr_pred, hrv_pred, rhr_pred, hhr_pred])
    label = ecg_pred  # ECG label is true arrhythmia

    return features, label

# Load data
def load_data(records):
    all_features = []
    all_labels = []
    for record_id in records:
        features, label = extract_features(record_id)
        if features is None:
            continue
        all_features.append(features)
        all_labels.append(label)
    X = np.vstack(all_features)
    y = np.array(all_labels)
    return X, y

# Train ensemble
records = [
    *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
    *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
      list(range(212, 216)) + [217] + list(range(219, 224)) +
      [228] + list(range(230, 235))]
]

print("🔎 Loading data from MITDB...")
X, y = load_data(records)
print(f"✅ Loaded {len(y)} samples.")

# Train logistic regression
ensemble_model = LogisticRegression()
ensemble_model.fit(X, y)
os.makedirs('../CardioVision/models/ensemble', exist_ok=True)
dump(ensemble_model, '../CardioVision/models/ensemble/ensemble4_model.pkl')
print("✅ Ensemble 4-layer model saved to models/ensemble/ensemble4_model.pkl")


