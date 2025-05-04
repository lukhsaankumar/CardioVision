import os
import wfdb
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

# Load model and scaler
model_path = "../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl"
scaler_path = "../CardioVision/models/highheartrateevents/scaler.pkl"
model = load(model_path)
scaler = load(scaler_path)

# Feature extraction
def extract_features_from_hr_series(hr_series, window_size=10, threshold_bpm=150):
    features = []
    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        duration_above = np.sum(window > threshold_bpm)
        sustained = int(np.all(window > threshold_bpm))
        max_hr = np.max(window)
        min_hr = np.min(window)
        avg_hr = np.mean(window)
        hr_slope = (window[-1] - window[0]) / window_size
        spike_freq = np.sum(np.diff(window) > 10)
        hr_stddev = np.std(window)
        feats = [duration_above, sustained, max_hr, min_hr, avg_hr, hr_slope, spike_freq, hr_stddev]
        features.append(feats)
    return np.array(features)

# Label sustained high-HR events
def label_high_hr_events(hr_series, window_size=10, threshold_bpm=150):
    return np.array([int(np.all(hr_series[i:i+window_size] > threshold_bpm))
                     for i in range(len(hr_series) - window_size)])

# Load HR signal from INCART
def load_hr(record, base_path='../CardioVision/data/incart/files'):
    rec = wfdb.rdrecord(os.path.join(base_path, record))
    ecg_signal = rec.p_signal[:, 0]  # default to first lead
    r_peaks = np.where(np.diff(np.sign(ecg_signal - np.mean(ecg_signal))) > 0)[0]
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # in ms
    hr_series = 60000 / rr_intervals
    return hr_series

# Evaluate record
def evaluate_record(record):
    try:
        hr_series = load_hr(record)
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    if len(hr_series) < 20:
        print(f"[Skip] {record}: Too short ECG")
        return

    X = extract_features_from_hr_series(hr_series)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_true = label_high_hr_events(hr_series)

    print(f"\n[Test] Record {record}")
    print(f"Windows: {len(y_true)} | High HR Events (True): {np.sum(y_true)} | Predicted: {np.sum(y_pred)}")
    labels = [0, 1]
    print(classification_report(y_true, y_pred, zero_division=0, labels=labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))

# Main
def main():
    test_records = [f"I{i:02d}" for i in range(1, 76)]
    for record in test_records:
        evaluate_record(record)

if __name__ == "__main__":
    main()
