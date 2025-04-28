import os
import wfdb
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

# Paths
MODEL_PATH = "../CardioVision/models/highheartrateevents/xgb_hhr_model.pkl"
SCALER_PATH = "../CardioVision/models/highheartrateevents/scaler.pkl"
DATA_PATH = "../CardioVision/data/mitdb"

# Load model and scaler
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def compute_hr_features(rr_intervals, window_size=10, threshold=160):
    features = []
    hr_series = 60000 / rr_intervals

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        rr_window = rr_intervals[i:i + window_size]

        # Mean, max, slope
        mean_hr = np.mean(window)
        max_hr = np.max(window)
        hr_slope = (window[-1] - window[0]) / window_size

        # Count spikes and max consecutive
        spike_counts = np.sum(window > threshold)

        max_consec = 0
        count = 0
        for val in window:
            if val > threshold:
                count += 1
                max_consec = max(max_consec, count)
            else:
                count = 0

        spike_density = spike_counts / window_size
        rr_min = np.min(rr_window)

        features.append([
            mean_hr, max_hr, spike_counts,
            max_consec, hr_slope, spike_density, rr_min
        ])

    return scaler.transform(np.array(features))


# -------------------------------
# LABELING FUNCTION
# -------------------------------
def label_high_hr_events(rr_intervals, window_size=10, threshold=160):
    hr_series = 60000 / rr_intervals
    labels = []

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        count = 0
        for val in window:
            if val > threshold:
                count += 1
                if count >= 3:  # sustained
                    labels.append(1)
                    break
            else:
                count = 0
        else:
            labels.append(0)

    return np.array(labels)


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_record(record):
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_PATH, record))
        ann = wfdb.rdann(os.path.join(DATA_PATH, record), 'atr')
        if 'MLII' not in rec.sig_name:
            raise ValueError("MLII lead not found.")
        r_peaks = ann.sample
        rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # ms
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    if len(rr_intervals) < 20:
        print(f"[Skip] Not enough RR data in {record}")
        return

    X = compute_hr_features(rr_intervals)
    y_true = label_high_hr_events(rr_intervals)
    y_pred = model.predict(X)

    print(f"\n[Test] Record {record}")
    print(f"Windows: {len(y_true)} | High HR Events (True): {np.sum(y_true)} | Predicted: {np.sum(y_pred)}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))  # Avoid warning

# -------------------------------
# MAIN
# -------------------------------
def main():
    test_records = [str(i) for i in range(100, 110)]
    for record in test_records:
        evaluate_record(record)


if __name__ == "__main__":
    main()
