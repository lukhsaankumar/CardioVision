"""
XGBoost High Heart Rate (HHR) Detection - MIT-BIH Evaluation
------------------------------------------------------------
This script tests the initial XGBoost model for High Heart Rate (HHR) detection using the MIT-BIH dataset.

Description:
- Loads a pre-trained XGBoost model for detecting high heart rate events.
- Extracts RR interval-based features from ECG records in the MIT-BIH dataset.
- Labels each window based on a high heart rate threshold (160 bpm).
- Evaluates the model's performance using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/mitbih/MITBIH_HHR.txt
"""

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
    """
    Computes high heart rate (HHR) features from RR intervals.

    Args:
        rr_intervals (array): Array of RR intervals (in ms).
        window_size (int): Size of the window for feature extraction.
        threshold (int): Heart rate threshold for high heart rate detection.

    Returns:
        np.array: Scaled feature matrix for the model.
    """
    features = []
    hr_series = 60000 / rr_intervals  # Convert RR intervals to HR (bpm)

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        rr_window = rr_intervals[i:i + window_size]

        # Feature calculations
        mean_hr = np.mean(window)
        max_hr = np.max(window)
        hr_slope = (window[-1] - window[0]) / window_size
        spike_counts = np.sum(window > threshold)

        # Consecutive high HR count
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
    """
    Labels high heart rate (HHR) events based on the HR threshold.

    Args:
        rr_intervals (array): Array of RR intervals (in ms).
        window_size (int): Size of the window for labeling.
        threshold (int): Heart rate threshold for high heart rate detection.

    Returns:
        np.array: Array of labels (1: HHR detected, 0: No HHR).
    """
    hr_series = 60000 / rr_intervals
    labels = []

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        count = 0
        for val in window:
            if val > threshold:
                count += 1
                if count >= 3:  # Minimum of 3 consecutive high HR
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
    """
    Evaluates the XGBoost HHR model on a single MIT-BIH record.

    Args:
        record (str): Record filename (without extension).
    """
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_PATH, record))
        ann = wfdb.rdann(os.path.join(DATA_PATH, record), 'atr')
        if 'MLII' not in rec.sig_name:
            raise ValueError("MLII lead not found.")
        r_peaks = ann.sample
        rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # Convert to ms
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    if len(rr_intervals) < 20:
        print(f"[Skip] Not enough RR data in {record}")
        return

    # Feature extraction and labeling
    X = compute_hr_features(rr_intervals)
    y_true = label_high_hr_events(rr_intervals)
    y_pred = model.predict(X)

    # Display evaluation metrics
    print(f"\n[Test] Record {record}")
    print(f"Windows: {len(y_true)} | High HR Events (True): {np.sum(y_true)} | Predicted: {np.sum(y_pred)}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main function to test the XGBoost HHR model on multiple MIT-BIH records.
    """
    test_records = [str(i) for i in range(100, 110)]
    for record in test_records:
        evaluate_record(record)


if __name__ == "__main__":
    main()
