"""
Random Forest High Heart Rate (HHR) Detection - INCART Evaluation
-----------------------------------------------------------------
This script tests the improved Random Forest model (rf_hhr2) for High Heart Rate (HHR) detection using the INCART dataset.

Description:
- Loads a pre-trained Random Forest model (rf_hhr2) for detecting high heart rate events.
- Extracts RR interval-based features from ECG records in the INCART dataset.
- Labels each window based on a high heart rate threshold (150 bpm).
- Evaluates the model's performance using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/INCART/INCART_HHR.txt
"""

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

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features_from_hr_series(hr_series, window_size=10, threshold_bpm=150):
    """
    Extracts high heart rate (HHR) features from the heart rate (HR) series.

    Args:
        hr_series (array): Array of heart rate values (bpm).
        window_size (int): Size of the sliding window for feature extraction.
        threshold_bpm (int): Threshold for detecting high heart rate (HHR).

    Returns:
        np.array: Feature matrix for HHR detection.
    """
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

# -------------------------------
# LABELING FUNCTION
# -------------------------------
def label_high_hr_events(hr_series, window_size=10, threshold_bpm=150):
    """
    Labels high heart rate (HHR) events based on the HR threshold.

    Args:
        hr_series (array): Array of heart rate values (bpm).
        window_size (int): Size of the sliding window for labeling.
        threshold_bpm (int): Threshold for detecting high heart rate (HHR).

    Returns:
        np.array: Array of labels (1: HHR detected, 0: No HHR).
    """
    return np.array([
        int(np.all(hr_series[i:i + window_size] > threshold_bpm))
        for i in range(len(hr_series) - window_size)
    ])

# -------------------------------
# LOAD HR DATA (INCART)
# -------------------------------
def load_hr(record, base_path='../CardioVision/data/incart/files'):
    """
    Loads ECG data and computes heart rate (HR) series from RR intervals.

    Args:
        record (str): Record filename (without extension).
        base_path (str): Path to the INCART data directory.

    Returns:
        np.array: Array of heart rate values (bpm).
    """
    rec = wfdb.rdrecord(os.path.join(base_path, record))
    ecg_signal = rec.p_signal[:, 0]  # default to first lead
    r_peaks = np.where(np.diff(np.sign(ecg_signal - np.mean(ecg_signal))) > 0)[0]
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # in ms
    hr_series = 60000 / rr_intervals
    return hr_series

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_record(record):
    """
    Evaluates the Random Forest HHR model on a single INCART record.

    Args:
        record (str): Record filename (without extension).
    """
    try:
        hr_series = load_hr(record)
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    if len(hr_series) < 20:
        print(f"[Skip] {record}: Too short ECG")
        return

    # Feature extraction and prediction
    X = extract_features_from_hr_series(hr_series)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_true = label_high_hr_events(hr_series)

    # Display evaluation metrics
    print(f"\n[Test] Record {record}")
    print(f"Windows: {len(y_true)} | High HR Events (True): {np.sum(y_true)} | Predicted: {np.sum(y_pred)}")
    labels = [0, 1]
    print(classification_report(y_true, y_pred, zero_division=0, labels=labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))

# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main function to test the Random Forest HHR model on multiple INCART records.
    """
    test_records = [f"I{i:02d}" for i in range(1, 76)]  # INCART record names
    for record in test_records:
        evaluate_record(record)

if __name__ == "__main__":
    main()
