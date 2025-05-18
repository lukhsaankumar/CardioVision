"""
Resting Heart Rate (RHR) Model - INCART Evaluation
----------------------------------------------------
This script tests the Resting Heart Rate (RHR) model using the INCART dataset.

Description:
- Loads a pre-trained RHR model for classifying resting heart rate values.
- Extracts RR intervals from ECG signals in the INCART dataset.
- Computes RHR values (average HR over relaxed intervals) from the RR intervals.
- Classifies each RHR value as normal or high-risk based on a threshold.
- Evaluates the model using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/incart/INCART_RHR.txt
"""

import os
import wfdb
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import find_peaks

# --- Extract RR Intervals ---
def extract_rr_intervals(ecg_signal, fs):
    """
    Extracts RR intervals (in ms) from an ECG signal using peak detection.

    Args:
        ecg_signal (array): ECG signal data.
        fs (int): Sampling frequency of the ECG signal.

    Returns:
        array: RR intervals in milliseconds.
    """
    # Detect peaks (R-peaks) in the ECG signal
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert peak intervals to milliseconds
    return rr_intervals

# --- Compute RHR Values ---
def compute_rhr(rr_intervals, window_rr_count=60):
    """
    Computes Resting Heart Rate (RHR) values from RR intervals.

    Args:
        rr_intervals (array): Array of RR intervals (in ms).
        window_rr_count (int): Number of RR intervals per RHR calculation.

    Returns:
        list: List of computed RHR values.
    """
    rhr_values = []  # List to store computed RHR values
    step = window_rr_count

    # Slide a window across the RR intervals and calculate RHR
    for i in range(0, len(rr_intervals) - step, step):
        segment = rr_intervals[i:i + step]
        if len(segment) > 0:
            mean_hr = 60000 / np.mean(segment)  # Convert RR intervals to HR (bpm)
            std_hr = np.std(60000 / segment)   # Standard deviation of HR
            if std_hr < 15:  # Threshold for stable (resting) HR
                rhr_values.append(mean_hr)
    return rhr_values

# --- Label RHR Values ---
def label_rhr_values(rhr_values, threshold=75):
    """
    Labels RHR values as high-risk (1) or normal (0) based on the threshold.

    Args:
        rhr_values (list): List of computed RHR values.
        threshold (float): Threshold for high-risk classification.

    Returns:
        list: List of binary labels (0: Normal, 1: High Risk).
    """
    # Classify each RHR value based on the threshold
    return [1 if hr > threshold else 0 for hr in rhr_values]

# --- Evaluate Model ---
def test_rhr_model_incart():
    """
    Tests the RHR model on the INCART dataset.
    """
    # Define the base path for the INCART dataset
    base_path = "../CardioVision/data/incart/files"
    record_ids = [f"I{i:02d}" for i in range(1, 76)]  # INCART record identifiers

    features, labels = [], []

    # Load each record in the INCART dataset
    for rec_id in record_ids:
        path = os.path.join(base_path, rec_id)
        try:
            rec = wfdb.rdrecord(str(path))
            ecg = rec.p_signal[:, 0]  # Use the first channel (Lead I)
            rr_intervals = extract_rr_intervals(ecg, rec.fs)
            rhr_vals = compute_rhr(rr_intervals)  # Calculate RHR values
            rhr_labels = label_rhr_values(rhr_vals)  # Label each RHR value
            features.extend(np.array(rhr_vals).reshape(-1, 1))
            labels.extend(rhr_labels)
        except Exception as e:
            print(f"[Skip] {rec_id}: {e}")

    # Ensure valid data was extracted
    if not features:
        print("No valid RHR data extracted. Check ECG signal quality or segment length.")
        return

    X = np.array(features)
    y = np.array(labels)

    # Load the trained RHR model and scaler
    model_path = "../CardioVision/models/restingheartrate/rhr_model.pkl"
    scaler_path = "../CardioVision/models/restingheartrate/scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Please train it first.")
        return

    model = load(model_path)
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    # Evaluate the model performance
    acc = accuracy_score(y, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["Normal", "High Risk"], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds, labels=[0, 1]))

# --- Run the test ---
if __name__ == "__main__":
    test_rhr_model_incart()
