"""
Resting Heart Rate (RHR) Model - MIMIC-III Evaluation
----------------------------------------------------
This script tests the Resting Heart Rate (RHR) model using the MIMIC-III dataset.

Description:
- Loads a pre-trained RHR model for classifying resting heart rate values.
- Extracts RR intervals from ECG signals in the MIMIC-III dataset.
- Computes RHR values (average HR over relaxed intervals) from the RR intervals.
- Classifies each RHR value as normal or high-risk based on a threshold.
- Evaluates the model using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/mimic3/MIMIC3_RHR.txt
"""

import os
import wfdb
import numpy as np
import joblib
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
def compute_rhr(rr_intervals, window_size=60, fs=360):
    """
    Computes Resting Heart Rate (RHR) values from RR intervals.

    Args:
        rr_intervals (array): Array of RR intervals (in ms).
        window_size (int): Number of RR intervals per RHR calculation.
        fs (int): Sampling frequency of the ECG signal.

    Returns:
        list: List of computed RHR values.
    """
    rhr_values = []  # List to store computed RHR values
    step = window_size * fs  # Calculate step size for each window

    # Slide a window across the RR intervals and calculate RHR
    for i in range(0, len(rr_intervals) - step, step):
        segment = rr_intervals[i:i + step]
        if len(segment) > 0:
            mean_hr = 60000 / np.mean(segment)  # Convert RR intervals to HR (bpm)
            std_hr = np.std(60000 / segment)   # Standard deviation of HR
            if std_hr < 5:  # Threshold for stable (resting) HR
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

# --- Load and Process Records ---
def get_all_segments(record_dir):
    """
    Retrieves all valid ECG segments in the specified directory.

    Args:
        record_dir (str): Path to the directory containing ECG segments.

    Returns:
        list: List of segment identifiers.
    """
    segments = set()
    for fname in os.listdir(record_dir):
        if fname.endswith('.dat') and '_' in fname:
            base = fname.split('.dat')[0]
            if os.path.exists(os.path.join(record_dir, base + '.hea')):
                segments.add(base)
    return sorted(list(segments))

def process_records(record_ids, base_path, label_threshold=75):
    """
    Processes multiple ECG records, extracting RHR values and labels.

    Args:
        record_ids (list): List of record identifiers.
        base_path (str): Path to the directory containing ECG records.
        label_threshold (float): Threshold for RHR classification.

    Returns:
        tuple: Arrays of extracted RHR values (features) and their labels.
    """
    features, labels = [], []

    # Load and process each record
    for rec_id in record_ids:
        record_dir = os.path.join(base_path, rec_id)
        if not os.path.isdir(record_dir):
            print(f"[Skip] {record_dir} does not exist.")
            continue

        segments = get_all_segments(record_dir)
        for segment in segments:
            path = os.path.join(record_dir, segment)
            try:
                rec = wfdb.rdrecord(path)
                ecg = rec.p_signal[:, 0]
                rr_intervals = extract_rr_intervals(ecg, rec.fs)
                rhr_vals = compute_rhr(rr_intervals, fs=rec.fs)  # Calculate RHR values
                rhr_labels = label_rhr_values(rhr_vals, threshold=label_threshold)
                features.extend(np.array(rhr_vals).reshape(-1, 1))
                labels.extend(rhr_labels)
            except Exception as e:
                print(f"[Skip] {segment}: {e}")

    return np.array(features), np.array(labels)

# --- Evaluate Model ---
def test_rhr_model():
    """
    Tests the RHR model on the MIMIC-III dataset.
    """
    base_path = "../CardioVision/data/mimic3wdb/1.0/30"
    record_dirs = [
        "3000003", "3000031", "3000051", "3000060", "3000063", "3000065", "3000086",
        "3000100", "3000103", "3000105", "3000125", "3000126", "3000142", "3000154",
        "3000189", "3000190", "3000203", "3000221", "3000282", "3000336", "3000358",
        "3000386", "3000393", "3000397", "3000428", "3000435", "3000458", "3000480",
        "3000484", "3000497", "3000531", "3000544", "3000577", "3000596", "3000598",
        "3000611", "3000686", "3000701", "3000710", "3000713", "3000714", "3000715",
        "3000716", "3000717", "3000724", "3000762", "3000775", "3000781", "3000801",
        "3000834", "3000847", "3000855", "3000858", "3000860", "3000866", "3000878",
        "3000879", "3000885", "3000912", "3000960", "3000983", "3000995", "3001002"
    ]

    print("Extracting RHR values from test records...")
    X, y = process_records(record_dirs, base_path)

    print(f"Extracted {len(X)} test samples.")

    # Load trained RHR model and scaler
    model_path = "../CardioVision/models/restingheartrate/rhr_model.pkl"
    scaler_path = "../CardioVision/models/restingheartrate/scaler.pkl"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale features and make predictions
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    # Evaluate model
    acc = accuracy_score(y, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["Normal", "High Risk"], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds, labels=[0, 1]))

# --- Run the test ---
if __name__ == "__main__":
    test_rhr_model()
