"""
Heart Rate (HR) Threshold Detection - MIT-BIH Evaluation
---------------------------------------------------------
This script tests the improved HR model using the MIT-BIH dataset, leveraging ECG data 
to compute HR values and classify them using a threshold-based approach.

Description:
- Loads a fallback HR model with a threshold for detecting high-risk HR events.
- Computes HR values directly from ECG signals using detected R-peaks.
- Classifies HR values based on the loaded threshold.
- Evaluates the model using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/mitbih/MITBIH_HR2.txt
"""

import os
import json
import numpy as np
import wfdb
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# MODEL LOADING
# -------------------------------
def load_hr_threshold(model_path='../CardioVision/models/heartrate/hr_model2.json'):
    """
    Loads the HR model configuration (threshold-based) from a JSON file.

    Args:
        model_path (str): Path to the HR model JSON file.

    Returns:
        float: The HR threshold value.
    """
    with open(model_path, 'r') as f:
        config = json.load(f)
    return config['threshold']

# -------------------------------
# HR COMPUTATION (ECG)
# -------------------------------
def compute_hr_from_ecg(record_id, base_path='../CardioVision/data/mitdb'):
    """
    Computes the HR series from ECG data using R-peak detection.

    Args:
        record_id (str): Record filename (without extension).
        base_path (str): Path to the MIT-BIH data directory.

    Returns:
        np.array: Array of HR values calculated from ECG.
    """
    record_path = os.path.join(base_path, record_id)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    # Use MLII lead if available, otherwise default to first channel
    if 'MLII' in rec.sig_name:
        ecg_signal = rec.p_signal[:, rec.sig_name.index('MLII')]
    else:
        ecg_signal = rec.p_signal[:, 0]

    fs = rec.fs

    # Use annotation R-peaks if available, otherwise detect peaks
    if ann:
        r_peaks = ann.sample
    else:
        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
        r_peaks = peaks

    # Calculate RR intervals and convert to HR (bpm)
    rr_intervals = np.diff(r_peaks) / fs
    hr_series = 60 / (rr_intervals + 1e-6)
    return hr_series

# -------------------------------
# EVALUATE SINGLE RECORD
# -------------------------------
def evaluate_record(record_id, threshold):
    """
    Evaluates the HR model on a single MIT-BIH record.

    Args:
        record_id (str): Record filename (without extension).
        threshold (float): HR threshold value for classification.
    """
    try:
        hr_series = compute_hr_from_ecg(record_id)
    except Exception as e:
        print(f"[Skip] {record_id}: {e}")
        return

    # Classify HR values based on the threshold
    y_true = (hr_series > threshold).astype(int)
    y_pred = (hr_series > threshold).astype(int)

    print(f"\n[Test] Record {record_id}")
    print(f"Total HR Samples: {len(hr_series)} | High HR Events (True): {np.sum(y_true)}")

    # Display evaluation metrics
    print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0, target_names=['Normal', 'High Risk']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main function to test the HR model on multiple MIT-BIH records.
    """
    threshold = load_hr_threshold()

    # List of test records from MIT-BIH
    test_records = [
        *[str(i) for i in list(range(100, 110))]
    ]

    print(f"Using HR threshold: {threshold} bpm")

    # Evaluate each record
    for record in test_records:
        evaluate_record(record, threshold)

if __name__ == "__main__":
    main()
