"""
Heart Rate (HR) Fallback Threshold Model Training
-------------------------------------------------
This script trains a fallback HR model using HR values computed directly from ECG signals (R-peaks).

Description:
- Computes instantaneous HR from ECG signals using detected R-peaks.
- Applies a threshold to classify HR values as:
  - 0: Normal (below threshold)
  - 1: High Risk (above threshold)
- Saves the threshold value in a JSON model file.
- Saves the training HR values and their corresponding labels for reference.

Dataset:
- Source: MIT-BIH Arrhythmia Database (MITDB)
- Records used: 100-109, 111-119, 121-124
- Signal Type: ECG (Lead MLII)

Model:
- Rule-based model using a threshold (default: 160 bpm).
- Output: Binary classification (0: Normal, 1: High HR)

Results:
- The model (threshold value) is saved at:
  ../CardioVision/models/heartrate/hr_model2.json
- HR values and labels used for training are saved at:
  ../CardioVision/models/heartrate/hr_values_train2.npy
  ../CardioVision/models/heartrate/hr_labels_train2.npy
"""

import os
import json
import numpy as np
import wfdb
from scipy.signal import find_peaks

# Compute HR directly from ECG using R-peaks
def compute_hr_from_ecg(record_id, base_path='../CardioVision/data/mitdb'):
    """
    Compute instantaneous HR from R-peaks in an ECG signal.

    Args:
        record_id (str): Identifier of the ECG record.
        base_path (str): Base directory for ECG records.

    Returns:
        np.array: Array of HR values (bpm).
    """
    record_path = os.path.join(base_path, record_id)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')  # Read annotations if available

    # Use the MLII lead if available, otherwise fallback to first channel
    if 'MLII' in rec.sig_name:
        ecg_signal = rec.p_signal[:, rec.sig_name.index('MLII')]
    else:
        ecg_signal = rec.p_signal[:, 0]

    fs = rec.fs  # Sampling frequency

    # Use annotated R-peaks if available, otherwise detect peaks
    if ann:
        r_peaks = ann.sample
    else:
        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)  # ~600ms between beats
        r_peaks = peaks

    # Calculate RR intervals and convert to HR (bpm)
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    hr_series = 60 / (rr_intervals + 1e-6)  # bpm

    return hr_series

# Train a fallback HR threshold model using computed HR values
def train_hr2(records, threshold=160,
              model_path='../CardioVision/models/heartrate/hr_model2.json'):
    """
    Trains a fallback HR model using HR values computed from ECG signals.

    Args:
        records (list): List of record IDs to train on.
        threshold (int): HR threshold for classification (default: 160 bpm).
        model_path (str): Path to save the model.

    Saves:
        - Model (threshold value) as a JSON file.
        - Training HR values and labels as NumPy arrays.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the threshold rule as JSON
    with open(model_path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    print(f"Saved fallback HR model to {model_path}")

    # Collect HR data from specified records
    all_hr = []
    for rec in records:
        try:
            hr = compute_hr_from_ecg(rec)
            print(f"[Train2] Record {rec}: {len(hr)} samples, Mean HR = {np.mean(hr):.2f} bpm")
            all_hr.append(hr)
        except Exception as e:
            print(f"[Skip] {rec}: {e}")

    # Combine all HR data
    all_hr = np.concatenate(all_hr)
    hr_labels = (all_hr > threshold).astype(int)  # Apply threshold rule

    # Save training data (HR values and labels)
    np.save('../CardioVision/models/heartrate/hr_values_train2.npy', all_hr)
    np.save('../CardioVision/models/heartrate/hr_labels_train2.npy', hr_labels)
    print(f"Saved fallback training HR values and labels.")

# Main function to run the training
if __name__ == '__main__':
    training_records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))]
    ]
    train_hr2(training_records)
