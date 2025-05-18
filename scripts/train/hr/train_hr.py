"""
Heart Rate (HR) Threshold Model Training
-----------------------------------------
This script trains a rule-based model for heart rate (HR) classification using a threshold value.

Description:
- Loads HR values from specified MIMIC-III ECG records.
- Applies a threshold to classify HR values as:
  - 0: Normal (below threshold)
  - 1: High Risk (above threshold)
- Saves the threshold value in a JSON model file.
- Saves the training HR values and their corresponding labels for reference.

Dataset:
- Source: MIMIC-III Waveform Database (MIMIC3)
- Records used: 3000003, 3000105
- Signal Type: HR (Heart Rate) signal

Model:
- Rule-based model using a threshold (default: 160 bpm).
- Output: Binary classification (0: Normal, 1: High HR)

Results:
- The model (threshold value) is saved at:
  ../CardioVision/models/heartrate/hr_model.json
- HR values and labels used for training are saved at:
  ../CardioVision/models/heartrate/hr_values_train.npy
  ../CardioVision/models/heartrate/hr_labels_train.npy
"""

import os
import json
import numpy as np
import wfdb

# Load HR values from the specified record
def load_hr(record_id, base_path='../CardioVision/data/mimic3wdb/1.0'):
    """
    Loads HR values from a specified MIMIC-III record.

    Args:
        record_id (str): Identifier of the MIMIC-III record.
        base_path (str): Base directory for MIMIC-III data.

    Returns:
        np.array: Array of HR values (bpm).
    """
    subdir = record_id[:2]
    rec_path = os.path.join(base_path, subdir, record_id, record_id + 'n')
    rec = wfdb.rdrecord(rec_path)
    try:
        idx = rec.sig_name.index('HR')
    except ValueError:
        raise RuntimeError(f"'HR' channel not found in {rec_path}")
    return rec.p_signal[:, idx]

# Train a threshold-based HR model
def train_hr(records, threshold=160,
             model_path='../CardioVision/models/heartrate/hr_model.json'):
    """
    Trains a rule-based HR model using a threshold.

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
    print(f"Saved HR threshold model to {model_path}")

    # Collect HR data from specified records
    all_hr = []
    for rec in records:
        hr = load_hr(rec)
        print(f"[Train] Record {rec}: {len(hr)} samples, Mean HR = {np.mean(hr):.2f} bpm")
        all_hr.append(hr)

    # Combine all HR data
    all_hr = np.concatenate(all_hr)
    hr_labels = (all_hr > threshold).astype(int)  # Apply threshold rule

    # Save training data (HR values and labels)
    np.save('../CardioVision/models/heartrate/hr_values_train.npy', all_hr)
    np.save('../CardioVision/models/heartrate/hr_labels_train.npy', hr_labels)
    print(f"Saved training HR values and labels.")

# Main function to run the training
if __name__ == '__main__':
    training_records = ['3000003', '3000105']
    train_hr(training_records)
