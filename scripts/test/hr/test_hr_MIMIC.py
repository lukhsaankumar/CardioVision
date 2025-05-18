"""
Heart Rate (HR) Threshold Detection - MIMIC-III Evaluation
----------------------------------------------------------
This script tests the original HR model on the MIMIC-III dataset using the 'HR' (Heart Rate) channel directly.

Description:
- Loads a pre-trained HR model that uses a threshold-based approach for detecting high-risk HR.
- Reads HR values directly from the 'HR' channel in the MIMIC-III dataset.
- Evaluates the model by comparing HR values against the model's threshold.
- Assumes ground truth labels are determined by the same threshold (rule-based).
- Results are displayed in the console and can be found at:
  testresults/mimic3/MIMIC3_HR.txt
"""

import os
import json
import numpy as np
import wfdb
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# MODEL LOADING
# -------------------------------
def load_model(model_path='../CardioVision/models/heartrate/hr_model.json'):
    """
    Loads the HR model configuration (threshold-based) from a JSON file.

    Args:
        model_path (str): Path to the HR model JSON file.

    Returns:
        dict: Model configuration (including threshold value).
    """
    with open(model_path, 'r') as f:
        return json.load(f)

# -------------------------------
# HR LOADING (MIMIC-III)
# -------------------------------
def load_hr(record_id, base_path='../CardioVision/data/mimic3wdb/1.0'):
    """
    Loads HR values directly from the 'HR' channel in a MIMIC-III record.

    Args:
        record_id (str): MIMIC-III record ID.
        base_path (str): Path to the MIMIC-III data directory.

    Returns:
        np.array: Array of HR values.
    """
    subdir = record_id[:2]
    rec_path = os.path.join(base_path, subdir, record_id, record_id + 'n')
    rec = wfdb.rdrecord(rec_path)
    try:
        idx = rec.sig_name.index('HR')
    except ValueError:
        raise RuntimeError(f"'HR' channel not found in {rec_path}")
    return rec.p_signal[:, idx]

# -------------------------------
# HR TESTING FUNCTION
# -------------------------------
def test_hr(records, model_path='../CardioVision/models/heartrate/hr_model.json'):
    """
    Tests the HR model on multiple MIMIC-III records.

    Args:
        records (list): List of MIMIC-III record IDs.
        model_path (str): Path to the HR model JSON file.
    """
    # Load model threshold
    threshold = load_model(model_path)['threshold']
    print(f"Using HR threshold: {threshold} bpm")

    all_true, all_preds = [], []

    # Evaluate each record
    for rec in records:
        hr = load_hr(rec)
        preds = (hr > threshold).astype(int)

        # For this test, true labels are determined by the same threshold
        true_labels = (hr > threshold).astype(int)

        print(f"[Test] Record {rec}: {len(hr)} samples, High-risk count: {np.sum(preds)}")

        all_true.extend(true_labels)
        all_preds.extend(preds)

    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    # Display evaluation results
    print("\nClassification Report:")
    print(classification_report(
        all_true, all_preds,
        labels=[0, 1],  # Ensure both classes are represented
        target_names=['Normal', 'High Risk'],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds, labels=[0, 1]))

    # Save results for analysis
    np.save('../CardioVision/models/heartrate/hr_values_test.npy', all_true)
    np.save('../CardioVision/models/heartrate/hr_preds_test.npy', all_preds)
    print(f"Saved testing HR values and predictions.")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    """
    Main function to test the HR model on specific MIMIC-III records.
    """
    validation_records = ['3000003', '3000105']
    test_hr(validation_records)
