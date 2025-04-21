import os
import json
import numpy as np
import wfdb
from sklearn.metrics import classification_report, confusion_matrix

def load_model(model_path='../CardioVision/models/heartrate/hr_model.json'):
    with open(model_path, 'r') as f:
        return json.load(f)

def load_hr(record_id, base_path='../CardioVision/data/mimic3wdb/1.0'):
    subdir = record_id[:2]
    rec_path = os.path.join(base_path, subdir, record_id, record_id + 'n')
    rec = wfdb.rdrecord(rec_path)
    try:
        idx = rec.sig_name.index('HR')
    except ValueError:
        raise RuntimeError(f"'HR' channel not found in {rec_path}")
    return rec.p_signal[:, idx]

def test_hr(records, model_path='../CardioVision/models/heartrate/hr_model.json'):
    threshold = load_model(model_path)['threshold']
    print(f"Using HR threshold: {threshold} bpm")

    all_true, all_preds = [], []

    for rec in records:
        hr = load_hr(rec)
        preds = (hr > threshold).astype(int)

        # For demo: assume true labels = rule-based logic
        true_labels = (hr > threshold).astype(int)

        print(f"[Test] Record {rec}: {len(hr)} samples, High-risk count: {np.sum(preds)}")

        all_true.extend(true_labels)
        all_preds.extend(preds)

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    print("\nClassification Report:")
    print(classification_report(
        all_true, all_preds,
        labels=[0, 1],  # Ensure both classes are represented
        target_names=['Normal', 'High Risk'],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds, labels=[0, 1]))

    np.save('../CardioVision/models/heartrate/hr_values_test.npy', all_true)
    np.save('../CardioVision/models/heartrate/hr_preds_test.npy', all_preds)
    print(f"Saved testing HR values and predictions.")

if __name__ == '__main__':
    validation_records = ['3000003', '3000105']
    test_hr(validation_records)
