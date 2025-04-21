import os
import json
import numpy as np
import wfdb

def load_hr(record_id, base_path='../CardioVision/data/mimic3wdb/1.0'):
    subdir = record_id[:2]
    rec_path = os.path.join(base_path, subdir, record_id, record_id + 'n')
    rec = wfdb.rdrecord(rec_path)
    try:
        idx = rec.sig_name.index('HR')
    except ValueError:
        raise RuntimeError(f"'HR' channel not found in {rec_path}")
    return rec.p_signal[:, idx]

def train_hr(records, threshold=160,
             model_path='../CardioVision/models/heartrate/hr_model.json'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the threshold rule as JSON
    with open(model_path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    print(f"Saved HR threshold model to {model_path}")

    # Collect HR data
    all_hr = []
    for rec in records:
        hr = load_hr(rec)
        print(f"[Train] Record {rec}: {len(hr)} samples, Mean HR = {np.mean(hr):.2f} bpm")
        all_hr.append(hr)

    all_hr = np.concatenate(all_hr)
    hr_labels = (all_hr > threshold).astype(int)

    np.save('../CardioVision/models//heartrate/hr_values_train.npy', all_hr)
    np.save('../CardioVision/models/heartrate/hr_labels_train.npy', hr_labels)
    print(f"Saved training HR values and labels.")

if __name__ == '__main__':
    training_records = ['3000003', '3000105']
    train_hr(training_records)
