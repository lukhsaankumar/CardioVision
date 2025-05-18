"""
XGBoost High Heart Rate (HHR) Detection Model Training
------------------------------------------------------
This script trains an XGBoost model to detect high heart rate (HHR) events in ECG data. 

Description:
- Loads ECG records from the MIT-BIH Arrhythmia Database.
- Extracts high heart rate features using RR intervals:
  - Duration above threshold
  - Sustained high HR spikes
  - Max, Min, and Avg HR in the window
  - HR slope
  - Spike frequency
- Trains an XGBoost model on these features.
- Saves the trained model and the scaler used for feature normalization.

Dataset:
- MIT-BIH Arrhythmia Database
- Records used: 100-109, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234

Model:
- XGBoost Classifier
- Input: High HR features (duration, spikes, max/min/avg HR, slope, spike frequency)
- Output: Binary classification (0: Normal, 1: High HR Event)

Results:
- Model performance (Accuracy, Precision, Recall) is printed.
- Model and scaler are saved at: 
  ../CardioVision/models/highheartrateevents/xgb_hhr_model.pkl
  ../CardioVision/models/highheartrateevents/scaler.pkl
"""

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Function to extract high HR features from RR intervals
def extract_hr_from_beats(record, window_size=10, threshold_bpm=160, required_spikes=3):
    """
    Extracts high heart rate features from an ECG record.

    Args:
        record (str): ECG record identifier.
        window_size (int): Number of beats in each window.
        threshold_bpm (int): HR threshold to define high HR.
        required_spikes (int): Number of consecutive high HR beats to label high HR event.

    Returns:
        tuple: Features (numpy array) and labels (0: Normal, 1: High HR).
    """
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    fs = rec.fs

    features = []
    labels = []

    rr_intervals = np.diff(r_peaks) / fs * 1000  # RR intervals in ms
    hr_series = 60000 / rr_intervals  # HR in bpm

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        duration_above = np.sum(window > threshold_bpm)
        sustained = 0
        count = 0

        for hr in window:
            if hr > threshold_bpm:
                count += 1
                if count >= required_spikes:
                    sustained = 1
                    break
            else:
                count = 0

        max_hr = np.max(window)
        min_hr = np.min(window)
        avg_hr = np.mean(window)
        hr_slope = (window[-1] - window[0]) / window_size
        spike_freq = np.sum(np.diff(window) > 10)

        feats = [duration_above, sustained, max_hr, min_hr, avg_hr, hr_slope, spike_freq]
        features.append(feats)
        labels.append(sustained)

    return features, labels

# Load multiple records and prepare dataset
def prepare_dataset(records):
    """
    Loads and prepares the dataset for training.

    Args:
        records (list): List of record identifiers.

    Returns:
        tuple: Feature matrix (X) and label vector (y).
    """
    all_features = []
    all_labels = []

    for record in records:
        try:
            feats, labels = extract_hr_from_beats(record)
            all_features.extend(feats)
            all_labels.extend(labels)
        except Exception as e:
            print(f"[Skip] {record}: {e}")

    return np.array(all_features), np.array(all_labels)

# Training the XGBoost HHR Model
def train_hhr_model():
    """
    Trains an XGBoost model for high heart rate (HHR) detection.
    """
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
          list(range(212, 216)) + [217] + list(range(219, 224)) + [228] + list(range(230, 235))]
    ]

    print("Extracting features from MIT-BIH...")
    X, y = prepare_dataset(records)
    print(f"Extracted {len(y)} samples.")
    print("Label Distribution:", np.bincount(y))

    # Optional: Visualize class distribution
    plt.hist(y, bins=2)
    plt.title("Class Distribution: High HR Events")
    plt.xticks([0, 1], ['Normal', 'High HR'])
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training XGBoost model for HHR detection...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Model Evaluation
    preds = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, preds, zero_division=0))

    # Save model and scaler
    os.makedirs('../CardioVision/models/highheartrateevents', exist_ok=True)
    joblib.dump(model, '../CardioVision/models/highheartrateevents/xgb_hhr_model.pkl')
    joblib.dump(scaler, '../CardioVision/models/highheartrateevents/scaler.pkl')
    print("\nModel and scaler saved to models/highheartrateevents/")

# Run training
if __name__ == "__main__":
    train_hhr_model()
