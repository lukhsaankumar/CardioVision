"""
BiLSTM ECG Model Testing Script (INCART Dataset - Fine-Tuned Model)
-------------------------------------------------------------------
This script tests a fine-tuned BiLSTM model for cardiac arrest risk classification using the INCART dataset.

Description:
- Loads a fine-tuned BiLSTM model for 3-class ECG classification (Low, Medium, High risk).
- Fine-tuned model was trained using feedback samples (True Positive, False Negative) from MIT-BIH, Holter, and INCART datasets.
- Loads and preprocesses ECG beats from the INCART dataset (Lead I).
- Each ECG beat is normalized and processed into a fixed-length segment (250 samples).
- Evaluates the model on the INCART dataset with classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Applies submodel corrections (HR, HRV, RHR, HHR) to minimize False Negatives for high-risk classification.
- Results are displayed in the console, including a detailed classification report and confusion matrix.
"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import joblib
import json

# BiLSTM Model Definition
class BiLSTMModel(nn.Module):
    """
    BiLSTM Model for ECG Classification (3-class: Low, Medium, High).
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass for the BiLSTM model.
        """
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Helper Functions
def get_r_peaks(signal, fs):
    """
    Detects R-peaks in an ECG signal.
    """
    peaks, _ = find_peaks(signal, distance=fs * 0.6)
    return peaks

def extract_hr(signal, fs):
    """
    Extracts heart rate from an ECG segment.
    """
    r_peaks = get_r_peaks(signal, fs)
    rr = np.diff(r_peaks) / fs
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if len(rr) == 0:
        return np.array([60])
    hr_series = 60 / rr
    return hr_series

def compute_hrv_features(rr_intervals):
    """
    Computes HRV features from RR intervals.
    """
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    if len(rr_intervals) < 2:
        return np.zeros(14)
    diff_rr = np.diff(rr_intervals)
    features = [
        np.sqrt(np.mean(diff_rr ** 2)),
        np.std(rr_intervals),
        np.mean(rr_intervals),
        np.median(rr_intervals),
        np.percentile(rr_intervals, 25),
        np.percentile(rr_intervals, 75),
        np.max(diff_rr) if len(diff_rr) > 0 else 0,
        np.min(diff_rr) if len(diff_rr) > 0 else 0,
        np.mean(diff_rr) if len(diff_rr) > 0 else 0,
        np.std(diff_rr) if len(diff_rr) > 0 else 0,
        np.sum(np.abs(diff_rr) > 50) / len(diff_rr) if len(diff_rr) > 0 else 0,
        len(rr_intervals),
        np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
        np.max(rr_intervals) if len(rr_intervals) > 0 else 0
    ]
    return np.array(features)

# Extract beats from INCART record
def extract_beats_from_incart(record_path, symbols, label, window_size=250):
    """
    Extracts and preprocesses ECG beats from a given INCART record.

    Args:
        record_path (str): Path to the INCART record.
        symbols (list): List of symbols representing the beat classes.
        label (int): Label assigned to the extracted beats (0: Low, 1: Medium, 2: High).
        window_size (int): Length of each beat segment (default: 250).

    Returns:
        list: Extracted ECG segments.
        list: Corresponding labels for each segment.
    """
    beats, labels = [], []
    try:
        print(f"Reading record {record_path} ...")
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal[:, 0]  # Use lead I
        scaler = StandardScaler()

        for sym, loc in zip(ann.symbol, ann.sample):
            if sym not in symbols:
                continue
            start = loc - window_size // 2
            end = loc + window_size // 2
            if start < 0 or end > len(signal):
                continue
            segment = signal[start:end]
            if len(segment) != window_size:
                segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
            segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment)) or np.any(np.isinf(segment)) or np.std(segment) == 0:
                continue
            beats.append(segment)
            labels.append(label)
        print(f"Loaded {len(beats)} beats from {record_path}")
    except Exception as e:
        print(f"Skipping {record_path} due to error: {e}")
    return beats, labels

# Test BiLSTM model on INCART
def test_incart_ecg_model():
    """
    Tests the pre-trained BiLSTM model on the INCART dataset.
    Results are displayed in the console and can be found at:
    testresults/INCART_ECG3.txt
    """
    base_path = "../CardioVision/data/incart/files"
    incart_recs = [f"I{str(i).zfill(2)}" for i in range(1, 76)]
    window_size = 250

    # Define symbols for each class
    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    print("Loading and preprocessing INCART data...")
    for rec in tqdm(incart_recs, desc="INCART Records"):
        path = os.path.join(base_path, rec)
        low, l_lbl = extract_beats_from_incart(path, low_syms, 0, window_size)
        med, m_lbl = extract_beats_from_incart(path, med_syms, 1, window_size)
        high, h_lbl = extract_beats_from_incart(path, high_syms, 2, window_size)
        all_beats.extend(low + med + high)
        all_labels.extend(l_lbl + m_lbl + h_lbl)

    if not all_beats:
        raise ValueError("No valid INCART data loaded.")

    print(f"Total samples: {len(all_beats)}")

    # Convert to tensor for model input
    X = np.array(all_beats).astype(np.float32)
    y = np.array(all_labels)
    X_tensor = torch.tensor(X).unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    try:
        ecg_model.load_state_dict(torch.load("../CardioVision/models/healthkit/bilstm_finetuned.pth", map_location=device))
    except FileNotFoundError as e:
        print(f"Skipping due to error: {e}")
        return
    ecg_model.to(device)
    ecg_model.eval()

    # Load Submodels
    with open("../CardioVision/models/heartrate/hr_model2.json") as f:
        hr_model_data = json.load(f)
    hrv_model = joblib.load("../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl")
    scaler_hrv = joblib.load("../CardioVision/models/heartratevariability/scaler.pkl")
    rhr_model = joblib.load("../CardioVision/models/restingheartrate/rhr_model.pkl")
    scaler_rhr = joblib.load("../CardioVision/models/restingheartrate/scaler.pkl")
    hhr_model = joblib.load("../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl")

    y_true, y_pred = [], []
    batch_size = 128

    print("Evaluating model...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = ecg_model(X_batch)
            ecg_output = torch.softmax(outputs, dim=-1).cpu().numpy()

            # Submodel Predictions for FN Correction
            batch_preds = []
            for j in range(X_batch.size(0)):
                segment = X_batch[j].cpu().numpy().squeeze()
                fs = 250  # Assuming resampled to 250 Hz
                hr_series = extract_hr(segment, fs)
                idx = min(len(hr_series) - 1, j)
                hr_val = hr_series[idx] if idx >= 0 else 60
                pred_hr = 1 if hr_val > 100 else 0
                sustained_high_hr = int(np.all(hr_series[max(0, idx-10):idx+1] > 130)) if idx > 0 else 0
                pred_hhr = 1 if sustained_high_hr else 0
                rhr = np.mean(hr_series) if len(hr_series) > 0 else 60
                pred_rhr = 1 if rhr > 70 else 0
                r_peaks = get_r_peaks(segment, fs)
                rr_intervals = np.diff(r_peaks) / fs * 1000
                hrv_features = compute_hrv_features(rr_intervals)
                hrv_feature_names = ['rmssd', 'sdnn', 'mean_rr', 'median_rr', 'p25', 'p75', 'max_diff', 'min_diff', 'mean_diff', 'std_diff', 'pnn50', 'nn', 'min_rr', 'max_rr']
                scaled_hrv = scaler_hrv.transform([hrv_features]) if len(hrv_features) == 14 else np.zeros((1, 14))
                hrv_pred_binary = int(hrv_model.predict(pd.DataFrame(scaled_hrv, columns=hrv_feature_names))[0]) if len(hrv_features) == 14 else 0

                # Voting to Correct False Negatives
                ecg_pred = np.argmax(ecg_output[j])
                binary_votes = [pred_hr, pred_hhr, pred_rhr, hrv_pred_binary]
                abnormal_count = sum(binary_votes)
                corrected_pred = ecg_pred
                if ecg_pred == 0 and abnormal_count >= 2:
                    corrected_pred = 1
                elif (ecg_pred in [0, 1]) and abnormal_count >= 3:
                    corrected_pred = 2
                batch_preds.append(corrected_pred)

            y_pred.extend(batch_preds)
            y_true.extend(y[i:i+batch_size])

    # Display evaluation results
    print("\nINCART Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], digits=2))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_incart_ecg_model()