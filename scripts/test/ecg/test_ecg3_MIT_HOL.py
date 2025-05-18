"""
BiLSTM ECG Model Testing Script (MIT-BIH & Holter Dataset)
-----------------------------------------------------------
This script tests a pre-trained BiLSTM model for ECG classification on the MIT-BIH and Holter datasets.

Description:
- Loads a pre-trained BiLSTM model for ECG classification (3-class: Low, Medium, High risk).
- Loads and preprocesses MIT-BIH and Holter ECG data.
- Evaluates the model on the combined dataset (MIT-BIH + Holter).
- Displays evaluation metrics (Classification Report, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/holter/MITBIH-HOL_ECG3.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import wfdb
from scipy.signal import resample
from tqdm import tqdm

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

# Load and preprocess ECG segments
def load_ecg_segments(record_path, symbols, label, window_size=250, target_fs=250):
    """
    Extracts and preprocesses ECG beats from a given record.

    Args:
        record_path (str): Path to the ECG record.
        symbols (list): List of symbols representing the beat classes.
        label (int): Label assigned to the extracted beats (0: Low, 1: Medium, 2: High).
        window_size (int): Length of each beat segment (default: 250).
        target_fs (int): Target sampling frequency (default: 250 Hz).

    Returns:
        tuple: Extracted ECG segments and their corresponding labels.
    """
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal[:, 0]  # First lead
        fs = record.fs
        ann_samples = annotation.sample
        ann_symbols = annotation.symbol

        segments, labels = [], []
        scaler = StandardScaler()

        for idx, (sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
            if symbol not in symbols:
                continue

            start = max(0, sample - window_size // 2)
            end = min(len(signal), sample + window_size // 2)
            segment = signal[start:end]

            if len(segment) < window_size:
                segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')

            # Resample to 250 Hz
            segment_resampled = resample(segment, window_size)

            # Normalize
            segment_normalized = scaler.fit_transform(segment_resampled.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment_normalized)) or np.any(np.isinf(segment_normalized)) or np.std(segment_normalized) == 0:
                continue

            segments.append(segment_normalized)
            labels.append(label)

        segments = np.array(segments)
        labels = np.array(labels)
        print(f"Loaded {len(segments)} beats from {os.path.basename(record_path)}")
        return segments, labels
    except Exception as e:
        print(f"Skipping {record_path}: {e}")
        return np.array([]), np.array([])

# Test model
def test_model():
    """
    Tests the pre-trained BiLSTM model on the MIT-BIH and Holter datasets.
    Results are displayed in the console and can be found at:
    testresults/holter/MITBIH-HOL_ECG3.txt
    """
    # Paths and symbols
    mitdb_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"

    # MIT-BIH and Holter Recordings
    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]

    # Beat symbols for each class
    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_segments, all_labels = [], []

    # Load Low-risk beats
    print("Loading Low-risk beats...")
    for rec in low_risk_recs:
        segments, labels = load_ecg_segments(os.path.join(mitdb_path, rec), low_syms, 0)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    # Load Medium-risk beats
    print("Loading Medium-risk beats...")
    for rec in med_risk_recs:
        segments, labels = load_ecg_segments(os.path.join(mitdb_path, rec), med_syms, 1)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    # Load High-risk beats
    print("Loading High-risk beats...")
    for rec in high_risk_recs:
        segments, labels = load_ecg_segments(os.path.join(holter_path, rec), high_syms, 2)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    if not all_segments:
        print("No valid segments collected.")
        return

    # Convert to numpy arrays
    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"Evaluating on {len(X)} beats...")

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, 250, 1]
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    model.load_state_dict(torch.load("../CardioVision/models/ecg/bilstm_model_multiclass.pth", map_location=device))
    model.to(device)
    model.eval()

    # Inference
    y_true, y_pred = [], []
    batch_size = 128

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_tensor[i:i+batch_size].numpy())

    # Evaluation results
    print("\nMIT-BIH/Holter Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_model()
