"""
BiLSTM ECG Model Testing Script (INCART Dataset)
------------------------------------------------
This script tests a pre-trained BiLSTM model for ECG classification on the INCART dataset.

Description:
- Loads a pre-trained BiLSTM model for ECG classification (3-class: Low, Medium, High risk).
- Loads and preprocesses INCART ECG data.
- Evaluates the model on the INCART dataset.
- Displays evaluation metrics (Classification Report, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/incart/INCART_ECG3.txt
"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# BiLSTM Model Definition
class BiLSTMModel(nn.Module):
    """
    BiLSTM Model for ECG Classification (3-class: Low, Medium, High).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
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
    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    model.load_state_dict(torch.load("../CardioVision/models/ecg/bilstm_model_multiclass.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    batch_size = 128

    print("Evaluating model...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y[i:i+batch_size])

    # Display evaluation results
    print("\nINCART Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], digits=2))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_incart_ecg_model()
