"""
LSTM ECG Model Testing Script (MIT-BIH + Holter)
------------------------------------------------
This script tests a pre-trained LSTM model for ECG classification on the MIT-BIH and Holter databases.

Description:
- Loads a pre-trained LSTM model for ECG classification (3-class: Low, Medium, High risk).
- Evaluates the model on the MIT-BIH and Holter databases.
- Calculates and displays evaluation metrics (Classification Report, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/holter/MITBIH-HOL_ECG2.txt

"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# LSTM Model Definition (same as used in training)
class LSTMModel(nn.Module):
    """
    LSTM Model for ECG Classification (3-class: Low, Medium, High).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        """
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Beat Loading Function
def load_beats(record, symbols, label, base_path, window_size=250):
    """
    Load ECG beats from a specified record.

    Args:
        record (str): Record name (e.g., '100' for MIT-BIH).
        symbols (list): List of ECG symbols representing specific conditions.
        label (int): The label assigned to the loaded beats (0: Low, 1: Medium, 2: High).
        base_path (str): Path to the dataset directory.
        window_size (int): Size of each beat segment (default: 250).

    Returns:
        np.ndarray: Array of extracted beats.
        np.ndarray: Array of corresponding labels.
    """
    try:
        path = os.path.join(base_path, record)
        print(f"Reading record {path} ...")
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
        signal = rec.p_signal[:, 0]
        beats, labels = [], []

        for sym, beat in zip(ann.symbol, ann.sample):
            if sym not in symbols:
                continue
            start = beat - window_size // 2
            end = beat + window_size // 2
            if start < 0 or end > len(signal):
                continue
            segment = signal[start:end]
            if np.isnan(segment).any():
                continue
            beats.append(segment)
            labels.append(label)
        return np.array(beats), np.array(labels)
    except Exception as e:
        print(f"⚠️ Skipping {record} due to error: {e}")
        return np.array([]), np.array([])

# Model Evaluation Function
def evaluate_model(model, records, symbols, label, path):
    """
    Evaluate the model on a list of records.

    Args:
        model (torch.nn.Module): Pre-trained LSTM model.
        records (list): List of record names.
        symbols (list): List of symbols representing the class.
        label (int): Label assigned to these symbols.
        path (str): Base path to the records.

    Returns:
        torch.Tensor: Combined beats tensor for all records.
        np.ndarray: Combined labels for all records.
    """
    all_beats, all_labels = [], []
    for rec in records:
        beats, labels = load_beats(rec, symbols, label, path)
        if beats.size > 0:
            all_beats.append(beats)
            all_labels.append(labels)
    
    if not all_beats:
        return None, None
    
    X = torch.tensor(np.concatenate(all_beats), dtype=torch.float32).unsqueeze(-1)
    y = np.concatenate(all_labels)
    return X, y

# Main Testing Function
def test_model():
    """
    Tests the pre-trained LSTM model on MIT-BIH and Holter databases.
    Results are displayed in the console and can be found at:
    testresults/holter/MITBIH-HOL_ECG2.txt
    """
    # Load the pre-trained LSTM model
    model = LSTMModel(1, 128, 3, 3)
    model.load_state_dict(torch.load("../CardioVision/models/ecg/lstm_model_multiclass.pth"))
    model.eval()

    # Paths to the MIT-BIH and Holter databases
    mit_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"

    # Record IDs for testing
    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]

    # Symbol classes
    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    # Load data for each class
    X_low, y_low = evaluate_model(model, low_risk_recs, low_syms, 0, mit_path)
    X_med, y_med = evaluate_model(model, med_risk_recs, med_syms, 1, mit_path)
    X_high, y_high = evaluate_model(model, high_risk_recs, high_syms, 2, holter_path)

    # Combine data for evaluation
    X_all, y_all = [], []
    for X, y in [(X_low, y_low), (X_med, y_med), (X_high, y_high)]:
        if X is not None:
            X_all.append(X)
            y_all.append(y)

    X_test = torch.cat(X_all, dim=0)
    y_true = np.concatenate(y_all)

    # Make predictions
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Display evaluation results
    print("\nClassification Report:\n", classification_report(y_true, preds, target_names=["Low", "Med", "High"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, preds))

if __name__ == "__main__":
    test_model()
