"""
LSTM ECG Model Testing Script (MIT-BIH)
---------------------------------------
This script tests a pre-trained LSTM model for ECG classification on MIT-BIH records.

Description:
- Loads a pre-trained LSTM model for ECG classification (binary: Normal vs Arrhythmia).
- Evaluates the model on multiple MIT-BIH records.
- Calculates and displays evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
- Results are displayed in the console and can be found at:
  testresults/mitbih/MITBIH_ECG.txt

"""

import os
import wfdb
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from scripts.train.ecg.train_ecg import LSTMModel, load_beats

# Load trained model
def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    """
    Loads a pre-trained LSTM model from the specified path.

    Args:
        model_path (str): Path to the saved LSTM model file.
        input_size (int): Input size of the LSTM model.
        hidden_size (int): Number of hidden units in the LSTM.
        num_layers (int): Number of LSTM layers.
        output_size (int): Output size of the model.

    Returns:
        model (torch.nn.Module): Loaded LSTM model.
    """
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load and preprocess individual ECG record
def load_ecg(record, window_size=250):
    """
    Loads and preprocesses an ECG record from MIT-BIH for testing.

    Args:
        record (str): Record ID (e.g., "100", "101").
        window_size (int): Length of each ECG segment (default: 250).

    Returns:
        beats_tensor (torch.Tensor): Tensor of segmented ECG beats.
        labels_tensor (torch.Tensor): Corresponding labels (0 for normal, 1 for arrhythmia).
    """
    # Load record and annotations
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', extension='atr')

    signal = rec.p_signal[:, 0]
    beats = []
    labels = []

    # Define normal symbols (normal beats)
    normal_symbols = ['N', 'L', 'R', 'e', 'j']

    for idx, beat in enumerate(ann.sample):
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        beats.append(segment)
        labels.append(0 if ann.symbol[idx] in normal_symbols else 1)

    beats = np.array(beats)
    labels = np.array(labels)

    # Convert to tensors
    beats_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(-1)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return beats_tensor, labels_tensor

# Function to make predictions
def predict(model, beats):
    """
    Generates predictions for a batch of ECG beats.

    Args:
        model (torch.nn.Module): Pre-trained LSTM model.
        beats (torch.Tensor): Tensor of segmented ECG beats.

    Returns:
        torch.Tensor: Predicted labels (0 or 1).
    """
    with torch.no_grad():
        outputs = model(beats)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
    return predicted

# Evaluate model on a single record
def evaluate(model, record):
    """
    Evaluates the model on a single ECG record.

    Args:
        model (torch.nn.Module): Pre-trained LSTM model.
        record (str): Record ID (e.g., "100", "101").
    """
    beats, labels = load_ecg(record)

    if beats.size(0) == 0:
        print(f"No valid beats in record {record}")
        return None

    predictions = predict(model, beats)
    
    # Convert tensors to numpy arrays
    predictions = predictions.numpy()
    labels = labels.numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, predictions)
    except ValueError:
        roc_auc = float('nan')

    # Confusion matrix and metrics
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    try:
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        print("Warning: Confusion matrix could not be unpacked to 4 values. Only one class present.")
        tn = fp = fn = tp = 0

    # Display results
    print(f"\nEvaluation on record {record}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")

    return accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp

# Main function for testing multiple records
def main():
    """
    Main function to test the LSTM model on multiple MIT-BIH records.
    Results are displayed in the console and can be found at:
    testresults/MITBIH_ECG.txt
    """
    model_path = '../CardioVision/models/ecg/lstm_model.pth'
    model = load_model(model_path, input_size=1, hidden_size=128, num_layers=3, output_size=1)
    
    # Test on multiple records
    test_records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
           list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
           list(range(228, 229)) + list(range(230, 235))]
    ]
    
    for record in test_records:
        evaluate(model, record)

if __name__ == "__main__":
    main()
