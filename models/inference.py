import os
import wfdb
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from train_lstm import LSTMModel, load_beats

# Load trained model
def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load and preprocess individual ECG record
def load_ecg(record, window_size=250):
    # Load record and annotations
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', extension='atr')

    signal = rec.p_signal[:,0]
    beats = []
    labels = []

    # Normal symbols according to MIT-BIH
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
    with torch.no_grad():
        outputs = model(beats)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
    return predicted

# Evaluate model on a single record
def evaluate(model, record):
    beats, labels = load_ecg(record)

    if beats.size(0) == 0:
        print(f"No valid beats in record {record}")
        return None

    predictions = predict(model, beats)
    
    # Convert tensors to numpy arrays
    predictions = predictions.numpy()
    labels = labels.numpy()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    print(f"\nEvaluation on record {record}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc

# Main function for testing multiple records
def main():
    model_path = '../models/lstm_model.pth'
    model = load_model(model_path, input_size=1, hidden_size=128, num_layers=3, output_size=1)
    
    # Test on multiple records
    test_records = ['100', '101', '103', '105', '106']
    
    for record in test_records:
        evaluate(model, record)

if __name__ == "__main__":
    main()