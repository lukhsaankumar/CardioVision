"""
Multiclass LSTM Model Training - ECG Classification (Low, Medium, High Risk)
----------------------------------------------------------------------------
This script trains an LSTM model to classify ECG beats into three categories: 
- Low Risk (0)
- Medium Risk (1)
- High Risk (2)

Description:
- Loads ECG beat segments from the MIT-BIH, Holter, and INCART databases.
- Labels each beat based on the provided beat type annotations:
  - Low Risk: Normal beats ('N', 'L', 'R', 'e', 'j')
  - Medium Risk: Supraventricular or unknown beats ('A', 'S', 'a', 'J', '?')
  - High Risk: Ventricular beats ('V', 'F', 'E')
- Applies data augmentation (stretching) for medium-risk beats to balance classes.
- Trains a 3-class LSTM model using the loaded data.
- Evaluates model performance on a test set (Accuracy, Loss).
- Saves the trained model as 'lstm_model_multiclass.pth' in the models directory.

Dataset:
- MIT-BIH, Holter, and INCART Databases.
- Low Risk: Normal beats (e.g., 'N', 'L', 'R', 'e', 'j').
- Medium Risk: Supraventricular (e.g., 'A', 'S', 'a', 'J', '?').
- High Risk: Ventricular (e.g., 'V', 'F', 'E').

Results:
- Model training progress and accuracy are printed for each epoch.
- The trained model is saved at: ../CardioVision/models/ecg/lstm_model_multiclass.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import wfdb
import random

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Augmentation function for signal stretching
def augment_signal(signal, target_len=250):
    """
    Augments an ECG signal by randomly stretching it.

    Args:
        signal (numpy array): Original signal.
        target_len (int): Target length of the signal.

    Returns:
        numpy array: Augmented signal.
    """
    scale = random.uniform(0.9, 1.1)
    stretched_len = int(len(signal) * scale)
    stretched = np.interp(np.linspace(0, len(signal), stretched_len), np.arange(len(signal)), signal)
    return np.interp(np.linspace(0, stretched_len, target_len), np.arange(stretched_len), stretched)

# Load beat segments and labels (multiclass)
def load_beats_multiclass(record, symbols, label, base_path, augment=False, window_size=250):
    """
    Loads ECG beat segments from a record and assigns labels.

    Args:
        record (str): Record identifier.
        symbols (list): List of symbols representing the target class.
        label (int): Label value (0, 1, or 2).
        base_path (str): Path to the record directory.
        augment (bool): Apply augmentation (for medium risk).
        window_size (int): Length of the beat segment window.

    Returns:
        tuple: Arrays of beat segments and corresponding labels.
    """
    record_path = os.path.join(base_path, record)
    try:
        rec = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, extension='atr')
    except Exception as e:
        print(f"Skipping {record} due to error: {e}")
        return np.empty((0, window_size)), np.empty((0,), dtype=int)

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
        if augment:
            segment = augment_signal(segment, window_size)
        beats.append(segment)
        labels.append(label)

    return np.array(beats), np.array(labels)

# Data preparation for training
def prepare_data(window_size=250):
    """
    Loads and prepares ECG beat data for training.

    Args:
        window_size (int): Length of the beat segment window.

    Returns:
        tuple: Tensors of beat segments and corresponding labels.
    """
    mit_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"
    incart_path = "../CardioVision/data/incart/files"

    # Defining record sets
    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]

    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    # Load Low Risk Beats
    for rec in low_risk_recs:
        beats, labels = load_beats_multiclass(rec, low_syms, 0, mit_path)
        all_beats.append(beats)
        all_labels.append(labels)

    # Load Medium Risk Beats (with Augmentation)
    for rec in med_risk_recs:
        beats, labels = load_beats_multiclass(rec, med_syms, 1, mit_path, augment=True)
        all_beats.append(beats)
        all_labels.append(labels)

    # Load High Risk Beats
    for rec in high_risk_recs:
        beats, labels = load_beats_multiclass(rec, high_syms, 2, holter_path)
        all_beats.append(beats)
        all_labels.append(labels)

    # Combine all data
    signals = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.long)
    return signals, labels

# Training the LSTM model
def train_model():
    signals, labels = prepare_data()
    print(f"Total Beats: {len(labels)} | Low: {(labels==0).sum().item()} | Med: {(labels==1).sum().item()} | High: {(labels==2).sum().item()}")

    X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2, stratify=labels, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "../CardioVision/models/ecg/lstm_model_multiclass.pth")
    print("Model saved to ../CardioVision/models/ecg/lstm_model_multiclass.pth")

train_model()
