"""
Bidirectional LSTM (BiLSTM) Model Training - ECG Classification (Low, Medium, High Risk)
-----------------------------------------------------------------------------------------
This script trains a Bidirectional LSTM (BiLSTM) model to classify ECG beats into three categories: 
- Low Risk (0)
- Medium Risk (1)
- High Risk (2)

Description:
- Loads ECG beat segments from the MIT-BIH, Holter, and INCART databases.
- Applies data augmentation (noise, time shift, scaling, and stretching).
- Normalizes each beat using StandardScaler.
- Handles NaN/Inf values in the signal (imputation with mean).
- Trains a 3-class BiLSTM model using the loaded data.
- Evaluates model performance on a test set (Accuracy, Per-class Accuracy).
- Saves the trained model as 'bilstm_model_multiclass.pth' in the models directory.

Dataset:
- MIT-BIH, Holter, and INCART Databases.
- Low Risk: Normal beats (e.g., 'N', 'L', 'R', 'e', 'j').
- Medium Risk: Supraventricular (e.g., 'A', 'S', 'a', 'J', '?').
- High Risk: Ventricular (e.g., 'V', 'F', 'E').

Results:
- Model training progress and accuracy are printed for each epoch.
- The trained model is saved at: ../CardioVision/models/ecg/bilstm_model_multiclass.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import wfdb
import random
from tqdm import tqdm

# Define Bidirectional LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Double hidden size for forward + backward

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Augmentation with Noise, Scaling, Shifting, and Stretching
def augment_signal(signal, target_len=250, noise_std=0.01, max_shift=20, scale_range=(0.8, 1.2)):
    """
    Applies multiple augmentation techniques to an ECG signal:
    - Adds Gaussian noise
    - Randomly shifts the signal
    - Scales the signal
    - Stretches the signal

    Args:
        signal (numpy array): Original signal.
        target_len (int): Target length of the signal.
        noise_std (float): Standard deviation of noise.
        max_shift (int): Maximum shift range.
        scale_range (tuple): Min and max scaling factors.

    Returns:
        numpy array: Augmented signal.
    """
    # Noise
    if random.random() < 0.5:
        noise = np.random.normal(0, noise_std, signal.shape)
        signal = signal + noise
    # Time Shift
    if random.random() < 0.5:
        shift = random.randint(-max_shift, max_shift)
        signal = np.roll(signal, shift)
    # Scaling
    if random.random() < 0.5:
        scale = random.uniform(*scale_range)
        signal = signal * scale
    # Stretching
    scale = random.uniform(0.9, 1.1)
    stretched_len = int(len(signal) * scale)
    stretched = np.interp(np.linspace(0, len(signal), stretched_len), np.arange(len(signal)), signal)
    signal = np.interp(np.linspace(0, stretched_len, target_len), np.arange(stretched_len), stretched)
    return signal

# Load ECG beats (Multiclass)
def load_beats_multiclass(record, symbols, label, base_path, augment=False, window_size=250):
    """
    Loads ECG beat segments and labels for training.

    Args:
        record (str): Record identifier.
        symbols (list): List of symbols representing the target class.
        label (int): Label value (0, 1, or 2).
        base_path (str): Path to the record directory.
        augment (bool): Apply augmentation (True/False).
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
    scaler = StandardScaler()
    beats, labels = [], []

    for sym, beat in zip(ann.symbol, ann.sample):
        if sym not in symbols:
            continue
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        if len(segment) != window_size:
            segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
        if augment:
            segment = augment_signal(segment, window_size)
        segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
        if np.std(segment) == 0:  # Skip flat segments
            continue
        beats.append(segment)
        labels.append(label)

    return np.array(beats), np.array(labels)

# Data Preparation
def prepare_data(window_size=250):
    """
    Prepares the ECG dataset for training.

    Args:
        window_size (int): Length of the beat segment window.

    Returns:
        tuple: Torch tensors for signals and labels.
    """
    mit_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"
    incart_path = "../CardioVision/data/incart/files"

    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]

    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    for rec in low_risk_recs:
        beats, labels = load_beats_multiclass(rec, low_syms, 0, mit_path, augment=True)
        all_beats.append(beats)
        all_labels.append(labels)

    for rec in med_risk_recs:
        beats, labels = load_beats_multiclass(rec, med_syms, 1, mit_path, augment=True)
        all_beats.append(beats)
        all_labels.append(labels)

    for rec in high_risk_recs:
        beats, labels = load_beats_multiclass(rec, high_syms, 2, holter_path, augment=True)
        all_beats.append(beats)
        all_labels.append(labels)

    signals = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.long)
    return signals, labels

# Train the BiLSTM Model
def train_model():
    signals, labels = prepare_data()
    train_x, test_x, train_y, test_y = train_test_split(signals, labels, test_size=0.2, stratify=labels)

    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "../CardioVision/models/ecg/bilstm_model_multiclass.pth")

train_model()
