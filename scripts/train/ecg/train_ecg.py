"""
LSTM Model Training - MIT-BIH Arrhythmia Dataset
------------------------------------------------
This script trains an LSTM model to classify ECG beats as normal or arrhythmic using the MIT-BIH Arrhythmia dataset.

Description:
- Loads ECG beat segments from multiple MIT-BIH records.
- Labels each beat as normal (0) or arrhythmic (1).
- Trains an LSTM model using the loaded data.
- Evaluates model performance on a test set (Accuracy, Loss).
- Saves the trained model as 'lstm_model.pth' in the models directory.

Dataset:
- MIT-BIH Arrhythmia Database (records 100-110, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234).
- The records are labeled based on beat type annotations ('N', 'L', 'R', 'e', 'j' are considered normal).

Results:
- Model training progress and accuracy are printed for each epoch.
- The trained model is saved at: ../CardioVision/models/ecg/lstm_model.pth
"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load beat segments and labels from a record using annotations
def load_beats(record, window_size=250):
    """
    Loads beat segments and labels from a specified ECG record.

    Args:
        record (str): Record identifier (MIT-BIH record).
        window_size (int): Length of the beat segment window.

    Returns:
        tuple: Arrays of beat segments and corresponding labels.
    """
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', extension='atr')
    signal = rec.p_signal[:, 0]
    beats = []
    labels = []

    normal_symbols = ['N', 'L', 'R', 'e', 'j']  # Define normal beat symbols

    for idx, beat in enumerate(ann.sample):
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue  # Skip if window goes out of bounds
        segment = signal[start:end]
        beats.append(segment)
        labels.append(0 if ann.symbol[idx] in normal_symbols else 1)  # Label: 0 (normal), 1 (arrhythmic)
    
    return np.array(beats), np.array(labels)

# LSTM Model definition
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

# Prepare dataset by aggregating beat segments from multiple records
def prepare_data(records, window_size=250):
    """
    Prepares ECG beat data and labels from multiple records.

    Args:
        records (list): List of record identifiers.
        window_size (int): Length of the beat segment window.

    Returns:
        tuple: Tensors of beat segments and corresponding labels.
    """
    all_beats = []
    all_labels = []

    for record in records:
        beats, labels = load_beats(record, window_size=window_size)
        if beats.size == 0:
            continue
        all_beats.append(beats)
        all_labels.append(labels)

    signals = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Convert to torch tensors for training
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.float32)
    return signals, labels

# Train the LSTM model
def train_model(hidden_size=128, num_layers=3, learning_rate=0.0005, batch_size=16, num_epochs=30, window_size=250):
    """
    Trains the LSTM model on ECG beat data.

    Args:
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of training batches.
        num_epochs (int): Number of training epochs.
        window_size (int): Length of the beat segment window.
    """
    # List of MIT-BIH records used for training
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
           list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
           list(range(228, 229)) + list(range(230, 235))]
    ]
    
    signals, labels = prepare_data(records, window_size=window_size)
    print(f"Total beats: {signals.shape[0]}, Normal: {(labels==0).sum().item()}, Arrhythmia: {(labels==1).sum().item()}")

    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=labels.numpy()
    )

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    # Initialize the LSTM model
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the trained model
    save_dir = os.path.abspath('../CardioVision/models/ecg')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_model.pth'))
    print("Model saved to '../CardioVision/models/ecg/lstm_model.pth'")

if __name__ == "__main__":
    train_model()
