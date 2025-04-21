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
    # Load record and its annotations (the MIT-BIH arrhythmia annotations have extension 'atr')
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', extension='atr')
    signal = rec.p_signal[:, 0]
    beats = []
    labels = []
    
    # Define what we consider as normal beat symbols.
    # You can adjust this list based on your criteria.
    normal_symbols = ['N', 'L', 'R', 'e', 'j']
    
    for idx, beat in enumerate(ann.sample):
        # Extract a window around the beat (centered on the R-peak)
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue  # skip if the window goes out of bounds
        segment = signal[start:end]
        beats.append(segment)
        # Label beat: 0 if normal, 1 if arrhythmic (i.e. not in normal_symbols)
        labels.append(0 if ann.symbol[idx] in normal_symbols else 1)
    
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
    all_beats = []
    all_labels = []
    for record in records:
        beats, labels = load_beats(record, window_size=window_size)
        if beats.size == 0:
            continue
        all_beats.append(beats)
        all_labels.append(labels)
    # Concatenate data from all records
    signals = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Convert to torch tensors; add channel dimension for LSTM input shape: (batch, seq_len, features)
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.float32)
    return signals, labels

# Train the LSTM model
def train_model(hidden_size=128, num_layers=3, learning_rate=0.0005, batch_size=16, num_epochs=30, window_size=250):
    # List of records from the MIT-BIH Arrhythmia Database
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
           list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
           list(range(228, 229)) + list(range(230, 235))]
    ]
    
    signals, labels = prepare_data(records, window_size=window_size)
    
    print(f"Total beats: {signals.shape[0]}, Normal: {(labels==0).sum().item()}, Arrhythmia: {(labels==1).sum().item()}")
    
    # Use stratified splitting to maintain the balance between classes
    train_x, test_x, train_y, test_y = train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=labels.numpy()
    )

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # Ensure the models directory exists
    save_dir = os.path.abspath('../CardioVision/models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_model.pth'))
    print("Model saved to '../CardioVision/models/ecg/lstm_model.pth'")

if __name__ == "__main__":
    train_model()
