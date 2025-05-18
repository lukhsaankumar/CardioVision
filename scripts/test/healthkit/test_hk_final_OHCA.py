"""
BiLSTM OHCA Model Testing Script (Fine-Tuned Model)
----------------------------------------------------
This script tests a fine-tuned BiLSTM model for cardiac arrest risk classification using OHCA (Out-of-Hospital Cardiac Arrest) ECG data.

Description:
- Loads a fine-tuned BiLSTM model for 3-class ECG classification (Low, Medium, High risk).
- Fine-tuned model was trained using feedback samples (True Positive, False Negative) from MIT-BIH, Holter, and INCART datasets.
- Loads and preprocesses OHCA ECG data from JSON files in the mockhealthkit/high_risk directory.
- Each ECG segment is resampled to 250 Hz, normalized, and segmented using a sliding window approach.
- The model is evaluated on the OHCA dataset with classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console, and the model's predictions are categorized and displayed by class.
"""


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from tqdm import tqdm

# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Load and Preprocess OHCA JSON Data
def load_ohca_segments(json_path, window_size=250, stride=125, target_fs=250, label=2):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        voltages = np.array(data['voltages'], dtype=np.float32)
        sampling_frequency = data.get('samplingFrequency', 512)
        if np.any(np.isnan(voltages)) or np.any(np.isinf(voltages)):
            print(f"Skipping {json_path} due to NaN/Inf in voltages")
            return [], [], []
        if len(voltages) < window_size:
            print(f"Skipping {json_path} due to insufficient length: {len(voltages)} < {window_size}")
            return [], [], []

        segments, labels, metadata = [], [], []
        scaler = StandardScaler()

        for idx, start in enumerate(range(0, len(voltages) - window_size + 1, stride)):
            segment = voltages[start:start + window_size]
            if len(segment) != window_size:
                continue

            # Resample to 250 Hz
            segment_resampled = resample(segment, window_size)
            segment_normalized = scaler.fit_transform(segment_resampled.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment_normalized)) or np.any(np.isinf(segment_normalized)) or np.std(segment_normalized) == 0:
                continue

            segments.append(segment_normalized)
            labels.append(label)
            metadata.append((os.path.basename(json_path), idx))

        segments = np.array(segments)
        labels = np.array(labels)
        print(f"Processed {json_path}: {len(segments)} segments")
        return segments, labels, metadata
    except Exception as e:
        print(f"Skipping {json_path} due to error: {e}")
        return [], [], []

# Evaluate Model
def test_ohca_model():
    base_path = "../CardioVision/data/mockhealthkit/high_risk"
    json_files = [f for f in os.listdir(base_path) if f.endswith(".json")]
    window_size = 250
    stride = 125
    target_fs = 250
    high_risk_label = 2
    target_samples = 3574  # Match projected OHCA sample size

    all_segments, all_labels, all_metadata = [], [], []

    print("Loading and preprocessing OHCA data...")
    for json_file in tqdm(json_files, desc="Processing JSONs"):
        json_path = os.path.join(base_path, json_file)
        segments, labels, metadata = load_ohca_segments(json_path, window_size, stride, target_fs, high_risk_label)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)
            all_metadata.extend(metadata)

    if not all_segments:
        print("No valid segments collected for testing.")
        return

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"Total samples: {len(X)} (High: {(y == high_risk_label).sum()})")

    # Subsample or pad to match target samples if necessary
    if len(X) > target_samples:
        indices = np.random.choice(len(X), target_samples, replace=False)
        X = X[indices]
        y = y[indices]
        all_metadata = np.array(all_metadata, dtype=object)[indices]
    elif len(X) < target_samples:
        print(f"OHCA dataset has only {len(X)} samples, less than target {target_samples}. Padding with duplicates.")
        while len(X) < target_samples:
            indices = np.random.choice(len(X), min(target_samples - len(X), len(X)), replace=False)
            X = np.concatenate((X, X[indices]))
            y = np.concatenate((y, y[indices]))
            all_metadata = np.concatenate((all_metadata, all_metadata[indices]))

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    try:
        model.load_state_dict(torch.load("../CardioVision/models/healthkit/bilstm_finetuned.pth", map_location=device))
    except FileNotFoundError:
        print("Model file '../CardioVision/models/healthkit/bilstm_finetuned.pth' not found.")
        return
    model.to(device)
    model.eval()

    batch_size = 128
    y_true, y_pred = [], []

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y[i:i+batch_size])

    print("\nOHCA Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], labels=[0, 1, 2], digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy: {classification_report(y_true, y_pred, output_dict=True, labels=[0, 1, 2])['accuracy']:.2f}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fn_total = sum(cm[i, j] for i in range(3) for j in range(3) if i != j)
    fn_high = cm[2, 0] + cm[2, 1] if cm.shape[0] > 2 else 0
    fn_total_rate = fn_total / len(y_true) if len(y_true) > 0 else 0
    fn_high_rate = fn_high / sum(cm[2, :]) if cm.shape[0] > 2 and sum(cm[2, :]) > 0 else 0
    print(f"High-risk FN: {fn_high}/{sum(cm[2, :])} = {fn_high_rate*100:.2f}%")
    print(f"Overall FN: {fn_total}/{len(y_true)} = {fn_total_rate*100:.2f}%")

    # Log Classified Segments
    class_names = ["Low", "Med", "High"]
    classified_segments = {name: [] for name in class_names}
    for idx, pred in enumerate(y_pred):
        classified_segments[class_names[pred]].append(all_metadata[idx])

    print("\nClassified Segments:")
    for class_name in class_names:
        segments = classified_segments[class_name]
        print(f"{class_name} Classified Segments ({len(segments)}):")
        if segments:
            for segment in segments[:10]:
                print(f"  - {segment[0]}, Segment {segment[1]}")
            if len(segments) > 10:
                print(f"  ... and {len(segments) - 10} more")
        else:
            print("  - None")

if __name__ == "__main__":
    test_ohca_model()