"""
Resting Heart Rate (RHR) Model Training Script
----------------------------------------------
This script trains a Resting Heart Rate (RHR) detection model using Logistic Regression.
- Extracts RR intervals from ECG signals to compute RHR values.
- Labels RHR values as high-risk (1) or normal (0) based on a threshold.
- Trains a Logistic Regression model using the extracted RHR values.
- Saves the trained model and scaler to the specified directory.

"""

import os
import wfdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import find_peaks

# Function to extract RR intervals from ECG signal
def extract_rr_intervals(ecg_signal, fs):
    """
    Extract RR intervals (in ms) from an ECG signal using peak detection.
    Args:
        ecg_signal (array): ECG signal data.
        fs (int): Sampling frequency of the ECG signal.

    Returns:
        np.array: Array of RR intervals (in ms).
    """
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to milliseconds
    return rr_intervals

# Function to compute RHR values from RR intervals
def compute_rhr(rr_intervals, window_size=60, fs=360):
    """
    Compute Resting Heart Rate (RHR) values from RR intervals.
    Args:
        rr_intervals (array): Array of RR intervals (in ms).
        window_size (int): Number of RR intervals per window.
        fs (int): Sampling frequency of the ECG signal.

    Returns:
        list: List of computed RHR values.
    """
    rhr_values = []
    step = window_size
    for i in range(0, len(rr_intervals) - step, step):
        segment = rr_intervals[i:i + step]
        if len(segment) > 0:
            mean_hr = 60000 / np.mean(segment)  # Convert RR intervals to BPM
            std_hr = np.std(60000 / segment)
            if std_hr < 5:  # Filter out unstable HR values
                rhr_values.append(mean_hr)
    return rhr_values

# Function to label RHR values as high-risk or normal
def label_rhr_values(rhr_values, threshold=75):
    """
    Label RHR values as high-risk (1) or normal (0).
    Args:
        rhr_values (list): List of computed RHR values.
        threshold (int): Threshold for high-risk RHR.

    Returns:
        list: List of labels (0 for normal, 1 for high-risk).
    """
    return [1 if hr > threshold else 0 for hr in rhr_values]

# Function to list all available ECG segments in the dataset
def get_all_segments(record_dir):
    """
    List all valid ECG segments in the specified directory.
    Args:
        record_dir (str): Path to the directory containing ECG files.

    Returns:
        list: List of ECG segment filenames.
    """
    segments = set()
    for fname in os.listdir(record_dir):
        if fname.endswith('.dat') and '_' in fname:
            base = fname.split('.dat')[0]
            if os.path.exists(os.path.join(record_dir, base + '.hea')):
                segments.add(base)
    return sorted(list(segments))

# Process all specified records to extract RHR values
def process_records(record_ids, base_path, label_threshold=75):
    """
    Process multiple records to extract RHR values and their labels.
    Args:
        record_ids (list): List of record IDs.
        base_path (str): Base directory for the ECG records.
        label_threshold (int): Threshold for labeling RHR values.

    Returns:
        tuple: (RHR values, labels)
    """
    features, labels = [], []
    for rec_id in record_ids:
        record_dir = os.path.join(base_path, rec_id)
        if not os.path.isdir(record_dir):
            print(f"[Skip] {record_dir} does not exist.")
            continue

        segments = get_all_segments(record_dir)
        for segment in segments:
            path = os.path.join(record_dir, segment)
            try:
                rec = wfdb.rdrecord(path)
                ecg = rec.p_signal[:, 0]  # Use first channel for RHR
                rr_intervals = extract_rr_intervals(ecg, rec.fs)
                rhr_vals = compute_rhr(rr_intervals, fs=rec.fs)
                rhr_labels = label_rhr_values(rhr_vals, threshold=label_threshold)
                features.extend(np.array(rhr_vals).reshape(-1, 1))
                labels.extend(rhr_labels)
            except Exception as e:
                print(f"[Skip] {segment}: {e}")
    return np.array(features), np.array(labels)

# Train and save the RHR model
def train_rhr_model():
    """
    Train a Logistic Regression model for Resting Heart Rate (RHR) detection.
    """
    base_path = "../CardioVision/data/mimic3wdb/1.0/31"
    train_records = [
        "3100011", "3100033", "3100038", "3100069", "3100101", "3100105", 
        "3100112", "3100119", "3100124", "3100132", "3100140", "3100156"
        # Additional record IDs can be added here
    ]

    print("Extracting RHR values...")
    X, y = process_records(train_records, base_path)

    print(f"Extracted {len(X)} samples.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save model and scaler
    os.makedirs("../CardioVision/models/restingheartrate", exist_ok=True)
    joblib.dump(model, "../CardioVision/models/restingheartrate/rhr_model.pkl")
    joblib.dump(scaler, "../CardioVision/models/restingheartrate/scaler.pkl")
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_rhr_model()
