import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

def load_ecg(record):
    record = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    signal = record.p_signal[:,0]
    return signal

def extract_features(signal):
    r_peaks, _ = find_peaks(signal, distance=200)
    rr_intervals = np.diff(r_peaks) / 360 # Sampling rate = 360 Hz
    hrv = np.std(rr_intervals) # Heart Rate Variability

    return {
        'heart_rate': float(60 / np.mean(rr_intervals)),
        'hrv': float(hrv)
    }

def main():
    signal = load_ecg('100')
    features = extract_features(signal)
    print(f"Extracted features: {features}")

if __name__ == "__main__":
    main()