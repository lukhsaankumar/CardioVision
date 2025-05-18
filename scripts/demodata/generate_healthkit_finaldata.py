"""
OHCA ECG Data Resampling and JSON Conversion Script
----------------------------------------------------
This script processes OHCA (Out-of-Hospital Cardiac Arrest) ECG data from text files,
resamples the ECG signals from 250 Hz to 512 Hz, and saves them in JSON format.
- Reads ECG data from specified subfolders ("noROEA", "ROEA", "indeterminable").
- Resamples the ECG signal from 250 Hz to 512 Hz for compatibility.
- Saves the processed ECG data as JSON files in the specified output directory.
- The JSON files are saved in the "mockhealthkit/high_risk" directory to simulate
  high-risk ECG data for demo purposes.
- Used for the demo as format of OHCA ECG's heavily similar to how Apple HealthKit ECG format
"""

import os
import json
from datetime import datetime
import numpy as np
from scipy.signal import resample

# Set input and output directories
input_base_dir = "../CardioVision/data/ohca/dataset/txt"
output_base_dir = "../CardioVision/data/mockhealthkit/high_risk"

# Original and target sampling rates
original_frequency = 250  # Original frequency of the input ECG data (250 Hz)
target_frequency = 512    # Target frequency for the output ECG data (512 Hz)

# Create output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Start time for mock ECG recording (current UTC time)
start_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# Subfolders containing ECG data to be processed
subfolders = ["noROEA", "ROEA", "indeterminable"]

# Process each subfolder
for folder in subfolders:
    input_dir = os.path.join(input_base_dir, folder)
    if not os.path.isdir(input_dir):
        continue  # Skip if the subfolder does not exist

    # Process each text file in the subfolder
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_dir, filename)
            try:
                # Load ECG data (second column is voltage)
                data = np.loadtxt(input_file_path, usecols=1)

                # Calculate duration and target sample count
                duration_sec = len(data) / original_frequency
                num_samples_target = int(duration_sec * target_frequency)

                # Resample from 250 Hz to 512 Hz
                resampled_voltages = resample(data, num_samples_target).tolist()

                # Build JSON object with ECG metadata
                json_data = {
                    "startTime": start_time,
                    "samplingFrequency": target_frequency,
                    "voltages": resampled_voltages
                }

                # Save processed data as JSON in the "high_risk" directory
                base_name = os.path.splitext(filename)[0]
                output_file_path = os.path.join(output_base_dir, f"{base_name}.json")
                with open(output_file_path, 'w') as json_file:
                    json.dump(json_data, json_file)

                print(f"Saved (High Risk): {output_file_path}")

            except Exception as e:
                print(f"Failed to process {input_file_path}: {e}")
