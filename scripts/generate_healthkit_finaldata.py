import os
import json
from datetime import datetime
import numpy as np
from scipy.signal import resample

# Set input and output directories
input_base_dir = "../CardioVision/data/ohca/dataset/txt"
output_base_dir = "../CardioVision/data/mockhealthkit/high_risk"

# Original and target sampling rates
original_frequency = 250
target_frequency = 512

# Create output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Start time for mock ECG recording
start_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# Process subfolders and files
subfolders = ["noROEA", "ROEA", "indeterminable"]

for folder in subfolders:
    input_dir = os.path.join(input_base_dir, folder)
    if not os.path.isdir(input_dir):
        continue

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_dir, filename)
            try:
                # Load the second column (ECG voltage)
                data = np.loadtxt(input_file_path, usecols=1)

                # Resample from 250 Hz to 512 Hz
                duration_sec = len(data) / original_frequency
                num_samples_target = int(duration_sec * target_frequency)
                resampled_voltages = resample(data, num_samples_target).tolist()

                # Build JSON object
                json_data = {
                    "startTime": start_time,
                    "samplingFrequency": target_frequency,
                    "voltages": resampled_voltages
                }

                # Save JSON file
                base_name = os.path.splitext(filename)[0]
                output_file_path = os.path.join(output_base_dir, f"{base_name}.json")
                with open(output_file_path, 'w') as json_file:
                    json.dump(json_data, json_file)

                print(f"Saved: {output_file_path}")
            except Exception as e:
                print(f"Failed to process {input_file_path}: {e}")
