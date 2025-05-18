"""
Mock HealthKit Data Generation Script
-------------------------------------
This script generates mock HealthKit data for two categories:
- Risk: Represents data with higher risk for cardiac events.
- No Risk: Represents normal, healthy data.

- Generates 100 JSON files for each category (risk and no_risk).
- Each JSON file contains randomly generated values for:
  - Resting Heart Rate (RHR)
  - Heart Rate Variability (HRV)
  - Heart Rate (HR)
  - High Heart Rate Events (HHR)
- Risk data is saved in the data/mockhealthkit/risk directory.
- No Risk data is saved in the data/mockhealthkit/no_risk directory. 
- Ranges are based on established values from the American Heart Association, Kubios HRV, WebMD, and Harvard Health
"""

import os
import json
import random

# Directories for storing generated data
base_dir = "../CardioVision/data/mockhealthkit"
risk_dir = os.path.join(base_dir, "risk")
no_risk_dir = os.path.join(base_dir, "no_risk")

# Ensure directories exist
os.makedirs(risk_dir, exist_ok=True)
os.makedirs(no_risk_dir, exist_ok=True)

def generate_sample(risk=True):
    """
    Generate a single HealthKit data sample.

    Args:
        risk (bool): If True, generates a high-risk sample. If False, a normal sample.

    Returns:
        dict: A dictionary with RHR, HRV, HR, and HHR values.
    """
    if risk:
        # High-risk values
        rhr = random.uniform(90, 110)           # Elevated resting heart rate
        hrv = random.uniform(10, 40)            # Low HRV (higher risk)
        hr = random.uniform(110, 140)           # Elevated heart rate
        hhr = random.randint(1, 5)              # High heart rate events
    else:
        # No-risk (normal) values
        rhr = random.uniform(55, 85)            # Normal resting heart rate
        hrv = random.uniform(55, 100)           # Healthy HRV
        hr = random.uniform(60, 100)            # Normal heart rate
        hhr = 0                                 # No high heart rate events

    return {
        "rhr": round(rhr, 2),
        "hrv": round(hrv, 2),
        "hr": round(hr, 2),
        "hhr": hhr
    }

# Generate 100 samples each for risk and no_risk categories
for i in range(100):
    # Save high-risk sample
    with open(os.path.join(risk_dir, f"{i:02}.json"), "w") as f:
        json.dump(generate_sample(risk=True), f, indent=4)
    
    # Save no-risk sample
    with open(os.path.join(no_risk_dir, f"{i:02}.json"), "w") as f:
        json.dump(generate_sample(risk=False), f, indent=4)

print("Mock HealthKit data generation complete.")
