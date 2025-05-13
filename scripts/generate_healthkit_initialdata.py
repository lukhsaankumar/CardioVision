import os
import json
import random

# Directories
base_dir = "../CardioVision/data/mockhealthkit"
risk_dir = os.path.join(base_dir, "risk")
no_risk_dir = os.path.join(base_dir, "no_risk")

# Ensure directories exist
os.makedirs(risk_dir, exist_ok=True)
os.makedirs(no_risk_dir, exist_ok=True)

def generate_sample(risk=True):
    if risk:
        rhr = random.uniform(90, 110)           # high resting heart rate
        hrv = random.uniform(10, 40)            # low HRV
        hr = random.uniform(110, 140)           # may be elevated HR
        hhr = random.randint(1, 5)              # high heart rate events
    else:
        rhr = random.uniform(55, 85)            # normal resting HR
        hrv = random.uniform(55, 100)           # healthy HRV
        hr = random.uniform(60, 100)            # normal HR
        hhr = 0                                  # no high HR events

    return {
        "rhr": round(rhr, 2),
        "hrv": round(hrv, 2),
        "hr": round(hr, 2),
        "hhr": hhr
    }

# Generate 100 samples each for risk and no_risk
for i in range(100):
    with open(os.path.join(risk_dir, f"{i:02}.json"), "w") as f:
        json.dump(generate_sample(risk=True), f, indent=4)
    with open(os.path.join(no_risk_dir, f"{i:02}.json"), "w") as f:
        json.dump(generate_sample(risk=False), f, indent=4)

