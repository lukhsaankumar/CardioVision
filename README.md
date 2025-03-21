CardioVision

Run 
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# to test current model on MIT-BIH records with feedback
python models/inference.py 


Apple HealthKit Metrics:

Apple's HealthKit offers several metrics that can be instrumental in predicting cardiac events:​
Heart Rate (HR): Measures the user's heartbeats per minute. ​
Heart Rate Variability (HRV): Assesses the variation in time intervals between heartbeats, providing insights into autonomic nervous system activity. ​
Resting Heart Rate: Estimates the user's lowest heart rate during periods of rest, serving as a significant health indicator. ​
High Heart Rate Events: Records occurrences when the user's heart rate exceeds a specified threshold, potentially signaling arrhythmic episodes. ​
Electrocardiogram (ECG): Captures electrical activity of the heart, aiding in the detection of irregular rhythms. 