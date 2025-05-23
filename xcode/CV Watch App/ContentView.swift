/*
ContentView: Main SwiftUI View for CardioVision App
---------------------------------------------------
This view is the main user interface of the CardioVision app, which provides:
- A disclaimer screen shown on initial launch for user awareness.
- Real-time heart rate display and risk prediction results.
- Interaction for recording an ECG (electrocardiogram) for further analysis.
- Conditional display of ECG analysis results.

Features:
1. Disclaimer Screen:
   - Provides a health disclaimer to inform users that this app is not a replacement for professional medical diagnosis.
   - Users must acknowledge the disclaimer to proceed.

2. Real-time Health Metrics:
   - Displays the latest heart rate fetched using HealthKit.
   - Displays the initial prediction result (No Risk, Possible Risk) based on fetched metrics.

3. ECG Recording and Analysis:
   - If the initial prediction indicates a possible risk, the user is prompted to record an ECG.
   - In demo mode, the app uses a test ECG sample.
   - The recorded ECG is analyzed, and the final prediction is displayed.

4. User Interaction:
   - The user can acknowledge the disclaimer, view health metrics, and record an ECG.
   - A spinner is shown while ECG analysis is in progress.

5. Dynamic UI Updates:
   - The view is updated in real-time using @StateObject for the ECGUploader.
   - Text color changes dynamically based on the prediction result.

Note:
- Ensure that HealthKit permissions are correctly set for heart rate, HRV, resting heart rate, and ECG data.
- The ECG functionality has not been fully tested due to inaccessibility to an Apple Watch Series 6 or above with ECG capabilities.

*/


import SwiftUI

struct ContentView: View {
    @StateObject private var uploader = ECGUploader()  // tracks ECG and heart data
    @State private var showHighRisk = false  // toggles detailed high risk UI
    @State private var showDisclaimer = true  // controls whether disclaimer is shown
    @State private var showSpinner = false  // shows spinner when processing ECG

    var body: some View {
        ZStack {
            if showDisclaimer {
                // show disclaimer screen
                Color.black
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 12) {
                        // warning icon
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 36))
                            .foregroundColor(.red)

                        // disclaimer text
                        Text("Disclaimer: This is only an Aid, not a replacement for a Professional Medical Diagnosis")
                            .font(.system(size: 19, weight: .semibold))
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 7)
                            .minimumScaleFactor(0.4)
                            .lineLimit(3)

                        // button to proceed after reading disclaimer
                        Button("Understood") {
                            withAnimation {
                                showDisclaimer = false  // hide disclaimer
                            }
                            uploader.requestAuthorization()  // ask for HealthKit permissions
                            uploader.startSendingData()  // begin periodic data updates
                        }
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.vertical, 6)
                        .padding(.horizontal, 16)
                    }
                    .padding(8)
                }
            } else {
                // show main app content
                Color.black
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        // app title
                        Text("CardioVision")
                            .font(.system(size: 22, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.top, 10)

                        // show latest heart rate
                        VStack {
                            Text("Latest Heart Rate")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(.gray)

                            Text("\(Int(uploader.latestHeartRate)) bpm")
                                .font(.system(size: 20, weight: .bold))
                                .foregroundColor(.blue)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }

                        // show initial prediction result
                        VStack {
                            Text("Prediction Result")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(.gray)
                                .padding(.bottom, 2)

                            Text(formatInitialPredictionResult(uploader.predictionResult))
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(predictionColor(for: uploader.predictionResult))
                                .multilineTextAlignment(.center)
                                .lineLimit(2)
                                .minimumScaleFactor(0.7)
                                .padding(.horizontal, 8)
                        }

                        // if risk possible ask for ECG recording
                        if formatInitialPredictionResult(uploader.predictionResult).contains("Possible Risk") {
                            VStack(spacing: 8) {
                                Text("Please record an ECG for detailed heart health analysis")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.white)
                                    .multilineTextAlignment(.center)
                                    .padding(.top, 10)

                                Button("ECG Recorded") {
                                    if AppSettings.demoMode {
                                        uploader.sendTestECGSample()  // use demo data
                                    } else {
                                        uploader.fetchECGSample()  // grab real ECG data
                                    }
                                    showSpinner = true  // show spinner while processing
                                    showHighRisk = true  // show high risk section

                                    // stop spinner after 3 seconds
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                                        showSpinner = false
                                    }
                                }
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(.white)
                                .padding(.vertical, 5)
                                .padding(.horizontal, 10)
                                .cornerRadius(27)

                                // if user recorded ECG show spinner or final prediction
                                if showHighRisk {
                                    if showSpinner {
                                        ProgressView()
                                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                            .scaleEffect(1.2)
                                            .padding(.top, 4)
                                    } else {
                                        Text(uploader.finalPrediction)
                                            .font(.system(size: 16, weight: .semibold))
                                            .foregroundColor(.red)
                                            .multilineTextAlignment(.center)
                                            .padding(.top, 4)
                                    }
                                }
                            }
                        }

                        Spacer()
                    }
                    .padding()
                }
            }
        }
    }

    // parse initial prediction from JSON string
    private func formatInitialPredictionResult(_ jsonString: String) -> String {
        guard let data = jsonString.data(using: .utf8) else {
            return "No Risk"
        }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let prediction = json["initialPrediction"] as? String {
            return prediction
        }
        return "No Risk"
    }

    // choose text color based on prediction value
    private func predictionColor(for jsonString: String) -> Color {
        guard let data = jsonString.data(using: .utf8) else {
            return .green
        }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let prediction = json["initialPrediction"] as? String {
            return prediction.contains("No Risk") ? .green : .orange
        }
        return .green
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
