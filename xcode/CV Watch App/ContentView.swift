import SwiftUI

struct ContentView: View {
    @StateObject private var uploader = ECGUploader()
    @State private var isFlashing = false
    @State private var showHighRisk = false
    @State private var showDisclaimer = true


    var body: some View {
        ZStack {
            if showDisclaimer {
                // Disclaimer screen
                Color.black
                    .ignoresSafeArea()

                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.red)

                    Text("Disclaimer: This is only an Aid, not a replacement for a Doctor")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 16)
                        .minimumScaleFactor(0.7)
                        .lineLimit(3)
                }
            } else {
                // Main content
                Color.black
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        Text("CardioVision")
                            .font(.system(size: 22, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.top, 10)

                        VStack {
                            Text("Latest Heart Rate")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(.gray)

                            Text("\(Int(uploader.latestHeartRate)) bpm")
                                .font(.system(size: 20, weight: .bold))
                                .foregroundColor(.green)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }

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

                        if formatInitialPredictionResult(uploader.predictionResult).contains("Risk") {
                            VStack(spacing: 8) {
                                Text("Take an ECG and click 'Start' when done")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.white)
                                    .multilineTextAlignment(.center)
                                    .padding(.top, 10)

                                Button("Start") {
//                                    uploader.fetchECGSample() // Turned Off for Demo Purposes, this will actively call Healthkit to
//                                    live data
                                    
                                    uploader.sendTestECGSample()
                                    showHighRisk = true
                                }
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(.white)
                                .padding(.vertical, 8)
                                .padding(.horizontal, 24)
                                .background(Color.blue)
                                .cornerRadius(20)
                                
                                if showHighRisk {
                                    Text(uploader.finalPrediction)
                                    .font(.system(size: 18, weight: .bold))
                                    .foregroundColor(.red)
                                    .multilineTextAlignment(.center)
                                    .lineLimit(2)
                                    .minimumScaleFactor(0.7)
                                    .padding(.horizontal, 8)

                                }


                                

                            }
                        }

                        Spacer()
                    }
                    .padding()
                }
            }
        }
        .onAppear {
            uploader.requestAuthorization()
            uploader.startSendingData()

            // Hide disclaimer after 5 Seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                withAnimation {
                    showDisclaimer = false
                }
            }
        }
    }

    private func formatInitialPredictionResult(_ jsonString: String) -> String {
        guard let data = jsonString.data(using: .utf8) else { return "Awaiting Prediction..." }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let prediction = json["initialPrediction"] as? String {
            return prediction
        }
        return "Awaiting Prediction..."
    }
    
    





    private func predictionColor(for jsonString: String) -> Color {
        guard let data = jsonString.data(using: .utf8) else { return .yellow }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let prediction = json["initialPrediction"] as? String {
            return prediction.contains("No Risk") ? .green : .orange
        }
        return .yellow
    }
}
