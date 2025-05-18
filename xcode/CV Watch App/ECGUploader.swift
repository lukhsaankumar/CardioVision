/*
ECGUploader: Swift Class for Health Metrics and ECG Data Collection (HealthKit)
------------------------------------------------------------------------------
This class is designed for integrating with Apple's HealthKit to:
- Request user authorization for accessing health metrics (Heart Rate, HRV, Resting Heart Rate, ECG).
- Periodically fetch and send health metrics (HR, HRV, RHR, HHR) to a remote server.
- Collect and send real-time ECG data from HealthKit to a server for analysis.
- Provide a demo mode for testing without requiring actual HealthKit data.

Features:
1. HealthKit Integration:
   - Requests authorization for reading heart rate, HRV, resting heart rate, and ECG data.
   - Periodically fetches the latest heart rate, HRV, resting heart rate, and high heart rate events (HHR).

2. ECG Sampling:
   - Fetches the most recent ECG sample from HealthKit (for Apple Watch Series 4 or above).
   - Extracts voltage values and sends them as JSON to a server endpoint for analysis.
   - Includes a demo method (sendTestECGSample) to simulate ECG sending.

3. JSON API Communication:
   - Sends health metrics and ECG data as JSON payloads to specified server endpoints.
   - Supports sending real-time data and mock data for demo/testing.

4. Demo Mode:
   - Simulates sending hardcoded metrics (HR, HRV, RHR, HHR) for testing purposes.

Note:
- Due to inaccessibility to an Apple Watch Series 6 or above with ECG capabilities, the HealthKit ECG functionality has not been fully tested.
- Ensure the server IP address is correctly set in the URL strings (e.g., "http://[YOUR IP ADDRESS]:8000/send_all_metrics").

Usage:
- Initialize an instance of ECGUploader in your SwiftUI app.
- Call requestAuthorization() to request HealthKit access.
- Use startSendingData() to start periodic data fetching and sending.
- Use fetchECGSample() to capture and send an ECG sample on demand.
*/


import Foundation
import HealthKit
import Combine

class ECGUploader: ObservableObject {
    private let healthStore = HKHealthStore()
    private var timer: Timer?

    @Published var latestHeartRate: Double = 0.00
    @Published var latestHRV: Double = 0.00
    @Published var latestRHR: Double = 0.00
    @Published var latestHHR: Int = 0
    @Published var predictionResult: String = "Press button to predict"
    @Published var finalPrediction: String = ""
    @Published var latestECGValues: [Double] = []

    // MARK: - Request Authorization
    // Ask user permission to read heart rate, HRV, resting heart rate, and ECG data
    func requestAuthorization() {
        let typesToRead: Set = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.electrocardiogramType()
        ]

        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if success {
                print("HealthKit access granted")
            } else {
                print("HealthKit access denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }

    // MARK: - Start Periodic Data Fetch
    func startSendingData() {
        // Fetch and send metrics every 5 seconds
        timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            self.fetchAndSendAllMetrics()
        }
    }

    deinit {
        // Clean up timer when uploader is deallocated
        timer?.invalidate()
    }

    // MARK: - Fetch and Send Metrics
    // Gather all metrics concurrently and send as JSON payload
    func fetchAndSendAllMetrics() {
        let dispatchGroup = DispatchGroup()
        print("inside")

        dispatchGroup.enter()
        fetchLatestHeartRate { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchLatestHRV { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchRestingHeartRate { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchHighHeartRateEvents { dispatchGroup.leave() }

        dispatchGroup.notify(queue: .main) {
            let payload: [String: Any]
            if AppSettings.demoMode {
                // Dummy values for demo/testing
                payload = [
                    "hr": 101.78,
                    "hrv": 29.28,
                    "rhr": 113.83,
                    "hhr": 2
                ]
            } else {
                // Real HealthKit measurements
                payload = [
                    "hr": self.latestHeartRate,
                    "hrv": self.latestHRV,
                    "rhr": self.latestRHR,
                    "hhr": self.latestHHR
                ]
            }

            self.sendJSON(to: "http://[YOUR IP ADDRESS]:8000/send_all_metrics", payload: payload)
        }
    }

    // MARK: - Fetch Metrics Helpers
    // Fetch most recent heart rate sample
    func fetchLatestHeartRate(completion: @escaping () -> Void) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sort]
        ) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let bpm = sample.quantity.doubleValue(
                    for: HKUnit.count().unitDivided(by: .minute())
                )
                DispatchQueue.main.async {
                    self.latestHeartRate = bpm
                    print("HR: \(bpm) bpm")
                }
            }
            completion()
        }

        healthStore.execute(query)
    }

    // Fetch most recent HRV sample
    func fetchLatestHRV(completion: @escaping () -> Void) {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: hrvType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sort]
        ) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                DispatchQueue.main.async {
                    self.latestHRV = ms
                    print("HRV: \(ms) ms")
                }
            }
            completion()
        }

        healthStore.execute(query)
    }

    // Fetch most recent resting heart rate sample
    func fetchRestingHeartRate(completion: @escaping () -> Void) {
        guard let rhrType = HKQuantityType.quantityType(forIdentifier: .restingHeartRate) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: rhrType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sort]
        ) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let rhr = sample.quantity.doubleValue(
                    for: HKUnit.count().unitDivided(by: .minute())
                )
                DispatchQueue.main.async {
                    self.latestRHR = rhr
                    print("RHR: \(rhr) bpm")
                }
            }
            completion()
        }

        healthStore.execute(query)
    }

    // Count high HR (>100 bpm) events in last hour
    func fetchHighHeartRateEvents(completion: @escaping () -> Void) {
        guard let hrType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            completion(); return
        }

        let oneHourAgo = Calendar.current.date(byAdding: .hour, value: -1, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: oneHourAgo, end: Date(), options: [])
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)

        let query = HKSampleQuery(
            sampleType: hrType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [sort]
        ) { _, samples, _ in
            let highHRCount = samples?
                .compactMap { $0 as? HKQuantitySample }
                .filter {
                    $0.quantity.doubleValue(
                        for: HKUnit.count().unitDivided(by: .minute())
                    ) > 100
                }
                .count ?? 0

            DispatchQueue.main.async {
                self.latestHHR = highHRCount
                print("HHR Events: \(highHRCount)")
            }
            completion()
        }

        healthStore.execute(query)
    }

    // MARK: - ECG Sampling
    // Grab latest ECG sample, convert to voltages, and send to server
    func fetchECGSample() {
        print("Fetching latest ECG Sample...")
        guard HKHealthStore.isHealthDataAvailable() else {
            print("Health data not available")
            return
        }

        let ecgType = HKObjectType.electrocardiogramType()
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)

        let query = HKSampleQuery(
            sampleType: ecgType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sort]
        ) { [weak self] _, samples, error in
            guard let self = self else { return }
            if let error = error {
                print("Error fetching ECG: \(error.localizedDescription)")
                return
            }
            guard let ecgSample = samples?.first as? HKElectrocardiogram else {
                print("No ECG sample found")
                return
            }

            var values = [Double]()
            let ecgQuery = HKElectrocardiogramQuery(ecgSample) { _, result in
                switch result {
                case .measurement(let m):
                    if let v = m.quantity(for: .appleWatchSimilarToLeadI)?
                                  .doubleValue(for: .volt()) {
                        values.append(v)
                    }

                case .done:
                    DispatchQueue.main.async {
                        self.latestECGValues = values
                        let rawHz = ecgSample.samplingFrequency?
                                       .doubleValue(for: .hertz()) ?? 0
                        let freq = Int(rawHz)
                        let iso = ISO8601DateFormatter()
                                    .string(from: ecgSample.startDate)

                        self.sendECGData(
                            startTime: iso,
                            samplingFrequency: freq,
                            voltages: values
                        )
                    }

                case .error(let err):
                    print("Error processing ECG: \(err.localizedDescription)")

                @unknown default:
                    print("Unknown ECG state")
                }
            }

            self.healthStore.execute(ecgQuery)
        }

        healthStore.execute(query)
    }

    // MARK: - Send ECG Data
    private func sendECGData(startTime: String,
                             samplingFrequency: Int,
                             voltages: [Double]) {
        let payload: [String: Any] = [
            "startTime": startTime,
            "samplingFrequency": samplingFrequency,
            "voltages": voltages
        ]

        guard let url = URL(string: "http://[YOUR IP ADDRESS]:8000/send_ecg"),
              let data = try? JSONSerialization.data(withJSONObject: payload)
        else {
            print("Invalid ECG endpoint or JSON")
            return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = data

        URLSession.shared.dataTask(with: req) { respData, resp, err in
            if let err = err {
                print("ECG POST failed: \(err.localizedDescription)")
                return
            }
            if let code = (resp as? HTTPURLResponse)?.statusCode {
                print("[send_ecg] status: \(code)")
            }
            if let d = respData, let s = String(data: d, encoding: .utf8) {
                print("[send_ecg] response: \(s)")
                if let json = try? JSONSerialization.jsonObject(with: d) as? [String:Any],
                   let fp = json["finalPrediction"] as? String {
                    DispatchQueue.main.async {
                        self.finalPrediction = fp
                    }
                }
            }
        }.resume()
    }

    // MARK: - Demo ECG Test
    func sendTestECGSample() {
        guard let url = URL(string: "http://[YOUR IP ADDRESS]:8000/send_test_ecg") else {
            print("Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("text/plain", forHTTPHeaderField: "Content-Type")
        request.httpBody = "Testing".data(using: .utf8)

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("[send_test_ecg] error: \(error.localizedDescription)")
                return
            }
            if let status = (response as? HTTPURLResponse)?.statusCode {
                print("[send_test_ecg] status: \(status)")
            }
            guard let data = data else { return }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let final = json["finalPrediction"] as? String {
                print("[send_test_ecg] finalPrediction: \(final)")
                DispatchQueue.main.async {
                    self.finalPrediction = final
                }
            } else if let text = String(data: data, encoding: .utf8) {
                print("[send_test_ecg] response: \(text)")
                DispatchQueue.main.async {
                    self.finalPrediction = text
                }
            }
        }.resume()
    }

    // MARK: - Generic JSON Sender
    private func sendJSON(to urlString: String, payload: [String: Any]) {
        print("Sending to \(urlString) with payload: \(payload)")

        guard let url = URL(string: urlString),
              let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            print("Invalid URL or JSON")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Request error: \(error.localizedDescription)")
                return
            }
            if let httpResponse = response as? HTTPURLResponse {
                print("[\(urlString)] Status: \(httpResponse.statusCode)")
            }
            if let data = data,
               let responseStr = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    self.predictionResult = responseStr
                }
                print("[\(urlString)] Response: \(responseStr)")
            }
        }.resume()
    }
}
