import SwiftUI

struct ContentView: View {
    @StateObject private var uploader = ECGUploader()

    var body: some View {
        VStack(spacing: 20) {
            Text("CardioVision")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.top)

            Button(action: {
                uploader.fetchAndSendLatestECG()
            }) {
                Text("Send Latest ECG")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .padding(.horizontal)

            Text("Prediction Result:")
                .font(.headline)

            Text(uploader.predictionResult)
                .font(.title2)
                .foregroundColor(.purple)
                .padding()
                .multilineTextAlignment(.center)

            if !uploader.ecgSignal.isEmpty {
                ECGWaveformView(data: uploader.ecgSignal)
                    .frame(height: 200)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
            } else {
                Text("No ECG data yet")
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
    }
}

struct ECGWaveformView: View {
    var data: [Double]

    var body: some View {
        GeometryReader { geometry in
            let path = Path { p in
                let stepX = geometry.size.width / CGFloat(data.count - 1)
                let scaleY = geometry.size.height / (data.max() ?? 1.0)

                p.move(to: CGPoint(x: 0, y: geometry.size.height / 2))

                for (i, value) in data.enumerated() {
                    let x = CGFloat(i) * stepX
                    let y = geometry.size.height / 2 - CGFloat(value) * scaleY
                    p.addLine(to: CGPoint(x: x, y: y))
                }
            }

            path
                .stroke(Color.red, lineWidth: 1.5)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
