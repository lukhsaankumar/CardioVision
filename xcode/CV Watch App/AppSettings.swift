/*
AppSettings: Configuration for CardioVision App
--------------------------------------------------------------
This file defines global application settings for the CardioVision app.

Description:
- Contains a single enumeration `AppSettings` with static properties.
- The `demoMode` setting controls whether the app operates in demo mode or live mode.

Features:
- Demo Mode (`demoMode`):
  - When set to `true`, the app uses mock data for testing (e.g., simulated ECG and health data).
  - When set to `false`, the app uses live HealthKit data for real-world monitoring.

Usage:
- Toggle `AppSettings.demoMode` to switch between demo and live modes.
- Demo mode is useful for testing the app without requiring an Apple Watch with ECG capabilities.

Example:
- Set `AppSettings.demoMode = false` to run the app with live HealthKit data.
*/


import Foundation

enum AppSettings {
    // When true, runs in demo mode and when false, runs live.
    static var demoMode: Bool = true
}

