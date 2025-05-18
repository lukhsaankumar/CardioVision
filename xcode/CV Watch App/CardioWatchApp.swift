/*
CardioWatchApp: Entry Point for the CardioVision App (watchOS)
--------------------------------------------------------------
This is the main entry point for the CardioVision app on Apple Watch.

Description:
- Sets up the main window for the app and loads the ContentView as the initial view.
- Leverages SwiftUI's `@main` attribute to define the app's main entry.
- The ContentView is where the user interface and functionality of the app are defined.

Features:
- Launches directly into ContentView.
- Uses SwiftUI's `WindowGroup` for a scalable, watchOS-compatible UI.
- Ensures a clean and modular app structure by isolating the main entry in this file.

Note:
- Make sure the ContentView is fully set up with the necessary UI and logic for the app.
- Ensure HealthKit permissions are correctly configured within ContentView.
*/


import SwiftUI

@main
struct CardioWatchApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}


