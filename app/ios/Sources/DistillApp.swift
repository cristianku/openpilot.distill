import SwiftUI

@main
struct DistillApp: App {
    @StateObject private var settings = ConnectionSettings()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(settings)
                .tint(.indigo)
        }
    }
}
