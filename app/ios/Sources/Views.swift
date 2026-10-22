import SwiftUI

struct RootView: View {
    @EnvironmentObject private var settings: ConnectionSettings

    var body: some View {
        Group {
            if settings.connected, let url = settings.serverURL, !settings.token.isEmpty {
                TabView {
                    DashboardView(url: url, token: settings.token)
                        .tabItem { Label("Overview", systemImage: "chart.bar.xaxis") }
                    PortalView(url: url, token: settings.token)
                        .tabItem { Label("Website", systemImage: "globe") }
                    ConnectionView(editing: true)
                        .tabItem { Label("Settings", systemImage: "gearshape") }
                }
            } else {
                ConnectionView(editing: false)
            }
        }
    }
}

struct ConnectionView: View {
    @EnvironmentObject private var settings: ConnectionSettings
    let editing: Bool

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://server.example:8766", text: $settings.urlText)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .accessibilityLabel("Server address")
                    SecureField("Access token", text: $settings.token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Your server")
                } footer: {
                    Text("Use the address and token from your own distill-web server. No cloud account needed.")
                }

                Section {
                    Toggle("Allow unencrypted HTTP", isOn: $settings.allowHTTP)
                    if settings.allowHTTP {
                        Label("HTTP exposes your access token on the network. Only use on a trusted private LAN or VPN.", systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                            .font(.footnote)
                    }
                }

                Section {
                    Button(editing ? "Save connection" : "Connect") { settings.save() }
                        .disabled(settings.serverURL == nil || settings.token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if editing {
                        Button("Disconnect and forget token", role: .destructive) { settings.disconnect() }
                    }
                }
            }
            .navigationTitle(editing ? "Settings" : "Connect to Distill")
        }
    }
}

struct DashboardView: View {
    let url: URL
    let token: String
    @StateObject private var model = DashboardModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let error = model.error {
                        Label(error, systemImage: "wifi.exclamationmark")
                            .foregroundStyle(.orange).font(.footnote)
                    }
                    if let status = model.snapshot {
                        HStack {
                            Circle().fill(model.error == nil ? .green : .orange).frame(width: 8, height: 8)
                            Text(model.error == nil ? "Connected" : "Last known status")
                            Spacer()
                            if let updated = model.updated {
                                Text(updated, style: .time).foregroundStyle(.secondary)
                            }
                        }.font(.footnote)

                        LazyVGrid(columns: [.init(.flexible()), .init(.flexible())], spacing: 12) {
                            MeterCard(title: "GPU", value: percent(status.system.gpu?.utilization), detail: status.system.gpu?.name ?? "Not available")
                            MeterCard(title: "CPU", value: percent(status.system.cpu?.total), detail: "Processor")
                            MeterCard(title: "Memory", value: percent(status.system.memory?.used_percent), detail: "RAM")
                            MeterCard(title: "Disk", value: disk(status.system.disk), detail: "Free / total")
                        }
                        panel("Training") {
                            if status.training.isEmpty { empty("No active training") }
                            ForEach(status.training) { run in
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(run.name).font(.subheadline).fontWeight(.medium).lineLimit(1)
                                    if let p = run.percent {
                                        ProgressView(value: min(100, max(0, p)), total: 100)
                                        Text("\(Int(p))% • \(run.steps_done ?? 0) / \(run.steps_total ?? 0) steps")
                                            .font(.caption).foregroundStyle(.secondary)
                                    } else { Text("Running").font(.caption).foregroundStyle(.secondary) }
                                }
                            }
                        }
                        panel("Jobs") {
                            let running = status.jobs.filter { $0.state == "running" }
                            if running.isEmpty { empty("No jobs running") }
                            ForEach(running) { job in
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(job.label ?? job.id).fontWeight(.medium)
                                    if let summary = job.summary { Text(summary).font(.caption).foregroundStyle(.secondary) }
                                }
                            }
                        }
                        panel("Comma") {
                            HStack {
                                Image(systemName: "car")
                                Text(status.comma?.reachable == true ? "Reachable" : "Offline")
                                Spacer()
                                if status.comma?.onroad == true { Text("On road").foregroundStyle(.secondary) }
                            }
                        }
                        Text("Server code: \(status.code ?? "unknown")")
                            .font(.caption2).foregroundStyle(.tertiary)
                    } else if model.error == nil {
                        ProgressView("Connecting…").frame(maxWidth: .infinity, minHeight: 220)
                    }
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Distill")
            .toolbar {
                Button { Task { await model.refresh(url: url, token: token) } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh status")
            }
            .refreshable { await model.refresh(url: url, token: token) }
            .task(id: url.absoluteString + token) {
                while !Task.isCancelled {
                    await model.refresh(url: url, token: token)
                    try? await Task.sleep(for: .seconds(10))
                }
            }
        }
    }

    private func percent(_ number: Double?) -> String {
        guard let number, number.isFinite else { return "—" }
        return "\(Int(number.rounded()))%"
    }

    private func disk(_ disk: DistillStatus.SystemInfo.Disk?) -> String {
        guard let disk, let free = disk.free_gb, let total = disk.total_gb else { return "—" }
        return String(format: "%.0f / %.0f GB", free, total)
    }

    private func empty(_ value: String) -> some View {
        Text(value).foregroundStyle(.secondary).font(.subheadline)
    }

    private func panel<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

struct MeterCard: View {
    let title: String
    let value: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(size: 30, weight: .semibold, design: .rounded))
                .minimumScaleFactor(0.7).lineLimit(1)
            Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
        }
        .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}
