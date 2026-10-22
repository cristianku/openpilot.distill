import Foundation

struct DistillStatus: Decodable {
    struct SystemInfo: Decodable {
        struct CPU: Decodable { let total: Double? }
        struct GPU: Decodable { let name: String?; let utilization: Double?; let temperature: Double? }
        struct Memory: Decodable { let used_percent: Double?; let total_gb: Double? }
        struct Disk: Decodable { let free_gb: Double?; let total_gb: Double? }
        let cpu: CPU?
        let gpu: GPU?
        let memory: Memory?
        let disk: Disk?
    }
    struct Job: Decodable, Identifiable {
        let id: String
        let label: String?
        let state: String
        let summary: String?
    }
    struct Training: Decodable, Identifiable {
        var id: String { name }
        let name: String
        let status: String
        let percent: Double?
        let steps_done: Int?
        let steps_total: Int?
    }
    struct Comma: Decodable {
        let reachable: Bool?
        let onroad: Bool?
    }
    let code: String?
    let system: SystemInfo
    let jobs: [Job]
    let training: [Training]
    let comma: Comma?
}

final class NoRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

enum DistillAPI {
    static func status(url: URL, token: String) async throws -> DistillStatus {
        let target = url.appending(path: "api/status")
        var request = URLRequest(url: target, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 12)
        request.setValue(token, forHTTPHeaderField: "X-Distill-Token")
        let session = URLSession(configuration: .ephemeral, delegate: NoRedirect(), delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (bytes, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard response.statusCode == 200 else {
            throw APIError.httpStatus(response.statusCode)
        }
        return try JSONDecoder().decode(DistillStatus.self, from: bytes)
    }
}

enum APIError: LocalizedError {
    case invalidResponse
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "Invalid server response."
        case .httpStatus(403): return "Access denied. Check your token."
        case .httpStatus(let code): return "Server returned HTTP \(code)."
        }
    }
}

@MainActor
final class DashboardModel: ObservableObject {
    @Published var snapshot: DistillStatus?
    @Published var error: String?
    @Published var updated: Date?

    func refresh(url: URL, token: String) async {
        do {
            let value = try await DistillAPI.status(url: url, token: token)
            snapshot = value
            error = nil
            updated = Date()
        } catch {
            self.error = error.localizedDescription
        }
    }
}
