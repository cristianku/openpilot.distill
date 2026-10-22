import Foundation
import Combine
import Security

enum ServerAddress {
    // Accept the shareable URL printed by helpers/web.sh, but never persist its
    // query string. The token moves directly into the secure credential field.
    static func extractLoginLink(_ input: String) -> (url: String, token: String)? {
        guard var parts = URLComponents(string: input.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = parts.scheme?.lowercased(), scheme == "https" || scheme == "http",
              parts.user == nil, parts.password == nil, parts.fragment == nil,
              parts.path.isEmpty || parts.path == "/",
              let items = parts.queryItems, items.count == 1,
              items[0].name == "token", let token = items[0].value, !token.isEmpty else { return nil }
        parts.queryItems = nil
        parts.path = ""
        guard let clean = parts.url?.absoluteString, parse(clean, allowHTTP: true) != nil else { return nil }
        return (clean, token)
    }
    static func parse(_ input: String, allowHTTP: Bool) -> URL? {
        guard var value = URLComponents(string: input.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = value.scheme?.lowercased(), (scheme == "https" || (scheme == "http" && allowHTTP)),
              let host = value.host, !host.isEmpty,
              value.user == nil, value.password == nil,
              value.query == nil, value.fragment == nil,
              value.path.isEmpty || value.path == "/",
              let port = value.port, (1...65535).contains(port) else {
            // A standard HTTPS URL without an explicit port is valid.
            return parseDefaultPort(input, allowHTTP: allowHTTP)
        }
        value.scheme = scheme
        value.path = ""
        value.query = nil
        value.fragment = nil
        return value.url
    }

    private static func parseDefaultPort(_ input: String, allowHTTP: Bool) -> URL? {
        guard var value = URLComponents(string: input.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = value.scheme?.lowercased(), (scheme == "https" || (scheme == "http" && allowHTTP)),
              let host = value.host, !host.isEmpty, value.port == nil,
              value.user == nil, value.password == nil,
              value.query == nil, value.fragment == nil,
              value.path.isEmpty || value.path == "/" else { return nil }
        value.scheme = scheme
        value.path = ""
        return value.url
    }
}

enum TokenKeychain {
    private static let account = "server-token"
    private static let service = "ch.bodhi.distill"

    static func get() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }

    static func put(_ token: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(query as CFDictionary)
        guard !token.isEmpty else { return }
        var item = query
        item[kSecValueData as String] = Data(token.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(item as CFDictionary, nil)
    }
}

@MainActor
final class ConnectionSettings: ObservableObject {
    @Published var urlText = UserDefaults.standard.string(forKey: "serverURL") ?? ""
    @Published var token = TokenKeychain.get()
    @Published var allowHTTP = UserDefaults.standard.bool(forKey: "allowHTTP")
    @Published var connected = UserDefaults.standard.bool(forKey: "connected")

    var serverURL: URL? { ServerAddress.parse(urlText, allowHTTP: allowHTTP) }

    func save() {
        guard serverURL != nil, !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        UserDefaults.standard.set(serverURL!.absoluteString, forKey: "serverURL")
        UserDefaults.standard.set(allowHTTP, forKey: "allowHTTP")
        TokenKeychain.put(token.trimmingCharacters(in: .whitespacesAndNewlines))
        connected = true
        UserDefaults.standard.set(true, forKey: "connected")
    }

    func disconnect() {
        connected = false
        token = ""
        TokenKeychain.put("")
        UserDefaults.standard.set(false, forKey: "connected")
    }
}
