import SwiftUI
import UIKit
import WebKit

struct PortalView: View {
    let url: URL
    let token: String

    var body: some View {
        NavigationStack {
            DistillWebView(url: url, token: token)
                .id(url.absoluteString + "|" + token)
                .ignoresSafeArea(edges: .bottom)
                .navigationTitle("Website")
                .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct DistillWebView: UIViewRepresentable {
    let url: URL
    let token: String

    func makeCoordinator() -> Coordinator { Coordinator(origin: url) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = context.coordinator
        view.allowsBackForwardNavigationGestures = true
        var entry = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        entry.path = "/"
        entry.queryItems = [URLQueryItem(name: "token", value: token)]
        if let address = entry.url { view.load(URLRequest(url: address, cachePolicy: .reloadIgnoringLocalCacheData)) }
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate {
        let origin: URL
        init(origin: URL) { self.origin = origin }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let address = navigationAction.request.url else { decisionHandler(.cancel); return }
            if address.scheme?.lowercased() == origin.scheme?.lowercased(),
               address.host?.lowercased() == origin.host?.lowercased(),
               address.port == origin.port {
                decisionHandler(.allow)
            } else {
                decisionHandler(.cancel)
                if address.scheme == "https" || address.scheme == "http" {
                    DispatchQueue.main.async { UIApplication.shared.open(address) }
                }
            }
        }
    }
}
