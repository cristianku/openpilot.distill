package ch.bodhi.distill

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

/** The background of the site (dark navy): no white flash while a page loads. */
const val SITE_BACKGROUND = 0xFF0B0F16.toInt()

private fun sameOrigin(uri: Uri, base: Uri): Boolean =
    uri.scheme == base.scheme && uri.host.equals(base.host, ignoreCase = true) && uri.port == base.port

/** One WebView for the whole app, so switching tabs keeps the page, its scroll and its sign-in. */
@SuppressLint("SetJavaScriptEnabled")
fun siteWebView(context: Context): WebView = WebView(context).apply {
    setBackgroundColor(SITE_BACKGROUND)
    settings.javaScriptEnabled = true
    settings.domStorageEnabled = true
    settings.allowFileAccess = false
    settings.allowContentAccess = false
    settings.setSupportMultipleWindows(false)
    settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
    CookieManager.getInstance().setAcceptCookie(true)
    CookieManager.getInstance().setAcceptThirdPartyCookies(this, false)
}

@Composable
fun Portal(config: Connection, webView: WebView, modifier: Modifier = Modifier) {
    val base = config.base ?: return
    var canGoBack by remember { mutableStateOf(webView.canGoBack()) }
    // Android back: the previous page of the site first, then out of the Website tab.
    BackHandler(enabled = canGoBack) { webView.goBack() }
    AndroidView(
        modifier = modifier,
        factory = { context ->
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.webViewClient = object : WebViewClient() {
                // Only our own server inside the app; other links (Hugging Face, W&B) open in the browser.
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                    if (!request.isForMainFrame) return false
                    val requestUri = request.url
                    if (sameOrigin(requestUri, Uri.parse(base))) return false
                    if (requestUri.scheme == "https" || requestUri.scheme == "http") {
                        context.startActivity(Intent(Intent.ACTION_VIEW, requestUri))
                    }
                    return true
                }

                override fun doUpdateVisitedHistory(view: WebView, url: String?, isReload: Boolean) {
                    canGoBack = view.canGoBack()
                }

                override fun onPageFinished(view: WebView, url: String?) {
                    // Keep the session cookie of a fresh sign-in even if the app is killed right after.
                    CookieManager.getInstance().flush()
                }
            }
            val current = webView.url?.let(Uri::parse)
            if (current == null || !sameOrigin(current, Uri.parse(base))) webView.loadUrl("$base/")
            webView
        },
        update = {}
    )
}
