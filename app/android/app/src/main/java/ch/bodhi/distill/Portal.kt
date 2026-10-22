package ch.bodhi.distill

import android.content.Intent
import android.net.Uri
import android.webkit.CookieManager
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun Portal(config: Connection, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val base = config.base ?: return
    // Only load our own origin. The website's HTTP-only cookie is established by a
    // one-time GET with a query token and then the server redirects to '/'.
    val entry = remember(base, config.token) {
        Uri.parse("$base/").buildUpon().appendQueryParameter("token", config.token).build().toString()
    }
    AndroidView(
        modifier = modifier,
        factory = {
            WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.allowFileAccess = false
                settings.allowContentAccess = false
                settings.setSupportMultipleWindows(false)
                settings.mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
                CookieManager.getInstance().setAcceptThirdPartyCookies(this, false)
                webViewClient = object : WebViewClient() {
                    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                        if (!request.isForMainFrame) return false
                        val requestUri = request.url
                        val destination = Uri.parse(base)
                        if (requestUri.scheme == destination.scheme &&
                            requestUri.host.equals(destination.host, ignoreCase = true) &&
                            requestUri.port == destination.port) return false
                        if (requestUri.scheme == "https" || requestUri.scheme == "http") {
                            context.startActivity(Intent(Intent.ACTION_VIEW, requestUri))
                        }
                        return true
                    }
                }
                loadUrl(entry)
            }
        },
        update = {}
    )
}
