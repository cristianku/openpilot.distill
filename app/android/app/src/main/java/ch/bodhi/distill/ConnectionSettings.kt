package ch.bodhi.distill

import android.content.Context
import java.net.URI
import java.security.KeyStore

/** The server this app talks to. Signing in happens on the site itself (its administrator page): the
 * session cookie lives in the WebView cookie store, never in these preferences. */
data class Connection(val url: String = "", val allowHttp: Boolean = false, val connected: Boolean = false) {
    val base: String? get() = ServerAddress.parse(url, allowHttp)
}

object ServerAddress {
    /** scheme://host[:port] of a plain server address; null for anything else (path, query, user info). */
    fun parse(input: String, allowHttp: Boolean): String? = try {
        val uri = URI(input.trim())
        val scheme = uri.scheme?.lowercase()
        if ((scheme != "https" && !(scheme == "http" && allowHttp)) ||
            uri.host.isNullOrBlank() || uri.rawUserInfo != null ||
            uri.rawQuery != null || uri.rawFragment != null ||
            (uri.rawPath != null && uri.rawPath != "" && uri.rawPath != "/") ||
            uri.port > 65535 || uri.port == 0 || uri.port < -1) {
            null
        } else {
            val host = if (uri.host.contains(":")) "[${uri.host}]" else uri.host
            "${scheme}://${host}${if (uri.port > 0) ":${uri.port}" else ""}"
        }
    } catch (_: Exception) {
        null
    }

    /** What people paste, the address of a page of the site (#status, /?token=… of an old link), reduced
     * to the server. User info is kept, so parse() still refuses it. */
    fun fromPasted(input: String): String = try {
        val uri = URI(input.trim())
        if (uri.scheme == null || uri.rawAuthority == null) input.trim() else "${uri.scheme}://${uri.rawAuthority}"
    } catch (_: Exception) {
        input.trim()
    }
}

class ConnectionSettings(context: Context) {
    private val prefs = context.getSharedPreferences("distill_connection", Context.MODE_PRIVATE)

    init {
        // The previews kept an access token; the site no longer uses one: forget it and its key.
        if (prefs.contains("token")) {
            prefs.edit().remove("token").putBoolean("connected", false).apply()
            try {
                KeyStore.getInstance("AndroidKeyStore").apply { load(null) }.deleteEntry("distill_server_token")
            } catch (_: Exception) {
            }
        }
    }

    fun load() = Connection(
        url = prefs.getString("url", "") ?: "",
        allowHttp = prefs.getBoolean("http", false),
        connected = prefs.getBoolean("connected", false)
    )

    fun save(value: Connection) {
        val normalized = value.base ?: return
        prefs.edit()
            .putString("url", normalized)
            .putBoolean("http", value.allowHttp)
            .putBoolean("connected", true)
            .apply()
    }

    fun disconnect() {
        prefs.edit().putBoolean("connected", false).apply()
    }
}
