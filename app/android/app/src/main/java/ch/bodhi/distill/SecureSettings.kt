package ch.bodhi.distill

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.net.URI
import java.net.URLDecoder
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

data class Connection(val url: String = "", val token: String = "", val allowHttp: Boolean = false, val connected: Boolean = false) {
    val base: String? get() = ServerAddress.parse(url, allowHttp)
}

object ServerAddress {
    // A pasted browser login URL is split before it can enter normal preferences.
    fun extractLoginLink(input: String): Pair<String, String>? = try {
        val uri = URI(input.trim())
        val query = uri.rawQuery?.split("&") ?: return null
        if (query.size != 1) return null
        val fields = query.single().split("=", limit = 2)
        if (fields.size != 2 || URLDecoder.decode(fields[0], "UTF-8") != "token") return null
        val token = URLDecoder.decode(fields[1], "UTF-8")
        if (token.isEmpty()) return null
        val clean = URI(uri.scheme, uri.rawAuthority, uri.path, null, null)
        val base = parse(clean.toString(), allowHttp = true) ?: return null
        base to token
    } catch (_: Exception) {
        null
    }

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
}

class SecureSettings(context: Context) {
    private val prefs = context.getSharedPreferences("distill_connection", Context.MODE_PRIVATE)
    private val alias = "distill_server_token"

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(alias, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    private fun encrypt(token: String): String {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val data = cipher.iv + cipher.doFinal(token.toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(data, Base64.NO_WRAP)
    }

    private fun decrypt(value: String?): String {
        if (value == null) return ""
        return try {
            val data = Base64.decode(value, Base64.NO_WRAP)
            if (data.size <= 12) return ""
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(128, data.copyOfRange(0, 12)))
            String(cipher.doFinal(data.copyOfRange(12, data.size)), Charsets.UTF_8)
        } catch (_: Exception) {
            ""
        }
    }

    fun load() = Connection(
        url = prefs.getString("url", "") ?: "",
        token = decrypt(prefs.getString("token", null)),
        allowHttp = prefs.getBoolean("http", false),
        connected = prefs.getBoolean("connected", false)
    )

    fun save(value: Connection) {
        val normalized = value.base ?: return
        if (value.token.isBlank()) return
        prefs.edit()
            .putString("url", normalized)
            .putString("token", encrypt(value.token.trim()))
            .putBoolean("http", value.allowHttp)
            .putBoolean("connected", true)
            .apply()
    }

    fun disconnect() {
        prefs.edit().remove("token").putBoolean("connected", false).apply()
    }
}
