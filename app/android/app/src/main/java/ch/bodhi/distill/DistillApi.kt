package ch.bodhi.distill

import android.webkit.CookieManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class Metric(val title: String, val value: String, val detail: String)
data class TrainingItem(val name: String, val percent: Double?, val done: Int?, val total: Int?)
data class JobItem(val label: String, val summary: String?)
data class Dashboard(
    val metrics: List<Metric>,
    val training: List<TrainingItem>,
    val jobs: List<JobItem>,
    val comma: String,
    val code: String
)

/** The site answered 401: this app has not signed in yet (or the session ended). */
class SignInNeeded : Exception("Sign in on the Website tab first.")

/** The native calls use the session of the WebView: the cookie the site set when you signed in there. */
object DistillApi {
    /** True when the address answers like a Distill site (GET /api/auth/state works without a session). */
    suspend fun probe(base: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val state = JSONObject(get(base, "/api/auth/state", null))
            state.has("configured") && state.has("signed_in")
        } catch (_: Exception) {
            false
        }
    }

    suspend fun status(config: Connection): Dashboard {
        val base = config.base ?: error("Invalid server URL")
        val cookie = CookieManager.getInstance().getCookie(base)
        return withContext(Dispatchers.IO) { parse(JSONObject(get(base, "/api/status", cookie))) }
    }

    private fun get(base: String, path: String, cookie: String?): String {
        val connection = (URL(base + path).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 10_000
            readTimeout = 12_000
            instanceFollowRedirects = false
            useCaches = false
            setRequestProperty("Accept", "application/json")
            if (!cookie.isNullOrBlank()) setRequestProperty("Cookie", cookie)
        }
        try {
            when (val status = connection.responseCode) {
                200 -> return connection.inputStream.bufferedReader().use { it.readText() }
                401 -> throw SignInNeeded()
                else -> throw IllegalStateException("The server answered HTTP $status.")
            }
        } finally {
            connection.disconnect()
        }
    }

    internal fun parse(payload: JSONObject): Dashboard {
        val system = payload.optJSONObject("system") ?: JSONObject()
        val cpu = system.optJSONObject("cpu")
        val gpu = system.optJSONObject("gpu")
        val ram = system.optJSONObject("memory")
        val disk = system.optJSONObject("disk")
        val free = disk.number("free_gb")
        val total = disk.number("total_gb")
        val metrics = listOf(
            Metric("GPU", percent(gpu.number("utilization")), gpu?.optString("name").orEmpty().ifBlank { "Not available" }),
            Metric("CPU", percent(cpu.number("total")), "Processor"),
            Metric("Memory", percent(ram.number("used_percent")), "RAM"),
            Metric("Disk", if (free != null && total != null) "%.0f / %.0f GB".format(free, total) else "—", "Free / total")
        )
        val trainingJson = payload.optJSONArray("training") ?: JSONArray()
        val training = (0 until trainingJson.length()).mapNotNull { index ->
            trainingJson.optJSONObject(index)?.let {
                TrainingItem(it.optString("name"), it.number("percent"), it.int("steps_done"), it.int("steps_total"))
            }
        }
        val jobsJson = payload.optJSONArray("jobs") ?: JSONArray()
        val jobs = (0 until jobsJson.length()).mapNotNull { index ->
            jobsJson.optJSONObject(index)?.takeIf { it.optString("state") == "running" }?.let {
                JobItem(it.optString("label").ifBlank { it.optString("id") }, it.optString("summary").takeIf { text -> text.isNotBlank() })
            }
        }
        val commaJson = payload.optJSONObject("comma")
        val comma = when {
            commaJson?.optBoolean("reachable") != true -> "Offline"
            commaJson.optBoolean("onroad") -> "On road"
            else -> "Reachable"
        }
        return Dashboard(metrics, training, jobs, comma, payload.optString("code", "unknown"))
    }

    private fun JSONObject?.number(key: String): Double? {
        if (this == null || isNull(key)) return null
        return opt(key).let { if (it is Number) it.toDouble().takeIf { n -> n.isFinite() } else null }
    }

    private fun JSONObject.int(key: String): Int? = if (isNull(key)) null else opt(key).let {
        if (it is Number) it.toInt() else null
    }

    private fun percent(value: Double?): String = if (value == null) "—" else "${value.toInt()}%"
}
