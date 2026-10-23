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

/** Whether the server lets an action start now, and why not (status.actions). */
data class ActionState(val allowed: Boolean, val reason: String)
data class Gpu(val index: Int, val name: String, val memoryGb: Int?)
data class Job(val id: String, val label: String, val summary: String, val percent: Double?, val done: Int?,
               val total: Int?, val stoppable: Boolean)
data class CommaState(val reachable: Boolean?, val onroad: Boolean, val bundle: String, val bundleDefault: Boolean?,
                      val backup: Boolean?, val host: String, val error: String, val checked: Double?)

/** GET /api/status, the part the app shows. */
data class ServerStatus(
    val metrics: List<Metric>,
    val training: List<TrainingItem>,
    val jobs: List<Job>,
    val comma: CommaState,
    val actions: Map<String, ActionState>,
    val gpus: List<Gpu>,
    val code: String
) {
    /** '' when the server allows the action, otherwise its reason (as the buttons of the site). */
    fun reason(action: String): String =
        actions[action]?.let { if (it.allowed) "" else it.reason } ?: "server status not read yet"
}

/** A training run (GET /api/runs). */
data class Run(val name: String, val status: String, val stepsDone: Int?, val stepsTotal: Int?, val hasPt: Boolean,
               val hasTrainingPt: Boolean, val resumable: Boolean, val started: Double?, val teacher: String?,
               val loss: Double?, val avgStepTime: Double?, val batchSize: Int?, val workers: Int?)

/** A result of the hold-out evaluation (GET /api/eval), the comparable ones only. */
data class EvalResult(val model: String, val avgMissM: Double, val date: Double?)

/** GET /api/hf: whether stable weights exist and which models are published. */
data class HfState(val ok: Boolean, val stable: Boolean?, val models: List<String>, val error: String)

/** The site answered 401: this app has not signed in yet (or the session ended). */
class SignInNeeded : Exception("Sign in on the Website tab first.")

/** The server refused a request and said why (its JSON "message"). */
class ServerRefused(message: String) : Exception(message)

/** The native calls use the session of the WebView: the cookie the site set when you signed in there.
 * POSTs carry no Origin and no Sec-Fetch-Site header, which the server accepts as same-origin. */
object DistillApi {
    /** True when the address answers like a Distill site (GET /api/auth/state works without a session). */
    suspend fun probe(base: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val state = JSONObject(request(base, "GET", "/api/auth/state", null, null))
            state.has("configured") && state.has("signed_in")
        } catch (_: Exception) {
            false
        }
    }

    suspend fun status(base: String): ServerStatus = call(base, "GET", "/api/status") { parseStatus(JSONObject(it)) }

    suspend fun runs(base: String): List<Run> = call(base, "GET", "/api/runs") { raw ->
        val list = JSONObject(raw).optJSONArray("runs") ?: JSONArray()
        list.objects().map { parseRun(it) }
    }

    suspend fun evalResults(base: String): List<EvalResult> = call(base, "GET", "/api/eval") { raw ->
        val list = JSONObject(raw).optJSONArray("results") ?: JSONArray()
        // The default evaluation only (hold-out, 5 Hz), as the ranking of the site.
        list.objects().filter {
            val selection = it.optString("selection", "")
            (selection.isEmpty() || selection == "heldout") && (it.isNull("rate_hz") || it.optDouble("rate_hz") == 5.0)
        }.mapNotNull {
            val miss = it.number("avg_miss_m") ?: return@mapNotNull null
            EvalResult(it.optString("model"), miss, it.number("date"))
        }
    }

    suspend fun hf(base: String): HfState = call(base, "GET", "/api/hf") { raw ->
        val hf = JSONObject(raw)
        val models = (hf.optJSONArray("models") ?: JSONArray()).let { array ->
            (0 until array.length()).map { array.optString(it) }.filter { NAME_RE.matches(it) }
        }
        HfState(hf.optBoolean("ok"), if (hf.isNull("stable")) null else hf.optBoolean("stable"), models,
            hf.optString("error"))
    }

    /** POST /api/run/<action>: starts a job; returns its label. */
    suspend fun runAction(base: String, action: String, params: JSONObject): String =
        call(base, "POST", "/api/run/$action", params) { raw ->
            JSONObject(raw).optJSONObject("job")?.optString("label").orEmpty().ifBlank { action }
        }

    suspend fun stopJob(base: String, id: String) {
        call(base, "POST", "/api/jobs/$id/stop", JSONObject()) { }
    }

    /** POST /api/comma/check: asks the comma now. */
    suspend fun checkComma(base: String): CommaState = call(base, "POST", "/api/comma/check", JSONObject()) {
        parseComma(JSONObject(it).optJSONObject("comma"))
    }

    private val NAME_RE = Regex("^[a-z0-9_]+$")

    private suspend fun <T> call(base: String, method: String, path: String, body: JSONObject? = null,
                                 parse: (String) -> T): T {
        val cookie = CookieManager.getInstance().getCookie(base)
        return withContext(Dispatchers.IO) { parse(request(base, method, path, cookie, body)) }
    }

    private fun request(base: String, method: String, path: String, cookie: String?, body: JSONObject?): String {
        val connection = (URL(base + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 10_000
            readTimeout = 20_000
            instanceFollowRedirects = false
            useCaches = false
            setRequestProperty("Accept", "application/json")
            if (!cookie.isNullOrBlank()) setRequestProperty("Cookie", cookie)
        }
        try {
            if (body != null) {
                val bytes = body.toString().toByteArray(Charsets.UTF_8)
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json")
                connection.setFixedLengthStreamingMode(bytes.size)
                connection.outputStream.use { it.write(bytes) }
            }
            val status = connection.responseCode
            if (status == 200) return connection.inputStream.bufferedReader().use { it.readText() }
            if (status == 401) throw SignInNeeded()
            val message = try {
                JSONObject(connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "").optString("message")
            } catch (_: Exception) {
                ""
            }
            throw ServerRefused(message.ifBlank { "The server answered HTTP $status." })
        } finally {
            connection.disconnect()
        }
    }

    internal fun parseStatus(payload: JSONObject): ServerStatus {
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
        val training = (payload.optJSONArray("training") ?: JSONArray()).objects().map {
            TrainingItem(it.optString("name"), it.number("percent"), it.int("steps_done"), it.int("steps_total"))
        }
        val jobs = (payload.optJSONArray("jobs") ?: JSONArray()).objects()
            .filter { it.optString("state") == "running" }
            .map {
                val progress = it.optJSONObject("progress")
                Job(it.optString("id"), it.optString("label").ifBlank { it.optString("id") }, it.optString("summary"),
                    progress.number("percent"), progress?.int("done"), progress?.int("total"), it.optBoolean("stoppable"))
            }
        val actions = mutableMapOf<String, ActionState>()
        payload.optJSONObject("actions")?.let { states ->
            for (name in states.keys()) {
                val state = states.optJSONObject(name) ?: continue
                actions[name] = ActionState(state.optBoolean("allowed"), state.optString("reason"))
            }
        }
        val gpus = (system.optJSONArray("gpus") ?: JSONArray()).objects().map {
            Gpu(it.optInt("index"), it.optString("name").replace(Regex("-PCIE.*|-SXM.*"), ""),
                it.number("memory_total_mb")?.let { mb -> Math.round(mb / 1024).toInt() })
        }
        return ServerStatus(metrics, training, jobs, parseComma(payload.optJSONObject("comma")), actions, gpus,
            payload.optString("code", "unknown"))
    }

    private fun parseComma(comma: JSONObject?): CommaState = CommaState(
        reachable = comma?.let { if (it.isNull("reachable")) null else it.optBoolean("reachable") },
        onroad = comma?.optBoolean("onroad") == true,
        bundle = comma?.optString("bundle").orEmpty(),
        bundleDefault = comma?.let { if (it.isNull("bundle_default") || !it.has("bundle_default")) null else it.optBoolean("bundle_default") },
        backup = comma?.let { if (it.isNull("backup") || !it.has("backup")) null else it.optBoolean("backup") },
        host = comma?.optString("host").orEmpty(),
        error = comma?.optString("error").orEmpty(),
        checked = comma.number("checked")
    )

    internal fun parseRun(run: JSONObject): Run {
        val config = run.optJSONObject("config")
        val last = run.optJSONObject("last")
        return Run(run.optString("name"), run.optString("status"), run.int("steps_done"), run.int("steps_total"),
            run.optBoolean("has_pt"), run.optBoolean("has_training_pt"), run.optBoolean("resumable"),
            run.number("started"), config?.optString("teacher")?.takeIf { it.isNotBlank() }, last.number("loss"),
            run.number("avg_step_time"), config?.int("batch_size"), config?.int("workers"))
    }

    private fun JSONArray.objects(): List<JSONObject> = (0 until length()).mapNotNull { optJSONObject(it) }

    private fun JSONObject?.number(key: String): Double? {
        if (this == null || !has(key) || isNull(key)) return null
        return opt(key).let { if (it is Number) it.toDouble().takeIf { n -> n.isFinite() } else null }
    }

    private fun JSONObject.int(key: String): Int? = if (!has(key) || isNull(key)) null else opt(key).let {
        if (it is Number) it.toInt() else null
    }

    private fun percent(value: Double?): String = if (value == null) "—" else "${value.toInt()}%"
}
