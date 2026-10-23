package ch.bodhi.distill

import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** A Distill server inside the app, to try it without one and for the Google Play review: server address
 * "demo", name "demo", password "demo". It answers the same API as distill-web (web/API.md) with made-up
 * data, and the jobs it starts only move a progress bar: nothing leaves the phone. */
object DemoServer {
    const val BASE = "https://demo.distill.invalid"
    const val ADDRESS = "demo"
    const val USER = "demo"
    const val PASSWORD = "demo"
    private const val SESSION = "distill_demo=signed-in"

    /** How long each demo job runs, in seconds. */
    private val DURATION = mapOf("train" to 180L, "eval" to 60L, "install" to 40L, "restore" to 30L)
    private val LABEL = mapOf("train" to "Training", "eval" to "Evaluation", "install" to "Install on the comma",
        "restore" to "Restore the original model")
    private val GPU_ACTIONS = setOf("train", "eval")

    private class DemoJob(val id: String, val action: String, val summary: String, val total: Int?,
                          val started: Long, val params: JSONObject)

    private class DemoRun(val name: String, val started: Double, val total: Int, val done: Int, val finished: Boolean,
                          val loss: Double)

    /** The clock of the jobs, in milliseconds (the tests move it). */
    internal var clock: () -> Long = System::currentTimeMillis
    private var jobs = mutableListOf<DemoJob>()
    private var runs = mutableListOf<DemoRun>()
    private var results = mutableListOf<Pair<String, Double>>()
    private var published = ""
    private var backup = false
    private var counter = 0

    init {
        reset()
    }

    /** Back to the first state: three runs, three evaluations, nothing running. */
    @Synchronized
    internal fun reset() {
        val now = clock() / 1000.0
        jobs = mutableListOf()
        runs = mutableListOf(
            DemoRun(runName(now - 26 * 3600), now - 26 * 3600, 5000, 2300, false, 41.2),
            DemoRun(runName(now - 50 * 3600), now - 50 * 3600, 5000, 5000, true, 38.4),
            DemoRun(runName(now - 122 * 3600), now - 122 * 3600, 1000, 1000, true, 52.0)
        )
        results = mutableListOf("stock" to 1.84, "teacher" to 1.52, "run:" + runs[1].name to 1.67)
        published = runs[1].name.replace("-", "_")
        backup = false
    }

    @Synchronized
    fun answer(method: String, path: String, cookie: String?, body: JSONObject?): DistillApi.Reply {
        val signedIn = cookie?.split(";")?.any { it.trim() == SESSION } == true
        if (method == "GET" && path == "/api/auth/state") {
            return reply(JSONObject().put("configured", true).put("signed_in", signedIn).put("api", DistillApi.API_VERSION))
        }
        if (method == "POST" && path == "/api/auth/login") {
            if (body?.optString("user") != USER || body.optString("password") != PASSWORD) {
                throw SignInNeeded("Wrong name or password. In the demo the name is $USER and the password $PASSWORD.")
            }
            return DistillApi.Reply("{\"ok\":true}", listOf("$SESSION; Path=/; HttpOnly; SameSite=Strict"))
        }
        if (!signedIn) throw SignInNeeded()
        advance()
        val run = Regex("/api/run/([a-z_]+)").matchEntire(path)
        val stop = Regex("/api/jobs/([^/]+)/stop").matchEntire(path)
        return when {
            method == "GET" && path == "/api/status" -> reply(status())
            method == "GET" && path == "/api/runs" -> reply(JSONObject().put("runs", JSONArray(runs.map { runJson(it) })))
            method == "GET" && path == "/api/eval" -> reply(JSONObject().put("results", JSONArray(results.map { (model, miss) ->
                JSONObject().put("model", model).put("avg_miss_m", miss).put("selection", "heldout").put("rate_hz", 5.0)
                    .put("date", clock() / 1000.0 - 20 * 3600)
            })))
            method == "GET" && path == "/api/hf" -> reply(JSONObject().put("ok", true).put("stable", true)
                .put("models", JSONArray(listOf(published))).put("error", ""))
            method == "POST" && path == "/api/auth/logout" ->
                DistillApi.Reply("{\"ok\":true}", listOf("distill_demo=; Path=/; Max-Age=0"))
            method == "POST" && path == "/api/comma/check" -> reply(JSONObject().put("ok", true).put("comma", comma()))
            method == "POST" && run != null -> reply(JSONObject().put("ok", true).put("job", start(run.groupValues[1], body ?: JSONObject())))
            method == "POST" && stop != null -> {
                val job = jobs.firstOrNull { it.id == stop.groupValues[1] } ?: throw ServerRefused("Unknown job.")
                jobs.remove(job)
                reply(JSONObject().put("ok", true).put("job", jobJson(job)))
            }
            else -> throw ServerRefused("Page not found.")
        }
    }

    /** '' when the action can start now, otherwise why not (as the actions of /api/status). */
    private fun reason(action: String): String {
        val busy = jobs.firstOrNull { (it.action in GPU_ACTIONS) == (action in GPU_ACTIONS) } ?: return ""
        return if (action in GPU_ACTIONS) "the GPU is busy: ${LABEL[busy.action]} is running"
               else "the comma is busy: ${LABEL[busy.action]} is running"
    }

    private fun start(action: String, params: JSONObject): JSONObject {
        if (action !in DURATION) throw ServerRefused("Unknown action.")
        reason(action).takeIf { it.isNotEmpty() }?.let { throw ServerRefused(capitalized(it) + ".") }
        val total: Int?
        val summary: String
        when (action) {
            "train" -> {
                val resumed = runs.firstOrNull { it.name == params.optString("run") }
                total = resumed?.total ?: params.optInt("steps", 5000).coerceIn(1, 200_000)
                summary = when (params.optString("mode")) {
                    "resume" -> "resume ${resumed?.name ?: "a run"}"
                    "continue" -> "${thousands(total)} steps from the stable model"
                    else -> "${thousands(total)} steps from scratch"
                }
            }
            "eval" -> { total = 40; summary = "hold-out segments, " + params.optString("model") }
            "install" -> { total = null; summary = "model " + params.optString("run") }
            else -> { total = null; summary = "openpilot's original model" }
        }
        counter += 1
        val id = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ENGLISH).format(Date(clock())) + "-$action-$counter"
        val job = DemoJob(id, action, summary, total, clock(), params)
        jobs.add(job)
        return jobJson(job)
    }

    /** The jobs that ran their time are done: a training leaves a finished run, an evaluation its result. */
    private fun advance() {
        val done = jobs.filter { fraction(it) >= 1.0 }
        for (job in done) {
            jobs.remove(job)
            when (job.action) {
                "train" -> {
                    val name = job.params.optString("run")
                    runs.removeAll { it.name == name }
                    val started = job.started / 1000.0
                    runs.add(0, DemoRun(name.ifBlank { runName(started) }, started, job.total ?: 5000, job.total ?: 5000,
                        true, 36.9))
                }
                "eval" -> {
                    val model = job.params.optString("model")
                    results.removeAll { it.first == model }
                    results.add(model to 1.58 + (model.hashCode().mod(17)) / 100.0)
                }
                "install" -> backup = true
            }
        }
    }

    private fun fraction(job: DemoJob): Double =
        ((clock() - job.started) / 1000.0 / (DURATION[job.action] ?: 60L)).coerceIn(0.0, 1.0)

    private fun jobJson(job: DemoJob): JSONObject {
        val fraction = fraction(job)
        val progress = JSONObject().put("percent", Math.round(fraction * 1000) / 10.0)
        job.total?.let { progress.put("done", (it * fraction).toInt()).put("total", it) }
        return JSONObject().put("id", job.id).put("action", job.action).put("label", LABEL[job.action])
            .put("summary", job.summary).put("state", "running").put("stoppable", true).put("progress", progress)
    }

    private fun status(): JSONObject {
        val training = jobs.firstOrNull { it.action == "train" }
        val gpuBusy = jobs.any { it.action in GPU_ACTIONS }
        val system = JSONObject()
            .put("cpu", JSONObject().put("total", if (gpuBusy) 38 else 4))
            .put("gpu", JSONObject().put("utilization", if (gpuBusy) 97 else 0).put("name", "Tesla V100-PCIE-32GB"))
            .put("memory", JSONObject().put("used_percent", if (gpuBusy) 46 else 21))
            .put("disk", JSONObject().put("free_gb", 812).put("total_gb", 1800))
            .put("gpus", JSONArray().put(JSONObject().put("index", 0).put("name", "Tesla V100-PCIE-32GB")
                .put("memory_total_mb", 32768)))
        val actions = JSONObject()
        for (action in DURATION.keys) {
            val reason = reason(action)
            actions.put(action, JSONObject().put("allowed", reason.isEmpty()).put("reason", reason)
                .put("label", LABEL[action]))
        }
        val trainingList = JSONArray()
        training?.let { job ->
            val total = job.total ?: 5000
            val fraction = fraction(job)
            trainingList.put(JSONObject().put("name", job.params.optString("run").ifBlank { runName(job.started / 1000.0) })
                .put("percent", Math.round(fraction * 1000) / 10.0).put("steps_done", (total * fraction).toInt())
                .put("steps_total", total))
        }
        return JSONObject().put("system", system).put("training", trainingList)
            .put("jobs", JSONArray(jobs.map { jobJson(it) })).put("comma", comma()).put("actions", actions)
            .put("code", "demo")
    }

    private fun comma(): JSONObject = JSONObject().put("reachable", true).put("onroad", false).put("bundle", "Default")
        .put("bundle_default", true).put("backup", backup).put("host", "comma (demo)").put("error", "")
        .put("checked", clock() / 1000.0)

    private fun runJson(run: DemoRun): JSONObject = JSONObject().put("name", run.name)
        .put("status", if (run.finished) "finished" else "interrupted").put("steps_done", run.done)
        .put("steps_total", run.total).put("has_pt", run.finished).put("has_training_pt", !run.finished)
        .put("resumable", !run.finished).put("started", run.started).put("avg_step_time", 0.71)
        .put("config", JSONObject().put("teacher", "md-v3").put("batch_size", 8).put("workers", 6))
        .put("last", JSONObject().put("loss", run.loss))

    private fun runName(unix: Double): String =
        SimpleDateFormat("yyyy-MM-dd_HHmm", Locale.ENGLISH).format(Date((unix * 1000).toLong()))

    private fun reply(json: JSONObject) = DistillApi.Reply(json.toString())
}
