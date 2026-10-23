package ch.bodhi.distill

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CancellationException
import org.json.JSONObject

/** What the native screens show of the server, read with the session of the WebView. Every field is
 * Compose state: a screen redraws when a refresh brings something new. */
class ServerData(private val base: String) {
    var status by mutableStateOf<ServerStatus?>(null)
        private set
    var runs by mutableStateOf<List<Run>?>(null)
        private set
    var evalResults by mutableStateOf<List<EvalResult>>(emptyList())
        private set
    var hf by mutableStateOf<HfState?>(null)
        private set
    var signInNeeded by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    suspend fun refreshStatus() = guard { status = DistillApi.status(base) }

    /** The runs, the evaluations and Hugging Face: what Train, Evaluate and Comma choose from. */
    suspend fun refreshLists() = guard {
        runs = DistillApi.runs(base).sortedByDescending { it.started ?: 0.0 }
        evalResults = DistillApi.evalResults(base)
        hf = try {
            DistillApi.hf(base)
        } catch (exception: ServerRefused) {
            HfState(false, null, emptyList(), exception.message ?: "")
        }
    }

    /** Starts an action; returns the label of the job. Throws with the reason of the server. */
    suspend fun start(action: String, params: JSONObject): String {
        val label = DistillApi.runAction(base, action, params)
        refreshStatus()
        return label
    }

    suspend fun stop(job: Job) {
        DistillApi.stopJob(base, job.id)
        refreshStatus()
    }

    suspend fun checkComma() = guard {
        DistillApi.checkComma(base)
        status = DistillApi.status(base)
    }

    fun evalOf(spec: String): EvalResult? = evalResults.firstOrNull { it.model == spec }

    private suspend fun guard(block: suspend () -> Unit) {
        try {
            block()
            signInNeeded = false
            error = null
        } catch (_: SignInNeeded) {
            signInNeeded = true
        } catch (exception: CancellationException) {
            throw exception
        } catch (exception: Exception) {
            error = exception.localizedMessage ?: "Server unavailable"
        }
    }
}
