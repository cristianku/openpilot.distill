package ch.bodhi.distill

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Layers
import androidx.compose.material.icons.outlined.Replay
import androidx.compose.material.icons.outlined.History
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import org.json.JSONObject

private val SESSIONS = listOf("Trial" to 1000, "Standard" to 5000)

/** New training, as the wizard of the site: where to start, how much (or which run), then the summary. */
@Composable
fun TrainScreen(data: ServerData, onStarted: (String) -> Unit) {
    val status = data.status
    val runs = data.runs.orEmpty()
    val hf = data.hf
    var step by remember { mutableIntStateOf(1) }
    var mode by remember { mutableStateOf("fresh") }
    var steps by remember { mutableIntStateOf(5000) }
    var custom by remember { mutableStateOf(false) }
    var resumeRun by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var problem by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()

    val resumable = runs.filter { it.resumable }
    // No stable weights on Hugging Face: "Continue" cannot start, so it cannot be chosen either.
    val noStable = hf != null && hf.ok && hf.stable == false
    if (mode == "continue" && noStable || mode == "resume" && resumable.isEmpty()) mode = "fresh"
    val chosenRun = resumable.firstOrNull { it.name == resumeRun } ?: resumable.firstOrNull()
    val gpus = status?.gpus.orEmpty()
    // The choice the site makes by default: all of them when there are more, else the only one.
    val gpuValue = if (gpus.size > 1) "all" else gpus.firstOrNull()?.index?.toString()
    val gpuText = when {
        gpus.size > 1 -> "${gpus.size} × ${gpus.first().name}"
        gpus.isNotEmpty() -> gpus.first().let { "GPU ${it.index} · ${it.name}" + (it.memoryGb?.let { gb -> " · $gb GB" } ?: "") }
        else -> "No GPU found"
    }
    val batch = if (mode == "resume") chosenRun?.batchSize ?: 8 else 8
    val workers = if (mode == "resume") chosenRun?.workers ?: 6 else 6
    // Seconds per step on this server: the latest run with the same batch, else the latest one.
    val timed = runs.filter { it.avgStepTime != null && (it.stepsDone ?: 0) >= 20 }.sortedByDescending { it.started ?: 0.0 }
    val stepSeconds = (timed.firstOrNull { it.batchSize == batch } ?: timed.firstOrNull())?.avgStepTime

    when (step) {
        1 -> WizardStep(1, 3, "Where do we start?", "One choice, then we take care of the setup.", onBack = null) {
            ChoiceCard("New model", "Start a new training from scratch", Icons.Outlined.AutoAwesome,
                selected = mode == "fresh") { mode = "fresh" }
            ChoiceCard("Continue", "From the stable model on Hugging Face", Icons.Outlined.Layers,
                selected = mode == "continue", enabled = !noStable) { mode = "continue" }
            ChoiceCard("Resume", "From where a run stopped", Icons.Outlined.Replay,
                selected = mode == "resume", enabled = resumable.isNotEmpty()) { mode = "resume" }
            if (noStable) Note("There are no stable weights on Hugging Face yet: publish a model as “stable” first.")
            PrimaryAction("Continue", "") { step = 2 }
        }
        2 -> if (mode == "resume") {
            WizardStep(2, 3, "Which run?", "Pick the run that stopped.", onBack = { step = 1 }) {
                resumable.forEach { run ->
                    ChoiceCard("Run " + (shortDate(run.started).ifBlank { run.name }),
                        "${run.stepsDone ?: 0}" + (run.stepsTotal?.let { " / $it" } ?: "") + " steps",
                        Icons.Outlined.History, selected = run.name == chosenRun?.name) { resumeRun = run.name }
                }
                Note("It continues from its last checkpoint with the same teacher and number of steps.")
                PrimaryAction("Summary", if (chosenRun == null) "no run to resume" else "") { step = 3 }
            }
        } else {
            WizardStep(2, 3, "How much to train?", "Choose a session.", onBack = { step = 1 }) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    SESSIONS.forEach { (name, value) ->
                        FilterChip(selected = !custom && steps == value, onClick = { custom = false; steps = value },
                            label = { Text("$name · ${thousands(value)}") }, shape = RoundedCornerShape(12.dp))
                    }
                    FilterChip(selected = custom || SESSIONS.none { it.second == steps }, onClick = { custom = true },
                        label = { Text("Custom") }, shape = RoundedCornerShape(12.dp))
                }
                if (custom) {
                    OutlinedTextField(value = steps.toString(), onValueChange = { text ->
                        steps = text.filter(Char::isDigit).take(6).toIntOrNull() ?: 0
                    }, label = { Text("Steps") }, singleLine = true, shape = RoundedCornerShape(12.dp),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth())
                }
                Note("The model learns from the examples of your drives. " + if (stepSeconds != null && steps > 0)
                    "On this server: ${aboutDuration(stepSeconds * steps)}." else "More steps take more time.")
                PrimaryAction("Summary", if (steps !in 1..200_000) "the steps go from 1 to 200,000" else "") { step = 3 }
            }
        }
        else -> WizardStep(3, 3, "Ready to start",
            "The GPU stays busy for the whole session; you can follow it in Overview and stop it.", onBack = { step = 2 }) {
            val start = when (mode) {
                "continue" -> "Continue from the stable model"
                "resume" -> "Resume " + (chosenRun?.name ?: "–")
                else -> "New model (from scratch)"
            }
            val rows = mutableListOf("Start" to start)
            if (mode != "resume") {
                val name = SESSIONS.firstOrNull { it.second == steps && !custom }?.first ?: "Custom"
                rows += "Session" to "$name · ${thousands(steps)} steps" +
                    (stepSeconds?.let { " · " + aboutDuration(it * steps) } ?: "")
                rows += "Teacher" to "MD v3 (current big model)"
            }
            rows += "Hardware" to "$gpuText · batch $batch · $workers workers"
            rows += "Charts" to "automatic (online if configured)"
            Summary(rows)
            var reason = status?.reason("train") ?: "server status not read yet"
            if (reason.isEmpty() && gpuValue == null) reason = "no GPU found"
            if (reason.isEmpty() && mode == "resume" && chosenRun == null) reason = "no run to resume"
            PrimaryAction("Start training", reason, busy = busy) {
                val params = JSONObject().put("mode", mode).put("batch_size", batch).put("workers", workers)
                    .put("wandb", "").put("gpus", gpuValue)
                if (mode == "resume") params.put("run", chosenRun?.name) else params.put("steps", steps).put("teacher", "md-v3")
                busy = true
                problem = ""
                scope.launch {
                    try {
                        onStarted(data.start("train", params))
                        step = 1
                    } catch (exception: Exception) {
                        problem = exception.localizedMessage ?: "The training did not start."
                    }
                    busy = false
                }
            }
            if (problem.isNotEmpty()) Note(problem, Danger)
        }
    }
}
