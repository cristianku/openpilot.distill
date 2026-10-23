package ch.bodhi.distill

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.DirectionsCar
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.Route
import androidx.compose.material.icons.outlined.School
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import kotlinx.coroutines.launch
import org.json.JSONObject

/** A model the server can evaluate: spec as the site sends it (run:<name>, stock, teacher, logged, hf:<name>). */
data class ModelChoice(val spec: String, val title: String, val detail: String, val icon: ImageVector)

/** Your models (finished or with a checkpoint), the reference models, the ones published on Hugging Face. */
fun modelChoices(runs: List<Run>, hf: HfState?): List<Pair<String, List<ModelChoice>>> {
    val latest = runs.firstOrNull { it.hasPt }?.name
    val yours = runs.filter { it.hasPt || it.hasTrainingPt }.map { run ->
        val details = mutableListOf<String>()
        run.teacher?.let { details += "teacher " + if (it == "md-v3") "MD v3" else it }
        run.stepsTotal?.let { details += thousands(it) + " steps" }
        run.loss?.let { details += "loss " + "%.1f".format(java.util.Locale.ENGLISH, it) }
        if (!run.hasPt) details += "not finished"
        if (run.name == latest) details += "latest"
        ModelChoice("run:" + run.name, "Your model · " + shortDate(run.started).ifBlank { run.name },
            details.joinToString(" · "), Icons.Outlined.Memory)
    }
    val references = listOf(
        ModelChoice("stock", "comma model (stock)", "The model on your comma today: the one to beat.", Icons.Outlined.DirectionsCar),
        ModelChoice("teacher", "openpilot big model (teacher)", "MD v3, the large model your models learn from.", Icons.Outlined.School),
        ModelChoice("logged", "What the car actually did", "The path recorded in the drive logs; no model is run.", Icons.Outlined.Route)
    )
    val published = hf?.takeIf { it.ok }?.models.orEmpty().map {
        ModelChoice("hf:$it", "Published · $it", "Downloaded from Hugging Face", Icons.Outlined.CloudDownload)
    }
    return listOf("Your models" to yours, "Reference models" to references, "Published on Hugging Face" to published)
        .filter { it.second.isNotEmpty() }
}

/** Evaluate a model, as the wizard of the site: which model, then the summary. */
@Composable
fun EvaluateScreen(data: ServerData, onStarted: (String) -> Unit) {
    var step by remember { mutableIntStateOf(1) }
    var chosen by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var problem by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val groups = modelChoices(data.runs.orEmpty(), data.hf)
    val all = groups.flatMap { it.second }
    val model = all.firstOrNull { it.spec == chosen } ?: all.firstOrNull()

    if (step == 1) {
        WizardStep(1, 2, "Which model?", "It is tested on the drives it never saw in training.", onBack = null) {
            groups.forEach { (group, choices) ->
                Text(group.uppercase(), style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                choices.forEach { choice ->
                    ChoiceCard(choice.title, choice.detail, choice.icon, selected = choice.spec == model?.spec,
                        trailing = data.evalOf(choice.spec)?.let { "%.2f m".format(java.util.Locale.ENGLISH, it.avgMissM) }
                    ) { chosen = choice.spec }
                }
            }
            PrimaryAction("Continue", if (model == null) "no model to evaluate" else "") { step = 2 }
        }
    } else {
        WizardStep(2, 2, "Ready to evaluate",
            "How many metres its path misses the one you really drove, over the next 10 s.", onBack = { step = 1 }) {
            val rows = mutableListOf("Model" to (model?.title ?: "–"))
            model?.let { data.evalOf(it.spec) }?.let {
                rows += "Last time" to "%.2f m average miss".format(java.util.Locale.ENGLISH, it.avgMissM) +
                    (shortDate(it.date).takeIf { date -> date.isNotBlank() }?.let { date -> " · $date" } ?: "")
            }
            rows += "Hardware" to "the GPU, for a few minutes"
            Summary(rows)
            val reason = data.status?.reason("eval") ?: "server status not read yet"
            PrimaryAction("Evaluate", if (model == null) "no model to evaluate" else reason, busy = busy) {
                val spec = model?.spec ?: return@PrimaryAction
                busy = true
                problem = ""
                scope.launch {
                    try {
                        onStarted(data.start("eval", JSONObject().put("model", spec)))
                    } catch (exception: Exception) {
                        problem = exception.localizedMessage ?: "The evaluation did not start."
                    }
                    busy = false
                }
            }
            if (problem.isNotEmpty()) Note(problem, Danger)
        }
    }
}
