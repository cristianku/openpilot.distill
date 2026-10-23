package ch.bodhi.distill

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.json.JSONObject

private val Ok = Color(0xFF4ADE80)
private val Todo = Color(0xFFFBBF24)
private val Bad = Color(0xFFF87171)

/** [kind, title, detail] rows from the comma state, the checklist of Publish › On your comma. */
private fun commaChecks(comma: CommaState): List<Triple<Color, String, String>> {
    if (comma.reachable == null) return listOf(Triple(Color.Gray, "Not checked yet", "Tap Check now to ask the comma."))
    if (!comma.reachable) {
        return listOf(Triple(Bad, "Not reachable", (if (comma.error.isNotBlank()) capitalized(comma.error) + ". " else "") +
            "Its address (${comma.host}) is set in the Settings of the site."))
    }
    val checks = mutableListOf(Triple(Ok, "Reachable", "Address: ${comma.host}"))
    checks += if (comma.onroad) Triple(Todo, "Driving", "Installing and restoring wait until the car is off (offroad).")
              else Triple(Ok, "Parked", "The car is off (offroad).")
    when (comma.bundleDefault) {
        false -> checks += Triple(Todo, "Models menu: ${comma.bundle}", "The comma drives with the model chosen there: " +
            "set it back to Default on the comma screen (Settings › Models) to install.")
        true -> checks += Triple(Ok, "Models menu on Default", "The comma drives with openpilot's model, or the one you installed.")
        null -> {}
    }
    when (comma.backup) {
        true -> checks += Triple(Ok, "Original model saved", "It can be put back at any time.")
        false -> checks += Triple(Color.Gray, "Original model not saved yet", "Installing a model saves it first.")
        null -> {}
    }
    return checks
}

/** Install on the comma, as Publish › On your comma: which finished model, then the comma and Install. */
@Composable
fun CommaScreen(data: ServerData, onStarted: (String) -> Unit) {
    var step by remember { mutableIntStateOf(1) }
    var chosen by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var checking by remember { mutableStateOf(false) }
    var confirm by remember { mutableStateOf<String?>(null) }
    var problem by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val finished = data.runs.orEmpty().filter { it.hasPt }
    val run = finished.firstOrNull { it.name == chosen } ?: finished.firstOrNull()

    fun start(action: String, params: JSONObject) {
        busy = true
        problem = ""
        scope.launch {
            try {
                onStarted(data.start(action, params))
            } catch (exception: Exception) {
                problem = exception.localizedMessage ?: "It did not start."
            }
            busy = false
        }
    }

    if (step == 1) {
        WizardStep(1, 2, "Which model?", "A finished training, ready to drive.", onBack = null) {
            if (finished.isEmpty()) Note("No finished training yet: start one in Train.")
            finished.forEach { item ->
                ChoiceCard("Your model · " + shortDate(item.started).ifBlank { item.name },
                    listOfNotNull(item.stepsTotal?.let { thousands(it) + " steps" },
                        item.loss?.let { "loss " + "%.1f".format(java.util.Locale.ENGLISH, it) }).joinToString(" · "),
                    Icons.Outlined.Memory, selected = item.name == run?.name,
                    trailing = data.evalOf("run:" + item.name)?.let { "%.2f m".format(java.util.Locale.ENGLISH, it.avgMissM) }
                ) { chosen = item.name }
            }
            PrimaryAction("Continue", if (run == null) "no finished model" else "") { step = 2 }
            OutlinedButton(onClick = { chosen = null; step = 2 }, modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp)) { Text("Put the original model back on the comma") }
        }
        return
    }

    val comma = data.status?.comma
    WizardStep(2, 2, "On your comma",
        "It takes the place of openpilot's driving model and the comma restarts; the first start takes a few extra minutes.",
        onBack = { step = 1 }) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Your comma", fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            OutlinedButton(onClick = {
                checking = true
                scope.launch { data.checkComma(); checking = false }
            }, enabled = !checking, shape = RoundedCornerShape(12.dp)) { Text(if (checking) "Checking…" else "Check now") }
        }
        Card(shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(vertical = 6.dp)) {
                (comma?.let { commaChecks(it) } ?: listOf(Triple(Color.Gray, "Reading the state of the comma…", ""))).forEach {
                    (color, title, detail) ->
                    Row(Modifier.padding(horizontal = 14.dp, vertical = 9.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Box(Modifier.padding(top = 4.dp).size(12.dp).clip(CircleShape).background(color))
                        Column(Modifier.weight(1f)) {
                            Text(title, fontWeight = FontWeight.SemiBold)
                            if (detail.isNotBlank()) Text(detail, fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
        val installReason = when {
            run == null -> "choose a finished model in step 1"
            else -> data.status?.reason("install") ?: "server status not read yet"
        }
        PrimaryAction("Install " + (run?.let { shortDate(it.started).ifBlank { it.name } } ?: "on the comma"),
            installReason, busy = busy, danger = true) { confirm = "install" }
        Text("Put the original model back", fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Note("openpilot's own driving model comes back and the comma restarts. A software update from the comma screen does the same.")
        val restoreReason = data.status?.reason("restore") ?: "server status not read yet"
        OutlinedButton(onClick = { confirm = "restore" }, enabled = restoreReason.isEmpty() && !busy,
            modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp)) {
            Text("Put the original model back", color = Danger)
        }
        if (restoreReason.isNotEmpty()) Note(capitalized(restoreReason) + ".")
        if (problem.isNotEmpty()) Note(problem, Danger)
    }

    when (confirm) {
        "install" -> Confirm("Install on the comma?",
            "Install the model ${run?.name} on the comma? It takes the place of openpilot's driving model and the comma restarts.",
            "Install", onConfirm = {
                confirm = null
                run?.let { start("install", JSONObject().put("run", it.name).put("confirm", "INSTALL")) }
            }, onDismiss = { confirm = null })
        "restore" -> Confirm("Put the original back?",
            "Put openpilot's original driving model back on the comma? The comma restarts.",
            "Put it back", onConfirm = {
                confirm = null
                start("restore", JSONObject().put("confirm", "RESTORE"))
            }, onDismiss = { confirm = null })
    }
}
