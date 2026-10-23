package ch.bodhi.distill

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// The language of the wizards of the site (web/static/app.css): one card, one question per step, big
// choice cards with a check mark, one big primary button, a back arrow, dots and "Step N of M".
val Danger = Color(0xFFDC2626)
private val AccentSoft = Color(0x293B82F6)

/** A wizard step: back arrow and dots on top, the big title, then the content, then "Step N of M". */
@Composable
fun WizardStep(step: Int, count: Int, title: String, subtitle: String, onBack: (() -> Unit)?,
               content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Box(Modifier.fillMaxWidth().height(40.dp)) {
            if (onBack != null) {
                IconButton(onClick = onBack, modifier = Modifier.align(Alignment.CenterStart)) {
                    Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = "Back")
                }
            }
            if (count > 1) {
                Row(Modifier.align(Alignment.Center), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    for (index in 1..count) {
                        Box(Modifier.size(7.dp).clip(CircleShape).background(
                            if (index == step) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant))
                    }
                }
            }
        }
        Text(title, fontSize = 28.sp, lineHeight = 33.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth())
        if (subtitle.isNotBlank()) {
            Text(subtitle, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
        }
        content()
        if (count > 1) {
            Text("Step $step of $count", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
        }
        Spacer(Modifier.height(8.dp))
    }
}

/** A big choice card: an icon, the title, one line under it, the check mark when chosen. */
@Composable
fun ChoiceCard(title: String, detail: String, icon: ImageVector, selected: Boolean, enabled: Boolean = true,
               trailing: String? = null, onClick: () -> Unit) {
    val border = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
    Card(
        shape = RoundedCornerShape(18.dp),
        border = BorderStroke(if (selected) 2.dp else 1.dp, border),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier.fillMaxWidth().clickable(enabled = enabled, onClick = onClick)
    ) {
        Row(
            Modifier.background(Brush.radialGradient(listOf(AccentSoft, Color.Transparent))).padding(16.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Box(Modifier.size(52.dp).clip(RoundedCornerShape(14.dp)).background(AccentSoft),
                contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = null, tint = if (enabled) Color(0xFF93C5FD) else MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(28.dp))
            }
            Column(Modifier.weight(1f)) {
                Text(title, fontSize = 18.sp, fontWeight = FontWeight.Bold,
                    color = if (enabled) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant)
                if (detail.isNotBlank()) {
                    Text(detail, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (trailing != null) {
                Text(trailing, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Box(Modifier.size(28.dp).clip(CircleShape)
                .background(if (selected) MaterialTheme.colorScheme.primary else Color.Transparent),
                contentAlignment = Alignment.Center) {
                if (selected) Icon(Icons.Outlined.Check, contentDescription = "Chosen", tint = Color.White,
                    modifier = Modifier.size(18.dp))
            }
        }
    }
}

/** The one big button of a step, and under it why it cannot start (the reason of the server). */
@Composable
fun PrimaryAction(label: String, reason: String, busy: Boolean = false, danger: Boolean = false, onClick: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Button(
            onClick = onClick, enabled = reason.isEmpty() && !busy, shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = if (danger) Danger else MaterialTheme.colorScheme.primary),
            modifier = Modifier.fillMaxWidth().height(52.dp)
        ) {
            if (busy) CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp, color = Color.White)
            else Text(label, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }
        if (reason.isNotEmpty()) {
            Text(capitalized(reason) + ".", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
        }
    }
}

/** label: value lines in a rounded box, as the summary of the training wizard. */
@Composable
fun Summary(rows: List<Pair<String, String>>) {
    Card(shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            rows.forEach { (label, value) ->
                Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.width(96.dp))
                    Text(value, modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

/** A grey line of help under a group of choices. */
@Composable
fun Note(text: String, color: Color = MaterialTheme.colorScheme.onSurfaceVariant) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = color, textAlign = TextAlign.Center,
        modifier = Modifier.fillMaxWidth())
}

/** The card that replaces a screen until the app is signed in on the site. */
@Composable
fun SignInCard(onSignIn: () -> Unit) {
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically)) {
        Text("Sign in first", fontSize = 28.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth())
        Note("Sign in on the Website tab with the administrator of the site; this screen then works with your server.")
        PrimaryAction("Sign in", "", onClick = onSignIn)
    }
}

/** A confirmation dialog for what restarts the comma or uses the GPU for a long time. */
@Composable
fun Confirm(title: String, text: String, confirm: String, onConfirm: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(text) },
        confirmButton = { TextButton(onClick = onConfirm) { Text(confirm, color = Danger) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
        containerColor = MaterialTheme.colorScheme.surface
    )
}

fun capitalized(text: String): String = text.replaceFirstChar { it.uppercase() }

/** 23 Oct, 09:29 */
fun shortDate(unix: Double?): String {
    if (unix == null) return ""
    val format = java.text.SimpleDateFormat("d MMM, HH:mm", java.util.Locale.ENGLISH)
    return format.format(java.util.Date((unix * 1000).toLong()))
}

/** about 2 h 35 min, as the estimate of the site. */
fun aboutDuration(seconds: Double): String {
    if (seconds < 90) return "about a minute"
    val minutes = Math.round(seconds / 60).toInt()
    if (minutes < 60) return "about $minutes min"
    val rest = Math.round((minutes % 60) / 5.0).toInt() * 5
    return "about ${minutes / 60} h" + if (rest > 0) " $rest min" else ""
}

/** 5,000 */
fun thousands(value: Int): String = String.format(java.util.Locale.ENGLISH, "%,d", value)
