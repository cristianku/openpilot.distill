package ch.bodhi.distill

import android.os.Bundle
import android.webkit.CookieManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Dashboard
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import androidx.compose.ui.graphics.Color

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val storage = SecureSettings(applicationContext)
        setContent {
            MaterialTheme(colorScheme = lightColorScheme(primary = Color(0xFF3F4DA3),
                background = Color(0xFFF5F5F7), surface = Color.White)) {
                DistillScreen(storage)
            }
        }
    }
}

@Composable
private fun DistillScreen(storage: SecureSettings) {
    var connection by remember { mutableStateOf(storage.load()) }
    var tab by remember { mutableStateOf(0) }
    if (!connection.connected || connection.base == null || connection.token.isBlank()) {
        ConnectionForm(
            initial = connection,
            isEditing = false,
            onSave = {
                storage.save(it)
                connection = storage.load()
                tab = 0
            },
            onDisconnect = {}
        )
        return
    }
    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf(Triple("Overview", Icons.Outlined.Dashboard, 0),
                       Triple("Website", Icons.Outlined.Language, 1),
                       Triple("Settings", Icons.Outlined.Settings, 2)).forEach { (label, icon, index) ->
                    NavigationBarItem(
                        selected = tab == index,
                        onClick = { tab = index },
                        icon = { androidx.compose.material3.Icon(icon, contentDescription = label) },
                        label = { Text(label) }
                    )
                }
            }
        }
    ) { padding ->
        when (tab) {
            0 -> DashboardScreen(connection, Modifier.padding(padding))
            1 -> Portal(connection, Modifier.fillMaxSize().padding(padding))
            else -> Box(Modifier.padding(padding)) {
                ConnectionForm(
                    initial = connection, isEditing = true,
                    onSave = { storage.save(it); connection = storage.load() },
                    onDisconnect = {
                        storage.disconnect()
                        CookieManager.getInstance().removeAllCookies(null)
                        CookieManager.getInstance().flush()
                        connection = storage.load()
                        tab = 0
                    }
                )
            }
        }
    }
}

@Composable
private fun ConnectionForm(initial: Connection, isEditing: Boolean,
                           onSave: (Connection) -> Unit, onDisconnect: () -> Unit) {
    var url by remember(initial.url) { mutableStateOf(initial.url) }
    var token by remember(initial.token) { mutableStateOf(initial.token) }
    var allowHttp by remember(initial.allowHttp) { mutableStateOf(initial.allowHttp) }
    val target = Connection(url, token, allowHttp, true)
    LazyColumn(
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
        item {
            Text(if (isEditing) "Settings" else "Connect to Distill",
                style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            Text("Paste the complete URL (with ?token=), or enter the server address and token separately.",
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        item {
            OutlinedTextField(
                value = url, onValueChange = { newText ->
                    val login = ServerAddress.extractLoginLink(newText)
                    if (login != null) {
                        url = login.first
                        token = login.second
                    } else {
                        url = newText
                    }
                }, modifier = Modifier.fillMaxWidth(),
                label = { Text("Server URL") },
                placeholder = { Text("https://server.example:8766") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri)
            )
        }
        item {
            OutlinedTextField(
                value = token, onValueChange = { token = it }, modifier = Modifier.fillMaxWidth(),
                label = { Text("Access token") },
                visualTransformation = PasswordVisualTransformation(),
                singleLine = true
            )
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = allowHttp, onCheckedChange = { allowHttp = it })
                Text("Allow unencrypted HTTP")
            }
            if (allowHttp) {
                Text("HTTP exposes your token to the network. Use only on a trusted private LAN or VPN.",
                    color = Color(0xFF9A6400), style = MaterialTheme.typography.bodySmall)
            }
        }
        item {
            Button(onClick = { onSave(target) }, enabled = target.base != null && token.isNotBlank(),
                modifier = Modifier.fillMaxWidth()) {
                Text(if (isEditing) "Save connection" else "Connect")
            }
        }
        if (isEditing) {
            item {
                TextButton(onClick = onDisconnect) {
                    Text("Disconnect and forget token", color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}

@Composable
private fun DashboardScreen(connection: Connection, modifier: Modifier = Modifier) {
    var snapshot by remember { mutableStateOf<Dashboard?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(connection.url, connection.token) {
        while (true) {
            try {
                snapshot = DistillApi.status(connection)
                error = null
            } catch (exception: Exception) {
                error = exception.localizedMessage ?: "Server unavailable"
            }
            delay(10_000)
        }
    }
    LazyColumn(modifier = modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            Text("Distill", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Text(if (error == null) "Live system overview" else "Connection unavailable",
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (error != null) item { Text(error ?: "", color = MaterialTheme.colorScheme.error) }
        val value = snapshot
        if (value == null) {
            item {
                Box(Modifier.fillMaxWidth().height(150.dp), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            }
        } else {
            items(value.metrics.chunked(2)) { pair ->
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    pair.forEach { metric ->
                        Card(modifier = Modifier.weight(1f), shape = RoundedCornerShape(20.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(metric.title, style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(metric.value, style = MaterialTheme.typography.headlineSmall,
                                    fontWeight = FontWeight.SemiBold)
                                Text(metric.detail, style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2)
                            }
                        }
                    }
                    if (pair.size == 1) Spacer(Modifier.weight(1f))
                }
            }
            item {
                SectionCard("Training") {
                    if (value.training.isEmpty()) Text("No active training", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    value.training.forEach { run ->
                        Text(run.name, fontWeight = FontWeight.Medium)
                        run.percent?.let {
                            androidx.compose.material3.LinearProgressIndicator(
                                progress = { (it / 100).toFloat().coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Text("${it.toInt()}% • ${run.done ?: 0} / ${run.total ?: 0} steps",
                                style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            item {
                SectionCard("Running jobs") {
                    if (value.jobs.isEmpty()) Text("No jobs running", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    value.jobs.forEach { job ->
                        Text(job.label, fontWeight = FontWeight.Medium)
                        job.summary?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                    }
                }
            }
            item { SectionCard("Comma") { Text(value.comma) } }
            item {
                Text("Server code: ${value.code}", style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(shape = RoundedCornerShape(20.dp), modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}
