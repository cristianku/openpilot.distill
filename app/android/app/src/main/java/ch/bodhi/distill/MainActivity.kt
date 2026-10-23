package ch.bodhi.distill

import android.os.Bundle
import android.webkit.CookieManager
import android.webkit.WebStorage
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Dashboard
import androidx.compose.material.icons.outlined.DirectionsCar
import androidx.compose.material.icons.outlined.Insights
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material.icons.outlined.ModelTraining
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// The palette of the site (web/static/app.css, dark navy with blue accents).
private val SiteColors = darkColorScheme(
    primary = Color(0xFF3B82F6), onPrimary = Color.White,
    background = Color(0xFF0B0F16), onBackground = Color(0xFFE8ECF3),
    surface = Color(0xFF161B24), onSurface = Color(0xFFE8ECF3),
    surfaceVariant = Color(0xFF1C2230), onSurfaceVariant = Color(0xFF9AA4B2),
    secondaryContainer = Color(0x293B82F6), onSecondaryContainer = Color(0xFFE8ECF3),
    error = Color(0xFFF87171), outline = Color(0x24FFFFFF)
)
private val Warning = Color(0xFFFBBF24)

/** The tabs at the bottom; the server settings open from the gear of Overview, as on the site. */
private enum class Tab(val label: String) { OVERVIEW("Overview"), TRAIN("Train"), EVALUATE("Evaluate"), COMMA("Comma"), WEBSITE("Website") }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge to edge, as Android 15 and later draw every app; light icons on the navy of the site.
        enableEdgeToEdge(SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        val storage = ConnectionSettings(applicationContext)
        setContent {
            MaterialTheme(colorScheme = SiteColors) {
                DistillScreen(storage)
            }
        }
    }
}

@Composable
private fun DistillScreen(storage: ConnectionSettings) {
    val context = LocalContext.current
    val webView = remember { siteWebView(context) }
    var connection by remember { mutableStateOf(storage.load()) }
    var tab by remember { mutableStateOf(Tab.OVERVIEW) }
    var serverSettings by remember { mutableStateOf(false) }
    val base = connection.base
    if (!connection.connected || base == null) {
        ConnectionForm(
            initial = connection,
            isEditing = false,
            onSave = {
                storage.save(it)
                connection = storage.load()
                // Next: the sign-in, in place of Overview until the app has a session.
                tab = Tab.OVERVIEW
            },
            onDisconnect = {}
        )
        return
    }
    val data = remember(base) { ServerData(base) }
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    // The state of the server every 5 s; what the native screens choose from every 20 s, while they are open.
    LaunchedEffect(base) {
        while (true) {
            data.refreshStatus()
            delay(5_000)
        }
    }
    LaunchedEffect(base, tab) {
        if (tab == Tab.WEBSITE || tab == Tab.OVERVIEW) return@LaunchedEffect
        while (true) {
            data.refreshLists()
            delay(20_000)
        }
    }
    // A job was started: say so, and show it running in Overview.
    val onStarted: (String) -> Unit = { label ->
        tab = Tab.OVERVIEW
        scope.launch { snackbar.showSnackbar("Started: $label") }
    }

    if (serverSettings) {
        BackHandler { serverSettings = false }
        ConnectionForm(
            initial = connection, isEditing = true,
            onSave = {
                storage.save(it)
                connection = storage.load()
                serverSettings = false
                tab = Tab.OVERVIEW
            },
            onDisconnect = {
                scope.launch {
                    // The session ends on the server too, then nothing of it stays on the phone.
                    DistillApi.signOut(base)
                    storage.disconnect()
                    CookieManager.getInstance().removeAllCookies(null)
                    CookieManager.getInstance().flush()
                    WebStorage.getInstance().deleteAllData()
                    webView.loadUrl("about:blank")
                    webView.clearHistory()
                    webView.clearCache(true)
                    serverSettings = false
                    connection = storage.load()
                }
            },
            onClose = { serverSettings = false }
        )
        return
    }
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                Tab.entries.forEach { item ->
                    NavigationBarItem(
                        selected = tab == item,
                        onClick = { tab = item },
                        icon = {
                            Icon(when (item) {
                                Tab.OVERVIEW -> Icons.Outlined.Dashboard
                                Tab.TRAIN -> Icons.Outlined.ModelTraining
                                Tab.EVALUATE -> Icons.Outlined.Insights
                                Tab.COMMA -> Icons.Outlined.DirectionsCar
                                Tab.WEBSITE -> Icons.Outlined.Language
                            }, contentDescription = item.label)
                        },
                        label = { Text(item.label, maxLines = 1) }
                    )
                }
            }
        }
    ) { padding ->
        val screen = Modifier.fillMaxSize().padding(padding).consumeWindowInsets(padding).imePadding()
        if (tab != Tab.WEBSITE && data.signInNeeded) {
            Box(screen) {
                SignInScreen(base, onSignedIn = {
                    // The site in the Website tab shows its sign-in page: open it again with the new session.
                    if (!connection.demo) webView.loadUrl("$base/")
                    scope.launch { data.refreshStatus() }
                }, onWebsite = { tab = Tab.WEBSITE })
            }
            return@Scaffold
        }
        when (tab) {
            Tab.OVERVIEW -> Box(screen) { DashboardScreen(data, onSettings = { serverSettings = true }) }
            Tab.TRAIN -> Box(screen) { TrainScreen(data, onStarted) }
            Tab.EVALUATE -> Box(screen) { EvaluateScreen(data, onStarted) }
            Tab.COMMA -> Box(screen) { CommaScreen(data, onStarted) }
            Tab.WEBSITE -> if (connection.demo) DemoWebsite(screen) else Portal(connection, webView, screen)
        }
    }
}

@Composable
private fun ConnectionForm(initial: Connection, isEditing: Boolean,
                           onSave: (Connection) -> Unit, onDisconnect: () -> Unit, onClose: (() -> Unit)? = null) {
    var url by remember(initial.url) { mutableStateOf(if (initial.demo) DemoServer.ADDRESS else initial.url) }
    var allowHttp by remember(initial.allowHttp) { mutableStateOf(initial.allowHttp) }
    var checking by remember { mutableStateOf(false) }
    var problem by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()
    val target = ServerAddress.connection(url, allowHttp)
    LazyColumn(
        contentPadding = PaddingValues(horizontal = 22.dp, vertical = 36.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
            .windowInsetsPadding(WindowInsets.safeDrawing)
    ) {
        onClose?.let { close ->
            item {
                TextButton(onClick = close) { Text("‹ Back") }
            }
        }
        item {
            Text(if (isEditing) "Your server" else "Connect to your Distill server",
                fontSize = 30.sp, fontWeight = FontWeight.Bold, lineHeight = 35.sp,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
            Text("The address you open distill-web at, for example http://192.168.1.10:8766. Then you sign in "
                + "with the administrator of the site, as in the browser. No server yet? Type demo to try the app.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
        }
        item {
            OutlinedTextField(
                value = url, onValueChange = { url = it; problem = null }, modifier = Modifier.fillMaxWidth(),
                label = { Text("Server address") },
                placeholder = { Text("https://server.example:8766") },
                singleLine = true, shape = RoundedCornerShape(12.dp),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri)
            )
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Switch(checked = allowHttp, onCheckedChange = { allowHttp = it; problem = null })
                Text("Allow plain HTTP")
            }
            if (allowHttp) {
                Spacer(Modifier.height(6.dp))
                Text("Over HTTP your password and session travel unencrypted: only on your own network or a VPN.",
                    color = Warning, style = MaterialTheme.typography.bodySmall)
            } else if (url.trim().startsWith("http://", ignoreCase = true)) {
                Spacer(Modifier.height(6.dp))
                Text("This address is plain HTTP: allow it above, or use an HTTPS address.",
                    color = Warning, style = MaterialTheme.typography.bodySmall)
            }
        }
        problem?.let { item { Text(it, color = MaterialTheme.colorScheme.error) } }
        item {
            Button(
                onClick = {
                    val base = target.base ?: return@Button
                    checking = true
                    scope.launch {
                        val info = DistillApi.probe(base)
                        checking = false
                        when {
                            info == null -> problem = "No Distill site answers at $base."
                            info.problem.isNotEmpty() -> problem = info.problem
                            else -> onSave(target)
                        }
                    }
                },
                enabled = target.base != null && !checking,
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                modifier = Modifier.fillMaxWidth().height(52.dp)
            ) {
                if (checking) CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp, color = Color.White)
                else Text(if (isEditing) "Save" else "Connect", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
            }
        }
        if (isEditing) {
            item {
                TextButton(onClick = onDisconnect, modifier = Modifier.fillMaxWidth()) {
                    Text("Disconnect and sign out of this phone", color = MaterialTheme.colorScheme.error)
                }
            }
        } else {
            item {
                Text("Step 1 of 2 · then sign in", color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun DashboardScreen(data: ServerData, onSettings: () -> Unit) {
    val value = data.status
    val error = data.error
    var stopping by remember { mutableStateOf<Job?>(null) }
    val scope = rememberCoroutineScope()
    LazyColumn(modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            Box(Modifier.fillMaxWidth()) {
                Column(Modifier.fillMaxWidth().padding(horizontal = 40.dp)) {
                    Text("Status", fontSize = 30.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth())
                    Text(when {
                        error != null -> "Connection unavailable"
                        value == null -> "Reading the server…"
                        value.jobs.isEmpty() -> "Nothing running: the server is free."
                        value.jobs.size == 1 -> "Working on: " + value.jobs.first().label + "."
                        else -> "${value.jobs.size} jobs running."
                    }, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth())
                }
                IconButton(onClick = onSettings, modifier = Modifier.align(Alignment.TopEnd)) {
                    Icon(Icons.Outlined.Settings, contentDescription = "Server settings")
                }
            }
        }
        if (error != null) item { Text(error, color = MaterialTheme.colorScheme.error) }
        if (value == null) {
            item {
                Box(Modifier.fillMaxWidth().height(150.dp), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            }
            return@LazyColumn
        }
        if (value.jobs.isNotEmpty()) {
            item {
                SectionCard("Running") {
                    value.jobs.forEach { job ->
                        Text(job.label, fontWeight = FontWeight.Medium)
                        if (job.summary.isNotBlank()) {
                            Text(job.summary, style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        job.percent?.let {
                            LinearProgressIndicator(progress = { (it / 100).toFloat().coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth())
                            Text("${it.toInt()}%" + (if (job.done != null && job.total != null) " · ${job.done} / ${job.total}" else ""),
                                style = MaterialTheme.typography.bodySmall)
                        }
                        if (job.stoppable) {
                            OutlinedButton(onClick = { stopping = job }, shape = RoundedCornerShape(12.dp)) {
                                Text("Stop", color = Danger)
                            }
                        }
                    }
                }
            }
        }
        items(value.metrics.chunked(2)) { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                pair.forEach { metric ->
                    Card(modifier = Modifier.weight(1f), shape = RoundedCornerShape(18.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(metric.title.uppercase(), style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(metric.value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
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
                        LinearProgressIndicator(progress = { (it / 100).toFloat().coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth())
                        Text("${it.toInt()}% · ${run.done ?: 0} / ${run.total ?: 0} steps",
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        item {
            SectionCard("Comma") {
                Text(when {
                    value.comma.reachable == null -> "Not checked yet"
                    !value.comma.reachable -> "Not reachable"
                    value.comma.onroad -> "Driving"
                    else -> "Reachable, parked"
                })
            }
        }
        item {
            Text("Server code: ${value.code}", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
    stopping?.let { job ->
        Confirm("Stop?", "Stop \"${job.label}\"? It stops within about twenty seconds.", "Stop",
            onConfirm = {
                stopping = null
                scope.launch {
                    try {
                        data.stop(job)
                    } catch (_: Exception) {
                    }
                }
            }, onDismiss = { stopping = null })
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title.uppercase(), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            content()
        }
    }
}
