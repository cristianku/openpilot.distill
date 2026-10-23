package ch.bodhi.distill

import android.os.Bundle
import android.webkit.CookieManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Dashboard
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
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
private const val TAB_OVERVIEW = 0
private const val TAB_WEBSITE = 1
private const val TAB_SETTINGS = 2

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
    var tab by remember { mutableStateOf(TAB_WEBSITE) }
    if (!connection.connected || connection.base == null) {
        ConnectionForm(
            initial = connection,
            isEditing = false,
            onSave = {
                storage.save(it)
                connection = storage.load()
                // Next: sign in with the administrator of the site, on the site itself.
                tab = TAB_WEBSITE
            },
            onDisconnect = {}
        )
        return
    }
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                listOf(Triple("Overview", Icons.Outlined.Dashboard, TAB_OVERVIEW),
                       Triple("Website", Icons.Outlined.Language, TAB_WEBSITE),
                       Triple("Server", Icons.Outlined.Settings, TAB_SETTINGS)).forEach { (label, icon, index) ->
                    NavigationBarItem(
                        selected = tab == index,
                        onClick = { tab = index },
                        icon = { Icon(icon, contentDescription = label) },
                        label = { Text(label) }
                    )
                }
            }
        }
    ) { padding ->
        when (tab) {
            TAB_OVERVIEW -> DashboardScreen(connection, onSignIn = { tab = TAB_WEBSITE }, modifier = Modifier.padding(padding))
            TAB_WEBSITE -> Portal(connection, webView, Modifier.fillMaxSize().padding(padding))
            else -> Box(Modifier.padding(padding)) {
                ConnectionForm(
                    initial = connection, isEditing = true,
                    onSave = {
                        storage.save(it)
                        connection = storage.load()
                        tab = TAB_WEBSITE
                    },
                    onDisconnect = {
                        storage.disconnect()
                        CookieManager.getInstance().removeAllCookies(null)
                        CookieManager.getInstance().flush()
                        webView.loadUrl("about:blank")
                        webView.clearHistory()
                        connection = storage.load()
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
    var allowHttp by remember(initial.allowHttp) { mutableStateOf(initial.allowHttp) }
    var checking by remember { mutableStateOf(false) }
    var problem by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()
    val target = Connection(ServerAddress.fromPasted(url), allowHttp, true)
    LazyColumn(
        contentPadding = PaddingValues(horizontal = 22.dp, vertical = 36.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
        item {
            Text(if (isEditing) "Your server" else "Connect to your Distill server",
                fontSize = 30.sp, fontWeight = FontWeight.Bold, lineHeight = 35.sp,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
            Text("The address you open distill-web at, for example http://192.168.1.10:8766. Then you sign in "
                + "with the administrator of the site, as in the browser.",
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
                        val ok = DistillApi.probe(base)
                        checking = false
                        if (ok) onSave(target) else problem = "No Distill site answers at $base."
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
private fun DashboardScreen(connection: Connection, onSignIn: () -> Unit, modifier: Modifier = Modifier) {
    var snapshot by remember { mutableStateOf<Dashboard?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var signInNeeded by remember { mutableStateOf(false) }
    LaunchedEffect(connection.url) {
        while (true) {
            try {
                snapshot = DistillApi.status(connection)
                error = null
                signInNeeded = false
            } catch (_: SignInNeeded) {
                signInNeeded = true
            } catch (exception: Exception) {
                error = exception.localizedMessage ?: "Server unavailable"
            }
            delay(10_000)
        }
    }
    LazyColumn(modifier = modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            Text("Status", fontSize = 30.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
            Text(when {
                signInNeeded -> "Sign in first"
                error != null -> "Connection unavailable"
                else -> "How the server is doing, every 10 s"
            }, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth())
        }
        if (signInNeeded) {
            item {
                SectionCard("Not signed in") {
                    Text("Sign in on the Website tab with the administrator of the site; this screen then follows "
                        + "your server.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Button(onClick = onSignIn, shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
                        Text("Sign in")
                    }
                }
            }
            return@LazyColumn
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
                        Card(modifier = Modifier.weight(1f), shape = RoundedCornerShape(18.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(metric.title.uppercase(), style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(metric.value, style = MaterialTheme.typography.headlineSmall,
                                    fontWeight = FontWeight.Bold)
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
                            LinearProgressIndicator(
                                progress = { (it / 100).toFloat().coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Text("${it.toInt()}% · ${run.done ?: 0} / ${run.total ?: 0} steps",
                                style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            item {
                SectionCard("Running") {
                    if (value.jobs.isEmpty()) Text("Nothing running: the server is free.", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
    Card(shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title.uppercase(), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            content()
        }
    }
}
