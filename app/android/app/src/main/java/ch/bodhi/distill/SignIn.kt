package ch.bodhi.distill

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

/** Sign in with the administrator of the site (POST /api/auth/login), in place of a screen until the app has
 * a session. The session is the one of the Website tab too. A server without its administrator yet sends
 * to the site, which creates it with the setup code. */
@Composable
fun SignInScreen(base: String, onSignedIn: () -> Unit, onWebsite: () -> Unit) {
    var info by remember(base) { mutableStateOf<ServerInfo?>(null) }
    var user by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var problem by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val demo = base == DemoServer.BASE
    LaunchedEffect(base) { info = DistillApi.probe(base) }

    fun submit() {
        if (user.isBlank() || password.isEmpty() || busy) return
        busy = true
        problem = ""
        scope.launch {
            try {
                DistillApi.signIn(base, user.trim(), password)
                password = ""
                onSignedIn()
            } catch (exception: CancellationException) {
                throw exception
            } catch (exception: Exception) {
                problem = exception.localizedMessage ?: "The sign-in did not work."
            }
            busy = false
        }
    }

    Column(
        Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).verticalScroll(rememberScrollState())
            .padding(horizontal = 22.dp, vertical = 28.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically)
    ) {
        Text("Sign in", fontSize = 30.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth())
        Note(if (demo) "The demo server: name ${DemoServer.USER}, password ${DemoServer.PASSWORD}."
             else "With the administrator of your site, as in the browser. ${base.substringAfter("://")}")
        if (info?.configured == false) {
            Note("This server has no administrator yet: create it on the website, with the setup code the " +
                "installer printed. The app is signed in at the same time.")
            PrimaryAction("Open the website", "", onClick = onWebsite)
        } else {
            info?.problem?.takeIf { it.isNotEmpty() }?.let { Note(it, Danger) }
            OutlinedTextField(
                value = user, onValueChange = { user = it; problem = "" }, label = { Text("Name") },
                singleLine = true, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(autoCorrectEnabled = false, keyboardType = KeyboardType.Ascii,
                    imeAction = ImeAction.Next)
            )
            OutlinedTextField(
                value = password, onValueChange = { password = it; problem = "" }, label = { Text("Password") },
                singleLine = true, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth(),
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { submit() })
            )
            PrimaryAction("Sign in", if (user.isBlank() || password.isEmpty()) "enter the name and the password" else "",
                busy = busy) { submit() }
            if (problem.isNotEmpty()) Note(problem, Danger)
            if (!demo) {
                TextButton(onClick = onWebsite, modifier = Modifier.fillMaxWidth()) {
                    Text("Sign in on the website instead")
                }
            }
        }
    }
}

/** The Website tab of the demo: there is no site behind it. */
@Composable
fun DemoWebsite(modifier: Modifier = Modifier) {
    Column(modifier.background(MaterialTheme.colorScheme.background).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically)) {
        Text("The website", fontSize = 28.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth())
        Note("With your own server this tab shows the whole distill-web site, signed in with the session of the " +
            "app: drives, charts of the trainings, comparison videos, settings. The demo has no website; the " +
            "other tabs show what the app does.")
    }
}
