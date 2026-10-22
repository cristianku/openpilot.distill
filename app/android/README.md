# Android — Distill

Requires Android Studio, JDK 17, Android SDK 35. Minimum Android version 8.0 (API 26).

Open `app/android` in Android Studio, allow Gradle sync, then run the `app` target on a physical phone. If building from the CLI, generate a Gradle wrapper with a locally installed Gradle 8.9 (`gradle wrapper --gradle-version 8.9`) and run `./gradlew assembleDebug`.

The dashboard calls `GET /api/status` every 10 seconds while visible and uses the `X-Distill-Token` header. The Website tab opens the existing site in Android WebView, using its first-party session cookie after one `/?token=...` bootstrap.

The URL is stored in private app preferences. The access token is encrypted with an AES-GCM key stored in Android Keystore (not stored in the APK or exported in backups). Disconnect erases the stored token and WebView cookies.

Android cleartext traffic is enabled only to support the current HTTP-only local Distill deployment; the form requires explicit opt-in. Prefer a trusted HTTPS reverse proxy or VPN for real-world use. Restrict this setting before app-store distribution.
