# Distill Mobile

Native companion apps for the existing `openpilot.distill` local web service.

- `ios/`: SwiftUI + WKWebView (iOS 17+). Generate the Xcode project with XcodeGen.
- `android/`: Kotlin + Jetpack Compose + Android WebView.
- Both use the **existing** `GET /api/status` with an `X-Distill-Token` header for the native read-only dashboard.
- Both open the full existing website in a first-party WebView. `GET /?token=...` returns a same-origin HttpOnly cookie and redirects to `/`; no custom login endpoint or CORS policy is required.
- Settings are stored on device; the token is stored in the iOS Keychain or Android Keystore, never in source control.
- Website actions (train, install, publish, stop, etc.) are available only in the web view, where the existing server enforces action constraints.
- This is not an installable comma/Sunnypilot branch.

## Connection

1. Start `distill-web` so your phone can reach it on a private network or VPN (e.g. `http://192.168.1.10:8766`).
2. Enter the **base URL** (no path, query or embedded token) and the current `DISTILL_WEB_TOKEN`.
3. Prefer HTTPS using a trusted reverse proxy / private VPN. If your server is HTTP-only, explicitly opt in to insecure LAN HTTP. HTTP sends the token unencrypted; never expose this service publicly.
4. Open Dashboard for CPU/GPU/RAM/training/job status, or Website to operate the full existing web UI.

## Scope and merge policy

This branch modifies **only `app/`**. The site and backend live on `master` and may be modified by a separate agent. Keep `app/` changes in this branch so the pull request remains easy to merge.

## Current API contract

`GET /api/status`, header `X-Distill-Token`, JSON fields:
`system.cpu.total`, `system.gpu.utilization`, `system.memory.used_percent`,
`system.disk.{total_gb,free_gb}`, `jobs[]`, `training[]`, `comma`, `code`.
Fields may be null (e.g. no NVIDIA GPU). No direct destructive API calls from native dashboard.

For the website session, use `GET /?token=<urlencoded-token>`; server responds `303` with a port-specific HttpOnly SameSite=Strict cookie and strips the query string. The WebView never forwards the token to another origin.

## Limitations

This repository's build environment does not include Xcode or the Android SDK, so the native apps must be built/tested on a Mac (iOS) and an Android SDK host (Android) before release. The Android Gradle wrapper executable/JAR is not checked in: run `gradle wrapper --gradle-version 8.9` once in `app/android` with a compatible Gradle installation, or use the IDE's Gradle setup.
