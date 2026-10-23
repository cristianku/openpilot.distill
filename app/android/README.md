# Distill for Android

The companion app of distill-web, the control site of this repository: native screens to follow the
server and to start trainings, evaluations and installations on the comma, plus the whole site. It is
not an openpilot/sunnypilot build: it talks to your distill-web server only, never to the comma or the
car directly, and only through the API of the server ([`web/API.md`](../../web/API.md)).

## Install

Two ways, the same app:

- **From Google Play**, once published ([`GOOGLE_PLAY.md`](GOOGLE_PLAY.md)): updates come by themselves.
- **From your server**: `install.sh` builds it when it installs the server, and the code update of the
  site builds it again when `app/android` changes; open distill-web on the phone, Settings › Distill for
  Android › Download the app, and allow installing apps from that source.

The two are signed with different keys (Google's, and the key of your server): to move from one to the
other, uninstall the app first. Android 8.0 (API 26) or newer.

## Use

1. **Connect**: the address you open distill-web at, e.g. `http://192.168.1.10:8766`; a pasted page
   address (`…/#status`) is reduced to the server. The app checks that a Distill site answers there
   (`GET /api/auth/state`) and that its API is one the app knows, before keeping it. Plain HTTP needs the
   explicit switch: use it only on your own network or a VPN; prefer HTTPS through a reverse proxy.
   **`demo`** as the address is a made-up server inside the app (name `demo`, password `demo`): to try the
   app, and for the Google Play review. Nothing leaves the phone.
2. **Sign in** with the administrator of the site, in the app (`POST /api/auth/login`) or on the site in
   the Website tab. The session cookie (HttpOnly, 30 days) stays in the app's WebView, so the native screens
   and the site are signed in together; the app stores no password and no token. A server that has no
   administrator yet sends you to the site, which creates it with the setup code.
3. **Overview**: the server every 5 s (CPU, GPU, RAM, disk, training, the comma) and the running jobs, with
   their progress and a Stop button. The gear opens **the server settings**: change the address, or
   disconnect (signs out on the server and erases the session and the site's data from the phone).
4. **Train**, **Evaluate**, **Comma**: native wizards, the same steps and rules as the site.
   - Train: where to start (new model, continue from the stable weights, resume a stopped run), how much
     (Trial 1,000 / Standard 5,000 / custom steps, with the time it takes on this server), then the summary
     and Start. Batch 8, 6 workers and the GPU are set as the site does by default.
   - Evaluate: which model (yours, the references, the published ones, with their score), then Evaluate.
   - Comma: which finished model, then the checklist of the comma, Check now, Install (with a
     confirmation) or put the original model back.
   A button is enabled only when the server allows the action, and says why not otherwise.
5. **Website**: the full site; Android back goes to the previous page of the site, links to other sites
   open in the browser.

Every action goes through the same `POST /api/run/<action>` as the site, with the session of the app, so
the server checks it exactly as it checks the site. The app on a phone and the server update separately:
[`web/API.md`](../../web/API.md) is the contract between them, with its version (`api` in
`/api/auth/state`). `ApiContractTest` reads `/api/status` and `/api/runs` fixtures produced by
`web/server.py`: regenerate them when the server changes those payloads; `AppApiTest` in
`tests/test_web_server.py` calls the server as the app does.

## Build

Always on the server, never by GitHub Actions:

```bash
helpers/build_android.sh                 # tests and the APK the site offers
helpers/build_android.sh --bundle        # also the AAB for Google Play, into outputs/android/
helpers/build_android.sh --play[=alpha]  # build and upload to Google Play (GOOGLE_PLAY.md)
```

It sets up what is missing (JDK 17, Android SDK platform 36 and build-tools 35, Gradle 8.14.3 in
`/opt/android` or `DISTILL_ANDROID_TOOLS`; the signing key of the server in `~/.config/distill-android/` or
`DISTILL_ANDROID_KEYS`), runs the unit tests and builds the release APK (R8, signed with that key) into
`outputs/android/` (the last 3 are kept). `--setup-only` stops before the build. `install.sh` (step 5) and the
code update of the site run it, so a new server needs nothing by hand; x86_64 only, as the SDK build tools.
Toolchain pinned as nav2pilot's: Android Gradle Plugin 8.13.2, Kotlin 2.2.21, Compose BOM 2025.12.01,
`compileSdk`/`targetSdk` 36 (what Google Play requires), `minSdk` 26. No wrapper jar is committed; by hand,
with the tools of `/opt/android`: `gradle :app:testDebugUnitTest :app:assembleDebug` in `app/android`. A
release build without the key of the server stops: it is never signed with a debug key.

Version: `0.3.<commits of HEAD>`, versionCode `1000 + <commits>`, `-dev` when `app/android` has local changes.

## Signing

The key of the server (`~/.config/distill-android/distill-release.p12`, PKCS12, alias `distill`, made on the
first build, its password in `ANDROID_KEYSTORE_PASSWORD.txt`) signs the APK of the site and is the upload key
of Google Play, which signs what it delivers with its own key. It never goes into git. Keep a backup: Android
installs an update only when it is signed with the same key as the installed app.
