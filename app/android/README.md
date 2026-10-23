# Distill for Android

The companion app of distill-web, the control site of this repository: native screens to follow the
server and to start trainings, evaluations and installations on the comma, plus the whole site. It is
not an openpilot/sunnypilot build: it talks to your distill-web server only, never to the comma or the
car directly.

## Install

Two ways, the same app:

- **From your server**: `install.sh` builds it when it installs the server, and the code update of the
  site builds it again when `app/android` changes; open distill-web on the phone, Settings › Distill for
  Android › Download the app.
- **From GitHub**: every change to `app/android/` on `master` is built by GitHub Actions
  ([`.github/workflows/android.yml`](../../.github/workflows/android.yml)) and published as a release
  `android-v0.3.<commits>` with `distill-android-0.3.<commits>.apk`.

Open the APK on the phone and allow installing apps from that source. Android 8.0 (API 26) or newer.
Both use version 0.3.<number of commits> and versionCode 1000 + that number, so a build of the server and
the GitHub release of the same commit install over each other (when signed with the same key, below).

## Use

1. **Connect**: the address you open distill-web at, e.g. `http://192.168.1.10:8766`; a pasted page
   address (`…/#status`) is reduced to the server. The app checks that a Distill site answers there
   (`GET /api/auth/state`) before keeping it. Plain HTTP needs the explicit switch: use it only on your
   own network or a VPN; prefer HTTPS through a reverse proxy.
2. **Sign in** on the Website tab with the administrator of the site, as in a browser. The session cookie
   (HttpOnly, 30 days) stays in the app's WebView; the app stores no password and no token.
3. **Overview**: the server every 5 s (CPU, GPU, RAM, disk, training, the comma) and the running jobs, with
   their progress and a Stop button. The gear opens **the server settings**: change the address, or
   disconnect (signs the phone out: its cookies are erased).
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
the server checks it exactly as it checks the site. `ApiContractTest` reads `/api/status` and `/api/runs`
fixtures produced by `web/server.py`: regenerate them when the server changes those payloads.

## Build

On the server, without GitHub:

```bash
helpers/build_android.sh
```

sets up what is missing (JDK 17, Android SDK platform 35 and build-tools 35, Gradle 8.9 in `/opt/android`
or `DISTILL_ANDROID_TOOLS`; the signing key of the server in `~/.config/distill-android/` or
`DISTILL_ANDROID_KEYS`), runs the unit tests and builds the release APK into `outputs/android/` (the last 3
are kept). `--setup-only` stops before the build. `install.sh` (step 5) and the code update of the site run
it, so a new server needs nothing by hand; x86_64 only, as the SDK build tools. No wrapper jar is
committed; by hand: `gradle :app:testDebugUnitTest :app:assembleDebug` in `app/android`.

## Release signing

Android installs an update only when it is signed with the same key as the installed app. Without
signing secrets, CI signs every build with a fresh debug key, so an update means uninstalling first. For
updates that install over the previous version, add these repository secrets (GitHub › Settings ›
Secrets and variables › Actions), and keep the same key in `~/.config/distill-android/` on the server
for `helpers/build_android.sh`: it makes that key on the first build (`distill-release.p12` and one
`ANDROID_*.txt` file per secret below, so the secrets are a copy and paste). The key never goes into git:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | the keystore file (PKCS12 or JKS), base64 |
| `ANDROID_KEYSTORE_PASSWORD` | its password |
| `ANDROID_KEY_ALIAS` | the alias of the key in it |
| `ANDROID_KEY_PASSWORD` | the password of the key |

Keep a copy of the keystore: a lost key means users must uninstall to get the next version.
