# Distill for Android

The companion app of distill-web, the control site of this repository: the whole site in the app, plus a
native overview of the server. It is not an openpilot/sunnypilot build and it never talks to the comma
or the car.

## Install

Every change to `app/android/` on `master` is built by GitHub Actions
([`.github/workflows/android.yml`](../../.github/workflows/android.yml)) and published as a release
`android-v0.2.<build>` with `distill-android-0.2.<build>.apk`. Download the APK on the phone, open it and
allow installing apps from that source. Android 8.0 (API 26) or newer.

## Use

1. **Connect**: the address you open distill-web at, e.g. `http://192.168.1.10:8766`; a pasted page
   address (`…/#status`) is reduced to the server. The app checks that a Distill site answers there
   (`GET /api/auth/state`) before keeping it. Plain HTTP needs the explicit switch: use it only on your
   own network or a VPN; prefer HTTPS through a reverse proxy.
2. **Sign in** on the Website tab with the administrator of the site, as in a browser. The session cookie
   (HttpOnly, 30 days) stays in the app's WebView; the app stores no password and no token.
3. **Website**: the full site; Android back goes to the previous page of the site, links to other sites
   open in the browser. **Overview**: CPU, GPU, RAM, disk, training, running jobs and the comma, read
   from `GET /api/status` every 10 s with the same session. **Server**: change the address, or
   disconnect (signs the phone out: its cookies are erased).

Actions (training, install, publish…) happen on the site only, where the server checks them.

## Build

JDK 17, Android SDK platform 35 and build-tools 35, Gradle 8.9 (no wrapper jar is committed):

```bash
cd app/android
gradle :app:testDebugUnitTest :app:assembleDebug
```

## Release signing

Android installs an update only when it is signed with the same key as the installed app. Without
signing secrets, CI signs every build with a fresh debug key, so an update means uninstalling first. For
updates that install over the previous version, add these repository secrets (GitHub › Settings ›
Secrets and variables › Actions); the key never goes into git:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | the keystore file (PKCS12 or JKS), base64 |
| `ANDROID_KEYSTORE_PASSWORD` | its password |
| `ANDROID_KEY_ALIAS` | the alias of the key in it |
| `ANDROID_KEY_PASSWORD` | the password of the key |

Keep a copy of the keystore: a lost key means users must uninstall to get the next version.
