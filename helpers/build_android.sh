#!/usr/bin/env bash
# Builds Distill for Android (app/android) on this server, without GitHub Actions: the unit tests, then
# the release APK in outputs/android/, which the site offers for download (Settings › Distill for Android).
# With --bundle or --play it also builds the Android App Bundle (AAB) that Google Play takes, and --play
# uploads it to Google Play. Everything is built here, never by GitHub.
# What is missing is set up first, so a fresh server needs nothing by hand (install.sh runs this script,
# the code update of the site runs it again when app/android changes):
#   - the tools, in /opt/android (or DISTILL_ANDROID_TOOLS): JDK 17, Android SDK platform 36 and
#     build-tools 35, Gradle 8.14.3 and its cache (about 1.5 GB; x86_64 only, as the SDK build tools)
#   - the signing key of this server, in ~/.config/distill-android (or DISTILL_ANDROID_KEYS):
#     distill-release.p12 (PKCS12, alias "distill", RSA 4096, ~27 years) and its password in
#     ANDROID_KEYSTORE_PASSWORD.txt. Never in git; keep a backup. For Google Play it is the upload key.
#
#   helpers/build_android.sh [--setup-only] [--bundle] [--play[=TRACK]] [--listing] [--help]
#     --setup-only   install the tools and make the key when missing, do not build
#     --bundle       also build the AAB, into outputs/android/ (e.g. for the first upload, by hand)
#     --play[=TRACK] build the AAB and upload it to Google Play, track internal (default), alpha
#                    (closed testing), beta (open testing) or production, then tag the commit
#                    android-v<version> and push the tag. Needs a clean app/android, HEAD pushed to
#                    origin, and the key of the Play service account in
#                    ~/.config/distill-android/play-service-account.json (app/android/GOOGLE_PLAY.md)
#     --listing      upload the store listing of app/android/app/src/main/play (texts and pictures)
#
# Version: 0.3.<commits of HEAD>, versionCode 1000 + <commits>: every commit of master has its own,
#   always higher, as Google Play requires. "-dev" is added when app/android has local changes (never
#   uploaded).
# Output: outputs/android/distill-android-<version>.apk (and .aab; the last 3 are kept) and
#   outputs/android/latest.json, what the site shows of the last build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOLS="${DISTILL_ANDROID_TOOLS:-/opt/android}"
KEYS="${DISTILL_ANDROID_KEYS:-$HOME/.config/distill-android}"
OUT="$ROOT/outputs/android"
JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
CMDLINE_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
GRADLE_VERSION="8.14.3"
GRADLE_URL="https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip"
# The SDK packages of app/android/app/build.gradle.kts (compileSdk 36; AGP 8.13 builds with build-tools 35).
PLATFORM="android-36"
BUILD_TOOLS="35.0.0"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SETUP_ONLY=0
BUNDLE=0
TRACK=""
LISTING=0
for argument in "$@"; do
  case "$argument" in
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --setup-only) SETUP_ONLY=1 ;;
    --bundle) BUNDLE=1 ;;
    --play) TRACK=internal ;;
    --play=*) TRACK="${argument#--play=}" ;;
    --listing) LISTING=1 ;;
    *) die "invalid option: $argument (see --help)" ;;
  esac
done
case "$TRACK" in
  ''|internal|alpha|beta|production) ;;
  *) die "unknown Google Play track: $TRACK (internal, alpha, beta or production)" ;;
esac
[ -n "$TRACK" ] && BUNDLE=1
PLAY_KEY="$KEYS/play-service-account.json"

# -- tools: each one only when missing, unpacked next to its final place and moved there at the end,
#    so an interrupted download leaves nothing half installed --
need() {
  for command_name in "$@"; do
    command -v "$command_name" >/dev/null 2>&1 || die "$command_name is missing (install.sh installs it)"
  done
}

setup_tools() {
  [ "$(uname -m)" = x86_64 ] || die "the Android build tools exist for x86_64 only (this machine: $(uname -m))"
  mkdir -p "$TOOLS"
  local work="$TOOLS/.download"
  rm -rf "$work"
  mkdir -p "$work"
  if [ ! -x "$TOOLS/jdk/bin/java" ]; then
    say "downloading JDK 17 (Eclipse Temurin) into $TOOLS/jdk"
    need curl tar
    curl -fsSL -o "$work/jdk.tar.gz" "$JDK_URL"
    mkdir -p "$work/jdk"
    tar -xzf "$work/jdk.tar.gz" -C "$work/jdk" --strip-components=1
    mv "$work/jdk" "$TOOLS/jdk"
  fi
  if [ ! -x "$TOOLS/sdk/cmdline-tools/latest/bin/sdkmanager" ]; then
    say "downloading the Android command-line tools into $TOOLS/sdk"
    need curl unzip
    curl -fsSL -o "$work/cmdline.zip" "$CMDLINE_URL"
    unzip -q "$work/cmdline.zip" -d "$work/cmdline"
    mkdir -p "$TOOLS/sdk/cmdline-tools"
    mv "$work/cmdline/cmdline-tools" "$TOOLS/sdk/cmdline-tools/latest"
  fi
  if [ ! -d "$TOOLS/sdk/platforms/$PLATFORM" ] || [ ! -d "$TOOLS/sdk/build-tools/$BUILD_TOOLS" ]; then
    say "installing Android SDK $PLATFORM and build-tools $BUILD_TOOLS (licenses accepted)"
    local sdkmanager="$TOOLS/sdk/cmdline-tools/latest/bin/sdkmanager"
    yes | JAVA_HOME="$TOOLS/jdk" "$sdkmanager" --sdk_root="$TOOLS/sdk" --licenses > "$work/licenses.log" 2>&1 || true
    JAVA_HOME="$TOOLS/jdk" "$sdkmanager" --sdk_root="$TOOLS/sdk" "platforms;$PLATFORM" "build-tools;$BUILD_TOOLS" \
      > "$work/sdk.log" 2>&1 || { tail -5 "$work/sdk.log" >&2; die "the Android SDK did not install"; }
  fi
  if [ ! -x "$TOOLS/gradle-$GRADLE_VERSION/bin/gradle" ]; then
    say "downloading Gradle $GRADLE_VERSION into $TOOLS/gradle-$GRADLE_VERSION"
    need curl unzip sha256sum
    curl -fsSL -o "$work/gradle.zip" "$GRADLE_URL"
    local expected
    expected="$(curl -fsSL "$GRADLE_URL.sha256")"
    [ "$(sha256sum "$work/gradle.zip" | cut -d' ' -f1)" = "$expected" ] || die "the Gradle download does not match its SHA-256"
    unzip -q "$work/gradle.zip" -d "$work"
    mv "$work/gradle-$GRADLE_VERSION" "$TOOLS/gradle-$GRADLE_VERSION"
  fi
  # The Gradle versions of earlier builds of this script are no longer used.
  for old in "$TOOLS"/gradle-[0-9]*; do
    if [ -d "$old" ] && [ "$old" != "$TOOLS/gradle-$GRADLE_VERSION" ]; then
      rm -rf "$old"
    fi
  done
  rm -rf "$work"
}

# -- the signing key of this server, made once --
setup_key() {
  [ -f "$KEYS/distill-release.p12" ] && return 0
  say "making the signing key of this server in $KEYS"
  (
    umask 077
    mkdir -p "$KEYS"
    chmod 700 "$KEYS"
    PW="$(head -c 18 /dev/urandom | base64 | tr '+/' '-_')"
    export PW
    "$TOOLS/jdk/bin/keytool" -genkeypair -noprompt -keystore "$KEYS/distill-release.p12.part" -storetype PKCS12 \
      -alias distill -keyalg RSA -keysize 4096 -validity 10000 -dname "CN=Distill for Android, O=openpilot.distill" \
      -storepass:env PW -keypass:env PW > /dev/null 2>&1
    mv "$KEYS/distill-release.p12.part" "$KEYS/distill-release.p12"
    printf '%s' "$PW" > "$KEYS/ANDROID_KEYSTORE_PASSWORD.txt"
    printf '%s' "$PW" > "$KEYS/ANDROID_KEY_PASSWORD.txt"
    printf '%s' distill > "$KEYS/ANDROID_KEY_ALIAS.txt"
    cat > "$KEYS/README.txt" <<'EOF'
Signing key of Distill for Android on this server (PKCS12, alias "distill", RSA 4096, valid ~27 years),
made by helpers/build_android.sh. KEEP A BACKUP of distill-release.p12 and its password somewhere safe:
a lost key means every user must uninstall the app to get the next version. Never put these files in git.

The passwords are in ANDROID_KEYSTORE_PASSWORD.txt and ANDROID_KEY_PASSWORD.txt, the alias in
ANDROID_KEY_ALIAS.txt. For Google Play this key is the upload key (Google signs the app it delivers with
its own key); the key of the Play service account goes next to it, as play-service-account.json
(see app/android/GOOGLE_PLAY.md in the repository).
EOF
  )
}

setup_tools
setup_key
[ "$SETUP_ONLY" -eq 1 ] && { say "Android tools in $TOOLS, signing key in $KEYS: ready"; exit 0; }

export JAVA_HOME="$TOOLS/jdk" ANDROID_HOME="$TOOLS/sdk" GRADLE_USER_HOME="$TOOLS/gradle-home"

count="$(git -C "$ROOT" rev-list --count HEAD)"
commit="$(git -C "$ROOT" rev-parse --short HEAD)"
version="0.3.$count"
if [ -n "$(git -C "$ROOT" status --porcelain -- app/android)" ]; then
  version="$version-dev"
fi
code=$((1000 + count))

export DISTILL_KEYSTORE="$KEYS/distill-release.p12"
DISTILL_KEYSTORE_PASSWORD="$(cat "$KEYS/ANDROID_KEYSTORE_PASSWORD.txt")"
DISTILL_KEY_ALIAS="$(cat "$KEYS/ANDROID_KEY_ALIAS.txt")"
DISTILL_KEY_PASSWORD="$(cat "$KEYS/ANDROID_KEY_PASSWORD.txt")"
export DISTILL_KEYSTORE_PASSWORD DISTILL_KEY_ALIAS DISTILL_KEY_PASSWORD

# Google Play gets only a commit that GitHub has too: the upload needs the key of the Play service account.
if [ -n "$TRACK" ] || [ "$LISTING" -eq 1 ]; then
  [ -f "$PLAY_KEY" ] || die "$PLAY_KEY is missing: the key of the Google Play service account (app/android/GOOGLE_PLAY.md)"
  export DISTILL_PLAY_CREDENTIALS="$PLAY_KEY"
fi
if [ -n "$TRACK" ]; then
  case "$version" in
    *-dev) die "app/android has local changes: commit and push them first, Google Play gets the pushed commit" ;;
  esac
  branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  git -C "$ROOT" fetch -q origin "$branch" || die "origin cannot be reached to check that $branch is pushed"
  [ "$(git -C "$ROOT" rev-parse HEAD)" = "$(git -C "$ROOT" rev-parse "origin/$branch")" ] \
    || die "$branch is not the same as origin/$branch: push first, Google Play gets the pushed commit"
fi

tasks=(:app:testDebugUnitTest :app:assembleRelease)
[ "$BUNDLE" -eq 1 ] && tasks+=(:app:bundleRelease)
# The upload comes after the tests and the build, in the same run: a failing test uploads nothing.
[ -n "$TRACK" ] && tasks+=(:app:publishReleaseBundle --track "$TRACK")
[ "$LISTING" -eq 1 ] && tasks+=(:app:publishListing)

say "Distill for Android $version (versionCode $code, commit $commit), signed with the key of $KEYS"
(cd "$ROOT/app/android" && nice -n 10 "$TOOLS/gradle-$GRADLE_VERSION/bin/gradle" "${tasks[@]}" \
  --no-daemon --console=plain -q -PdistillVersionCode="$code" -PdistillVersionName="$version")

apk="$ROOT/app/android/app/build/outputs/apk/release/app-release.apk"
[ -s "$apk" ] || die "the build made no APK"
mkdir -p "$OUT"
name="distill-android-$version.apk"
cp "$apk" "$OUT/$name.part" && mv "$OUT/$name.part" "$OUT/$name"
sha="$(sha256sum "$OUT/$name" | cut -d' ' -f1)"
bundle=""
if [ "$BUNDLE" -eq 1 ]; then
  aab="$ROOT/app/android/app/build/outputs/bundle/release/app-release.aab"
  [ -s "$aab" ] || die "the build made no AAB"
  bundle="distill-android-$version.aab"
  cp "$aab" "$OUT/$bundle.part" && mv "$OUT/$bundle.part" "$OUT/$bundle"
fi
# The Python of the project (install.sh made it); a system python3 otherwise.
PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" - "$OUT/latest.json" "$name" "$version" "$code" "$commit" release "$sha" "$bundle" <<'EOF'
import json, os, sys, time
path, name, version, code, commit, signed_with, sha, bundle = sys.argv[1:]
folder = os.path.dirname(path)
data = {'file': name, 'version': version, 'version_code': int(code), 'commit': commit, 'signed_with': signed_with,
        'sha256': sha, 'size': os.path.getsize(os.path.join(folder, name)), 'built': time.time()}
if bundle:
    data['bundle'] = bundle
with open(path + '.part', 'w') as out:
    json.dump(data, out, indent=1)
os.replace(path + '.part', path)
EOF
# Keep the last 3 APKs and AABs.
ls -1t "$OUT"/distill-android-*.apk | tail -n +4 | xargs -r rm -f
find "$OUT" -maxdepth 1 -name 'distill-android-*.aab' -printf '%T@ %p\n' | sort -rn | tail -n +4 | cut -d' ' -f2- \
  | xargs -r rm -f
say "APK: outputs/android/$name ($(awk -v b="$(stat -c %s "$OUT/$name")" 'BEGIN { printf "%.1f MB", b / 1e6 }'), sha256 $sha)"
if [ -n "$bundle" ]; then
  # The certificate Play Console shows for the upload key (Test and release › App integrity).
  certificate="$("$JAVA_HOME/bin/keytool" -printcert -jarfile "$OUT/$bundle" | awk '/SHA256:/ { print $2; exit }')"
  say "AAB: outputs/android/$bundle (upload key certificate SHA-256 $certificate)"
fi
if [ -n "$TRACK" ]; then
  say "Uploaded to Google Play, track $TRACK: https://play.google.com/console"
  tag="android-v$version"
  if git -C "$ROOT" rev-parse -q --verify "refs/tags/$tag" > /dev/null; then
    say "tag $tag already exists"
  elif git -C "$ROOT" tag "$tag" && git -C "$ROOT" push -q origin "refs/tags/$tag"; then
    say "tagged and pushed $tag"
  else
    say "WARNING: the tag $tag was not pushed (above); the upload is done"
  fi
fi
if [ "$LISTING" -eq 1 ]; then
  say "Store listing uploaded to Google Play."
fi
say "Download it from the site: Settings › Distill for Android."
