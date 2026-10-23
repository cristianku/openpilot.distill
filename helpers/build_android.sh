#!/usr/bin/env bash
# Builds Distill for Android (app/android) on this server, without GitHub Actions: the unit tests, then
# the release APK in outputs/android/, which the site offers for download (Settings › Distill for Android).
# What is missing is set up first, so a fresh server needs nothing by hand (install.sh runs this script,
# the code update of the site runs it again when app/android changes):
#   - the tools, in /opt/android (or DISTILL_ANDROID_TOOLS): JDK 17, Android SDK platform 35 and
#     build-tools 35, Gradle 8.9 and its cache (about 1.5 GB; x86_64 only, as the SDK build tools)
#   - the signing key of this server, in ~/.config/distill-android (or DISTILL_ANDROID_KEYS):
#     distill-release.p12 (PKCS12, alias "distill", RSA 4096, ~27 years) and one ANDROID_*.txt file
#     per GitHub secret, to give the GitHub releases the same key. Never in git; keep a backup.
#
#   helpers/build_android.sh [--setup-only] [--help]
#     --setup-only   install the tools and make the key when missing, do not build
#
# Version: 0.3.<commits of HEAD>, versionCode 1000 + <commits>, as the GitHub release of the same
#   commit: the two install over each other when signed with the same key. "-dev" is added when
#   app/android has local changes.
# Output: outputs/android/distill-android-<version>.apk (the last 3 are kept) and
#   outputs/android/latest.json, what the site shows of the last build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOLS="${DISTILL_ANDROID_TOOLS:-/opt/android}"
KEYS="${DISTILL_ANDROID_KEYS:-$HOME/.config/distill-android}"
OUT="$ROOT/outputs/android"
JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
CMDLINE_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
GRADLE_URL="https://services.gradle.org/distributions/gradle-8.9-bin.zip"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SETUP_ONLY=0
case "${1:-}" in
  -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
  --setup-only) SETUP_ONLY=1 ;;
  '') ;;
  *) die "invalid option: $1 (see --help)" ;;
esac

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
  if [ ! -d "$TOOLS/sdk/platforms/android-35" ] || [ ! -d "$TOOLS/sdk/build-tools/35.0.0" ]; then
    say "installing Android SDK platform 35 and build-tools 35 (licenses accepted)"
    local sdkmanager="$TOOLS/sdk/cmdline-tools/latest/bin/sdkmanager"
    yes | JAVA_HOME="$TOOLS/jdk" "$sdkmanager" --sdk_root="$TOOLS/sdk" --licenses > "$work/licenses.log" 2>&1 || true
    JAVA_HOME="$TOOLS/jdk" "$sdkmanager" --sdk_root="$TOOLS/sdk" "platforms;android-35" "build-tools;35.0.0" \
      > "$work/sdk.log" 2>&1 || { tail -5 "$work/sdk.log" >&2; die "the Android SDK did not install"; }
  fi
  if [ ! -x "$TOOLS/gradle-8.9/bin/gradle" ]; then
    say "downloading Gradle 8.9 into $TOOLS/gradle-8.9"
    need curl unzip sha256sum
    curl -fsSL -o "$work/gradle.zip" "$GRADLE_URL"
    local expected
    expected="$(curl -fsSL "$GRADLE_URL.sha256")"
    [ "$(sha256sum "$work/gradle.zip" | cut -d' ' -f1)" = "$expected" ] || die "the Gradle download does not match its SHA-256"
    unzip -q "$work/gradle.zip" -d "$work"
    mv "$work/gradle-8.9" "$TOOLS/gradle-8.9"
  fi
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
    base64 -w0 "$KEYS/distill-release.p12" > "$KEYS/ANDROID_KEYSTORE_BASE64.txt"
    printf '%s' "$PW" > "$KEYS/ANDROID_KEYSTORE_PASSWORD.txt"
    printf '%s' "$PW" > "$KEYS/ANDROID_KEY_PASSWORD.txt"
    printf '%s' distill > "$KEYS/ANDROID_KEY_ALIAS.txt"
    cat > "$KEYS/README.txt" <<'EOF'
Signing key of Distill for Android on this server (PKCS12, alias "distill", RSA 4096, valid ~27 years),
made by helpers/build_android.sh. KEEP A BACKUP of distill-release.p12 and its password somewhere safe:
a lost key means every user must uninstall the app to get the next version. Never put these files in git.

To sign the GitHub releases with the same key: GitHub > the repository > Settings > Secrets and variables >
Actions > New repository secret, one secret per file, the name of the file without .txt as name, the
content of the file as value:
  ANDROID_KEYSTORE_BASE64   ANDROID_KEYSTORE_PASSWORD   ANDROID_KEY_ALIAS   ANDROID_KEY_PASSWORD
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

say "Distill for Android $version (versionCode $code, commit $commit), signed with the key of $KEYS"
(cd "$ROOT/app/android" && nice -n 10 "$TOOLS/gradle-8.9/bin/gradle" :app:testDebugUnitTest :app:assembleRelease \
  --no-daemon --console=plain -q -PdistillVersionCode="$code" -PdistillVersionName="$version")

apk="$ROOT/app/android/app/build/outputs/apk/release/app-release.apk"
[ -s "$apk" ] || die "the build made no APK"
mkdir -p "$OUT"
name="distill-android-$version.apk"
cp "$apk" "$OUT/$name.part" && mv "$OUT/$name.part" "$OUT/$name"
sha="$(sha256sum "$OUT/$name" | cut -d' ' -f1)"
# The Python of the project (install.sh made it); a system python3 otherwise.
PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" - "$OUT/latest.json" "$name" "$version" "$code" "$commit" release "$sha" <<'EOF'
import json, os, sys, time
path, name, version, code, commit, signed_with, sha = sys.argv[1:]
folder = os.path.dirname(path)
data = {'file': name, 'version': version, 'version_code': int(code), 'commit': commit, 'signed_with': signed_with,
        'sha256': sha, 'size': os.path.getsize(os.path.join(folder, name)), 'built': time.time()}
with open(path + '.part', 'w') as out:
    json.dump(data, out, indent=1)
os.replace(path + '.part', path)
EOF
# Keep the last 3 APKs.
ls -1t "$OUT"/distill-android-*.apk | tail -n +4 | xargs -r rm -f
say "APK: outputs/android/$name ($(awk -v b="$(stat -c %s "$OUT/$name")" 'BEGIN { printf "%.1f MB", b / 1e6 }'), sha256 $sha)"
say "Download it from the site: Settings › Distill for Android."
