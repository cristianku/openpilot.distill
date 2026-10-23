#!/usr/bin/env bash
# Builds Distill for Android (app/android) on this server, without GitHub Actions: the unit tests, then
# the release APK in outputs/android/, which the site offers for download (Settings › Distill for Android).
#
#   helpers/build_android.sh [--help]
#
# Tools: JDK 17, Android SDK 35 (platform and build-tools) and Gradle 8.9 in /opt/android, or in
#   DISTILL_ANDROID_TOOLS. The Gradle cache stays there too, so a second build takes about a minute.
# Signing: with the key in ~/.config/distill-android (distill-release.p12 and its ANDROID_*.txt files,
#   the key of the GitHub releases) when it is there, else with the debug key of this server.
# Version: 0.3.<commits of HEAD>, versionCode 1000 + <commits>, as the GitHub release of the same
#   commit: the two install over each other. "-dev" is added when app/android has local changes.
# Output: outputs/android/distill-android-<version>.apk (the last 3 are kept) and
#   outputs/android/latest.json, what the site shows of the last build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOLS="${DISTILL_ANDROID_TOOLS:-/opt/android}"
KEYS="$HOME/.config/distill-android"
OUT="$ROOT/outputs/android"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
  '') ;;
  *) die "invalid option: $1 (see --help)" ;;
esac

[ -x "$TOOLS/jdk/bin/java" ] || die "no JDK 17 in $TOOLS/jdk"
[ -x "$TOOLS/gradle-8.9/bin/gradle" ] || die "no Gradle 8.9 in $TOOLS/gradle-8.9"
[ -d "$TOOLS/sdk/platforms/android-35" ] || die "no Android SDK platform 35 in $TOOLS/sdk"
export JAVA_HOME="$TOOLS/jdk" ANDROID_HOME="$TOOLS/sdk" GRADLE_USER_HOME="$TOOLS/gradle-home"

count="$(git -C "$ROOT" rev-list --count HEAD)"
commit="$(git -C "$ROOT" rev-parse --short HEAD)"
version="0.3.$count"
if [ -n "$(git -C "$ROOT" status --porcelain -- app/android)" ]; then
  version="$version-dev"
fi
code=$((1000 + count))

signed_with=debug
if [ -f "$KEYS/distill-release.p12" ]; then
  export DISTILL_KEYSTORE="$KEYS/distill-release.p12"
  DISTILL_KEYSTORE_PASSWORD="$(cat "$KEYS/ANDROID_KEYSTORE_PASSWORD.txt")"
  DISTILL_KEY_ALIAS="$(cat "$KEYS/ANDROID_KEY_ALIAS.txt")"
  DISTILL_KEY_PASSWORD="$(cat "$KEYS/ANDROID_KEY_PASSWORD.txt")"
  export DISTILL_KEYSTORE_PASSWORD DISTILL_KEY_ALIAS DISTILL_KEY_PASSWORD
  signed_with=release
fi

say "Distill for Android $version (versionCode $code, commit $commit), signed with the $signed_with key"
(cd "$ROOT/app/android" && nice -n 10 "$TOOLS/gradle-8.9/bin/gradle" :app:testDebugUnitTest :app:assembleRelease \
  --no-daemon --console=plain -q -PdistillVersionCode="$code" -PdistillVersionName="$version")

apk="$ROOT/app/android/app/build/outputs/apk/release/app-release.apk"
[ -s "$apk" ] || die "the build made no APK"
mkdir -p "$OUT"
name="distill-android-$version.apk"
cp "$apk" "$OUT/$name.part" && mv "$OUT/$name.part" "$OUT/$name"
sha="$(sha256sum "$OUT/$name" | cut -d' ' -f1)"
python3 - "$OUT/latest.json" "$name" "$version" "$code" "$commit" "$signed_with" "$sha" <<'EOF'
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
