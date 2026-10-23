# Distill for Android on Google Play

How the app gets to Google Play, all from the server: `helpers/build_android.sh` builds, tests, signs and
publishes by itself; nothing is built by GitHub. The general procedure (service account, internal and closed tests, the
12 testers for 14 days) is the one of nav2pilot, `docs/GOOGLE_PLAY_SETUP.md` in
[cristianku/nav2pilot](https://github.com/cristianku/nav2pilot); below is what is specific to Distill.

| | |
| --- | --- |
| Package | `ch.bodhi.distill` (permanent on Google Play) |
| Version | `0.3.<commits of master>`, versionCode `1000 + <commits>`: every commit has its own, always higher |
| Upload key | the key of the server, `~/.config/distill-android/distill-release.p12` (certificate SHA-256 `9F:B4:E3:74:…:BA:0F`) |
| App signing key | generated and kept by Google (Play App Signing, the default) |
| Service account key | `~/.config/distill-android/play-service-account.json` (never in git) |
| Store listing | `app/src/main/play/` (texts, icon, feature graphic, release notes) |
| Privacy policy | [`docs/android-privacy.md`](../../docs/android-privacy.md) |

## 1. Once: the service account

The account of the Google Cloud project that nav2pilot already uses (`cristianku-play`) works for every app of
the developer account.

1. Copy its JSON key to the server: `~/.config/distill-android/play-service-account.json`, then
   `chmod 600` it. (A new key: Google Cloud › IAM › Service accounts › Keys › Add key › JSON.)
2. When the app exists (step 2): Play Console › Users and permissions › the service account › App permissions ›
   add Distill with "Release to testing tracks", "Manage testing tracks" and, for `--listing`, "Manage store
   presence" (or Admin).

## 2. Create the app and upload the first bundle by hand

The API cannot make the first release of an app.

1. Play Console › Create app: name `Openpilot Distill`, default language English (United States),
   App, Free.
2. On the site: Settings › Distill for Android › Bundle for Google Play (the AAB of the last build,
   `outputs/android/distill-android-0.3.<N>.aab`; `helpers/build_android.sh` prints the certificate of its
   upload key).
3. Test and release › Testing › Internal testing › Create new release. At the first upload Play asks how to sign
   the app: keep **Google-generated key** (Play App Signing). The key of the server becomes the upload key: Play
   then shows its certificate under App integrity, the one the script printed.
4. Upload the AAB, Next, Save, and add a list of testers (the email of the Google account of each phone).

Why a Google key: an app from Google Play cannot then update an APK downloaded from the site of a server, or
the other way round (each server signs with its own key). Remove the one from the site once, before installing
from Google Play.

## 3. Every release after that

```bash
helpers/build_android.sh                   # build, then the internal track (no review) by itself
helpers/build_android.sh --play=alpha      # closed testing: promotes the version already on internal
helpers/build_android.sh --play=production
helpers/build_android.sh --listing         # also the store listing of app/src/main/play
```

Every build publishes: the tests, the APK and AAB, then the AAB with its R8 mapping and the release notes of
`app/src/main/play/release-notes/en-US/default.txt` on the internal track, and the tag `android-v0.3.<N>` pushed
to GitHub. It does so only for a version without local changes, whose commit is on GitHub and that is not on
Google Play yet (the tag); otherwise it says why not, and a failed upload is a warning (the site still gets its
APK). The code update of the site runs it when `app/android` changes. With `--play=TRACK` every problem is an
error, and a version already uploaded is promoted from the internal track. `--listing` replaces texts and
pictures edited in Play Console: edit them in the repository.

| Error of the upload | Cause | Remedy |
| --- | --- | --- |
| `403 The caller does not have permission` | the service account has no permission on Distill | step 1.2 |
| the same 403 on `--listing` only | no "Manage store presence" | step 1.2 |
| `404 Package not found` | no release yet | the first upload by hand, step 2 |
| `version code … has already been used` | that version is on Google Play | a new commit |

## 4. App content (Play Console › Policy › App content)

| Item | Answer |
| --- | --- |
| Privacy policy | `https://github.com/cristianku/openpilot.distill/blob/master/docs/android-privacy.md` (or, with GitHub Pages on `master` `/docs`, `https://cristianku.github.io/openpilot.distill/android-privacy`) |
| App access | "All or some functionality is restricted", with the instructions below |
| Ads | No ads |
| Content rating | Questionnaire: "All other app types", every answer No |
| Target audience | 18 and over |
| Data safety | below |
| Government, financial features, health | No |
| Category, contact | App › Auto & Vehicles; `admin@14bodhi.com`, website `https://github.com/cristianku/openpilot.distill` |

**App access** (instructions in English, at most 500 characters; the credentials grant full access):

> Distill controls the user's own self-hosted Distill server. To review it without one, use the demo server
> built into the app: type demo as Server address and tap Connect, then sign in with name demo, password demo.
> Overview shows the server; Train, Evaluate and Comma start simulated jobs that appear in Overview with their
> progress. Only the Website tab needs a real server. Demo data never leaves the phone.

Name `demo`, password `demo`.

**Data safety**: nothing reaches the developer, but Google counts as collected what leaves the phone, even
towards the user's own server. The answers that match the privacy policy:

- Collects data: Yes. Shares data: No.
- Personal info › User IDs: the name of the administrator, sent to the user's server at sign-in. Collected, not
  processed ephemerally, required, for App functionality and Account management.
- Encrypted in transit: No (the user can allow plain HTTP for a server on their own network).
- Accounts: the app does not create them (the administrator is made on the user's server).
- Deletion: Disconnect in the app ends the session and erases what the app kept; the data on the server is the
  user's own.

## 5. Store listing

`app/src/main/play/listings/en-US/`: title (30 characters at most), short description (80), full description
(4000), the 512×512 icon and the 1024×500 feature graphic. Phone screenshots (2 to 8, 9:16, at least 1080 px
for the promotions) go into `graphics/phone-screenshots/` before `--listing`, or straight into Play Console: the
demo server of the app gives clean ones.

## 6. Closed test and production

A new personal developer account reaches production only after a closed test with at least 12 testers for 14
days in a row (nav2pilot guide, section 6): `--play=alpha`, testers and countries in Closed testing › Alpha, then
"Apply for production".

## Keys: keep a backup

Back up `~/.config/distill-android/` (the key, its password, the service account key) outside the server. A lost
upload key can be replaced through Play Console (App integrity › Request upload key reset); the APKs that the site
of this server offers would then be signed with a new key, and the phones that installed one from there would have
to uninstall it once.
