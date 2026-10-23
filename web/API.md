# The API of distill-web

distill-web (`web/server.py`) is the site of a Distill server and the API that **Distill for Android**
(`app/android`) works with: the app knows only the address of the server, signs in there and does everything
through the requests below, the same ones the site sends. The app is published on Google Play and updates on its
own: a phone can have an app newer or older than the server it talks to. Keep this contract stable.

## Rules

- JSON over HTTP(S), on the address and port of the site (default 8766), behind a reverse proxy for HTTPS.
- **Session**: `POST /api/auth/login` answers with a cookie `distill_web_<port>` (HttpOnly, SameSite=Strict,
  Secure behind HTTPS, 7 days without use, 30 days in all). Every other call sends it back in `Cookie`. The app
  keeps it in the cookie store of its WebView, so the Website tab is signed in too.
- **POST**: `Content-Type: application/json` with `Content-Length`, at most 64 KiB, a JSON object (`{}` when there
  is nothing to send). A request with an `Origin` or `Sec-Fetch-Site` of another site is refused (403); the app
  sends neither, as a request of the same origin.
- **Errors**: any status but 200 has `{"ok": false, "error": "<reason>", "message": "<for people>"}`. The app shows
  `message`. 401 is "sign in" (`login_required`, or a wrong name or password), 429 too many attempts, 403 another
  origin or host, 409 not possible now (the reason says why), 400 a wrong parameter.
- **Version**: `GET /api/auth/state` has `"api": 1` (`API_VERSION` in `web/server.py`, `DistillApi.API_VERSION` in
  the app). A new field or endpoint keeps the number; removing or renaming a field the app reads, or changing what
  it means, needs a new number and an app that knows it. A server without `api` is version 1. An app that finds a
  higher number than its own asks to be updated.

## What the app uses

| Request | Session | Answer (what the app reads) |
| --- | --- | --- |
| `GET /api/auth/state` | no | `configured`, `signed_in`, `api`; signed in also `user`, `sessions` |
| `POST /api/auth/login` `{"user", "password"}` | no | `{"ok": true}` and the session cookie |
| `POST /api/auth/logout` `{}` | yes | ends this session; the cookie expires |
| `GET /api/status` | yes | `system` (`cpu.total`, `gpu.utilization`, `gpu.name`, `memory.used_percent`, `disk.free_gb`, `disk.total_gb`, `gpus[]` with `index`, `name`, `memory_total_mb`), `training[]` (`name`, `percent`, `steps_done`, `steps_total`), `jobs[]` (`id`, `label`, `summary`, `state`, `progress` with `percent`, `done`, `total`, `stoppable`), `comma` (`reachable`, `onroad`, `bundle`, `bundle_default`, `backup`, `host`, `error`, `checked`), `actions.<id>` (`allowed`, `reason`), `code` |
| `GET /api/runs` | yes | `runs[]`: `name`, `status`, `steps_done`, `steps_total`, `has_pt`, `has_training_pt`, `resumable`, `started`, `avg_step_time`, `config` (`teacher`, `batch_size`, `workers`), `last.loss` |
| `GET /api/eval` | yes | `results[]`: `model`, `avg_miss_m`, `selection`, `rate_hz`, `date` |
| `GET /api/hf` | yes | `ok`, `stable`, `models[]`, `error` |
| `POST /api/run/train` | yes | `{"ok": true, "job": {…, "label"}}`. Body: `mode` (`fresh`, `continue`, `resume`), `steps` and `teacher` or `run` (resume), `batch_size`, `workers`, `wandb`, `gpus` |
| `POST /api/run/eval` `{"model"}` | yes | the job; `model` is `stock`, `teacher`, `logged`, `run:<name>` or `hf:<name>` |
| `POST /api/run/install` `{"run", "confirm": "INSTALL"}` | yes | the job; the comma must be reachable, offroad, on the Default model |
| `POST /api/run/restore` `{"confirm": "RESTORE"}` | yes | the job |
| `POST /api/jobs/<id>/stop` `{}` | yes | the job, stopping |
| `POST /api/comma/check` `{}` | yes | `comma`, asked now |

The site uses more (`/api/settings…`, `/api/jobs/<id>/log`, `/api/drives`, `/api/compare`, the other actions);
the app reaches those through the site in its Website tab.

## Keeping the two in step

- `tests/test_web_server.py` `AppApiTest` signs in, reads, starts, stops and signs out as the app does, and checks
  the fields above.
- `app/android/app/src/test/…/ApiContractTest.kt` reads `status.json` and `runs.json`, produced by
  `web/server.py`: regenerate them when those answers change.
- The demo server of the app (`DemoServer.kt`, server address `demo`) answers the same API with made-up data;
  give it a new field the app reads.
