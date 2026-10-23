# Distill for Android: Privacy Policy

Last updated: 23 October 2026

## Who is responsible

Distill for Android (package `ch.bodhi.distill`) is an app developed by Cristian Zantedeschi, Switzerland, and
published on Google Play under the developer name “14 Bodhi”. Privacy contact:
[admin@14bodhi.com](mailto:admin@14bodhi.com).

Distill is an independent open-source project ([openpilot.distill](https://github.com/cristianku/openpilot.distill)).
It is not affiliated with, endorsed by or operated by comma.ai.

## In short

- The app is the remote control of **your own** distill-web server: it talks only to the address you enter.
- The developer runs no server for the app and receives none of the data described here.
- No analytics, advertising, crash-reporting or tracking libraries, and no account with the developer.

## Data the app handles

- **The address of your server.** Stored on the phone, in the app's private storage, with the choice to allow
  plain HTTP.
- **Your sign-in.** The name and password of the administrator of your site, typed in the app or on the site in
  the Website tab, are sent only to your server, to sign in. The app does not store them.
- **The session.** Your server answers a sign-in with a session cookie (HttpOnly, at most 30 days). The app keeps
  it in the cookie store of its web view on the phone and sends it only to your server, with each request.
- **What your server shows.** The state of the server (processor, graphics card, memory, disk), its trainings,
  evaluations and jobs, and the state of your comma device as your server sees it. The app reads them from your
  server and keeps them in memory while it runs; the pages of the site in the Website tab may stay in the web
  view's cache on the phone.
- **What you start.** A training, an evaluation, an installation on the comma or a stop is a request to your
  server, which carries it out.
- **The demo.** With the server address “demo”, everything (sign-in, data, jobs) is made up inside the app and
  nothing leaves the phone.

## Where the data goes

- **Your server only.** Requests go to the address you entered, over HTTPS, or over plain HTTP if you allow it
  (only advisable on your own network or a VPN; then the data travels unencrypted). What your server stores,
  logs or sends further (for example to Hugging Face or Weights & Biases, if you set that up on the site) is
  decided by you, its owner, and by the software installed on it.
- **Other websites.** Links of the site to other websites open in your browser and are governed by those
  websites.
- Like any app that goes online, each request reveals the phone's IP address to your server.

## What the app does not do

- It does not send data to the developer: there is no developer server.
- It does not include analytics, advertising, crash-reporting or tracking libraries.
- It does not create accounts: the administrator of the site exists on your server.
- It does not sell or share data.

## Permissions

Internet only, to reach your server.

## Storage, retention and deletion

- Android backup and device-to-device transfer are disabled for the app: its data is not copied to a Google
  backup or to a new phone.
- **Disconnect** (the gear in Overview) ends the session on your server and deletes the session and the site's
  data from the phone.
- Android Settings › Apps › Distill › Storage › Clear storage, or uninstalling the app, deletes everything the
  app kept on the phone.
- Data on your server stays there under your control; the site's own pages manage it.

## Children

The app is meant for adults who run their own training server; it is not directed at children.

## Changes and contact

If the app starts handling data differently, this page changes and so does the date at the top. Questions:
[admin@14bodhi.com](mailto:admin@14bodhi.com).
