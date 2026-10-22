# iOS — Distill

Requirements: macOS, Xcode 16+, [XcodeGen](https://github.com/yonaskolb/XcodeGen), iOS 17+.

```sh
cd app/ios
xcodegen generate
open Distill.xcodeproj
```

Select your signing team in Xcode and deploy to a phone on the same private network as the server.

The app starts on a connection form. Token goes to Keychain (AfterFirstUnlockThisDeviceOnly); the URL and HTTP warning choice are app preferences. Dashboard refreshes every 10 seconds while visible and uses `X-Distill-Token`, not a cookie. The Website tab establishes the server's existing HttpOnly session cookie in a nonpersistent WKWebView data store.

Current iOS project settings allow HTTP because local Distill deployments are HTTP-only; the app requires an explicit insecure-HTTP opt-in. Configure trusted HTTPS before distributing through an app store, then remove the broad ATS exception.
