# Local HTTP API

Start `node server.js` on Windows with the USB Link 2 connected. The service listens on IPv6 loopback. Use `http://localhost:5173` unless `AUTOCAMERA_PORT` was set. Each POST requires the matching `Origin` and `Host` headers. Keep the API on the camera host.

| POST route | JSON body | Success payload |
| --- | --- | --- |
| `/ptz/probe` | Empty | `{"ok":true}` |
| `/ptz/lens` | Empty | `{"ok":true,"zoom":{...},"focus":{...}}` |
| `/ptz/move` | Integers `pan`, `tilt` from −20 to 20 | `{"ok":true}` |
| `/ptz/zoom` | Nonzero integer `zoomSteps` from −100 to 100 | `{"ok":true,"zoom":{...}}` |

A lens range has integer `min`, `max`, `step`, and `value` fields, or is null. Zoom changes by `zoomSteps × step`, clamps to the driver range, then returns the driver readback. Focus readback does not change autofocus mode.

For gimbal movement, send both axes as zero to stop. The server also attempts an automatic stop about 420 ms after the last movement request completes. The CLI waits for the requested pulse duration and issues an explicit stop. The motion acknowledgment confirms the driver accepted the command.

HTTP 403 signals a mismatched host or origin. HTTP 404 signals an unknown route or method. HTTP 503 returns JSON with `ok: false` and an `error` message for invalid input or driver failures. If the camera is absent, connect it via USB and retry the relevant probe.
