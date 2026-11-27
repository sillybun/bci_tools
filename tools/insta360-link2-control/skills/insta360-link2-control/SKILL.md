---
name: insta360-link2-control
description: Operate an Insta360 Link 2 USB camera on a Windows host. Use for capability detection, bounded pan and tilt, stopping the gimbal, zoom readback and adjustment, or opening the preview for recording.
---

# Insta360 Link 2 Control

Run commands in the Insta360 Link 2 Control project directory on the Windows computer connected to the camera. The same files may be installed under `tools/insta360-link2-control` in `bci_tools`.

Start the local service with `node server.js` if it is down. It defaults to `http://localhost:5173`. Set `AUTOCAMERA_PORT` to the same value in the server and client environments when using a different port. Read [the API reference](references/api.md) when making direct HTTP calls or diagnosing an error.

| Task | Command |
| --- | --- |
| Probe gimbal control | `node link2.js probe` |
| Read zoom and manual focus ranges | `node link2.js lens` |
| Pan at speed 5 for 130 ms, then stop | `node link2.js move 5 0 130` |
| Stop the gimbal | `node link2.js stop` |
| Increase zoom by 10 hardware steps | `node link2.js zoom 10` |

Probe the relevant capability before the first adjustment. A null range indicates that the Windows driver did not expose a usable manual range. Zoom and focus are separate. The focus command surface reports capability and current value. Verify zoom using the returned `zoom.value`. On the tested Link 2, 100 means 1× and 400 means 4×.

Use short pulses for framing. Pan and tilt are relative speeds in the range −20 to 20. Match the direction to the live image after a small pulse. For a relative command whose outcome is uncertain, send `stop` for motion or read the lens state for zoom before issuing another adjustment.

Open the local console for video preview or recording, connect the Link 2, and use the recording button. Browser camera permission and recording belong to the console. Files download to the Windows computer.
