# Windows

Slug: `windows` — package `src/integrations/windows_integration.py`

## Status

Always connected on this platform (win32). The agent's native, always-available
integration.

## Tools

- `windows_status` — reports platform availability and which controls are usable.
- `open_application` — launches a program from a hard-coded allow-list
  (browser paths and the like). Applications outside the known list are
  refused, and the list is exposed to the agent so it cannot guess.

## Media keys

Volume and media controls (play/pause/etc.) for Spotify and other music apps use
`win32api.keybd_event` VM_MEDIA_* codes and are reported through `media.py`.
These are surfaced by the Spotify integration's tools but are native Windows
capability.

## Tests

`tests/test_integrations_status.py` (`TestWindows`) covers the launch
allow-list, refusal of unknown applications, and media-control status.