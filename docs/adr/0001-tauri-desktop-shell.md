# Use Tauri 2 for the v1 desktop shell

We need a cross-platform shell (tray, global shortcuts, autostart, single
instance, bundling) around a small status panel. GPUI is a renderer only and
pre-1.0; assembling the shell and a bundler around it is a second project.
Tauri 2 ships the shell and per-OS bundles, and the v1 panel is a small static
HTML surface, not a web app.

## Considered Options

- **GPUI**: native GPU renderer, no webview; no tray, shortcuts, autostart,
  or bundler; pre-1.0 and moved by Zed.
- **egui**: stable and native; still needs the whole shell hand-assembled.
- **Browser page served by the voice service**: cheapest; no tray, no
  always-on-top, not a desktop app.

## Consequences

Linux gains a WebKitGTK runtime dependency; Windows and macOS use their system
webviews. GPUI is deferred, not rejected: the Rust/Python IPC seam lets the
panel be replaced by a GPUI process without touching Python.
