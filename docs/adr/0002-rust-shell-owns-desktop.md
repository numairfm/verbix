# Rust owns the desktop shell; Python stays the voice service

The Python runtime works and is not being redesigned. Everything OS-facing
about a desktop app (tray, global shortcuts, autostart, single instance,
window, process supervision) is what Rust and Tauri are good at. The Rust
shell is the long-lived process and supervises the Python voice service as a
child; the seam between them is the existing JSONL IPC protocol, unchanged.

Python keeps the runtime, voice stack, tools, and the IPC server. The shell
forwards toggle and show commands, renders the panel, and restarts the child
on crash. Moving OS-facing tools (clipboard, media, app, open, system, notify,
screenshot) into Rust is deliberately deferred; it would add a second protocol
without changing the product.

## Consequences

Two processes per user. Quitting the shell stops the voice service. If OS
tools later move behind a Rust bridge, it is additive to this seam.
