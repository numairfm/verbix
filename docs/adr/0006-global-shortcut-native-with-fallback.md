# Global shortcut: native grab first, desktop-level fallback

The shell registers the toggle shortcut with the platform global-shortcut
crate: X11 on Linux (including XWayland), native on Windows and macOS. On
Wayland the OS may refuse or ignore the grab, so the supported fallback is a
desktop-level shortcut bound to the shell's toggle command, shown by first-run
setup. This is how Handy behaves in practice and keeps the surface small.

## Consequences

On GNOME and KDE Wayland, a one-time desktop shortcut binding may be required
and the app cannot promise a zero-interaction hotkey. The XDG GlobalShortcuts
portal is not used in v1; it can replace the fallback later without changing
the product surface.
