# Linux ships first; other platforms are seams

v1 targets Linux x86_64 on GNOME and KDE, Wayland and X11, because that is
what the maintainer can test. The destination is a universal application: one
codebase that produces a per-OS bundle. macOS and Windows live behind one
platform interface in the shell and are compiled in CI, but no artifacts ship
until someone can test them.

## Consequences

Platform-specific behavior (shortcuts, window style, autostart) stays behind
one Rust interface per concern; shared shell code and the Python side must not
assume Linux.
