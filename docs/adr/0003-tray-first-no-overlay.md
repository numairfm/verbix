# Tray-first surface, no overlay in v1

The README proposes a GPUI click-through overlay. Overlay behavior is
compositor-specific (GNOME has no layer-shell), steals focus on some
compositors, and Handy disables its own overlay by default on Linux for that
reason. v1 ships a tray icon plus an on-demand panel instead: the tray is the
glanceable state, the panel holds captions and tasks, and voice stays the
primary interface.

## Consequences

No always-on-top state while working in other applications; the tray icon
carries state at a glance. An overlay can be revisited after v1 if a real
need appears.
