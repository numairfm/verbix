"""Spike script investigating the XDG GlobalShortcuts portal on KDE/Wayland.

Interface: org.freedesktop.portal.GlobalShortcuts
Object Path: /org/freedesktop/portal/desktop

Findings:
1. CreateSession creates an asynchronous Request object (/org/freedesktop/portal/desktop/request/...).
2. The client must listen on DBus for the Response signal from the portal backend (KDE KWin portal).
3. BindShortcuts then prompts the user with an interactive GUI dialog (KDE Plasma Shortcut dialog)
   asking for permission to register the shortcut.
4. If approved, the portal emits Activated and Deactivated signals.
5. If the CLI or script terminates, the DBus connection drops and all shortcuts are unbound.

Verdict for CLI:
XDG GlobalShortcuts portal is heavy for a pure CLI / headless agent because it requires:
- An active GUI desktop session with interactive user confirmation modal.
- A permanent DBus mainloop (GMainLoop or GLib).
- Fails over SSH, headless daemons, or terminal-only environments.

Stand-in PTT recommendation for CLI:
1. Terminal Interactive PTT: Press Enter / Space in the CLI to speak (zero GUI, instant, works over SSH).
2. Native KDE Custom Shortcut via IPC: In KDE Plasma 6 System Settings -> Shortcuts, bind any global key
   (e.g. Meta+Space) to `verbix --ptt toggle` or `verbix --ptt down` / `verbix --ptt up`.
   The Verbix daemon listens on ~/.local/share/verbix/verbix.sock and triggers the audio pipeline directly.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    print("Spiking XDG GlobalShortcuts portal via gdbus...")
    cmd = [
        "gdbus",
        "call",
        "--session",
        "--dest",
        "org.freedesktop.portal.Desktop",
        "--object-path",
        "/org/freedesktop/portal/desktop",
        "--method",
        "org.freedesktop.portal.GlobalShortcuts.CreateSession",
        "{'session_handle_token': <'verbix_spike'>}",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("CreateSession response:", res.stdout.strip())
        print(
            "Result: Portal interface exists. However, binding requires interactive GUI prompt"
            " and a long-lived DBus event loop."
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("XDG GlobalShortcuts portal unavailable:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
