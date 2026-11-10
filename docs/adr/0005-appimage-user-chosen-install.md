# AppImage, with the install method left to the user

Linux v1 ships a Tauri AppImage that runs from wherever the user puts it. We
do not copy it to a stable path, install a systemd service, or ship a
package; the user chooses to run it in place or manage it with a tool such as
Gearlever. Autostart points at the running AppImage, and models and
configuration live in the global state folder.

## Consequences

Autostart breaks if the user moves the file; re-enabling it re-points the
entry. deb, rpm, Flatpak, and store distribution are out of v1 and can be
added later from the same shell.
