use std::path::Path;

use tauri::AppHandle;
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut as HotKey, ShortcutState,
};

use crate::state::Shortcut;

/// The one accelerator of v1.
pub const ACCELERATOR: &str = "Ctrl+Alt+Space";

/// The display server the shell runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    /// X11, including XWayland.
    X11,
    /// A native Wayland session, where a global grab is not available.
    Wayland,
}

/// Which display server the shell runs on. GTK picks the Wayland backend
/// when the session offers one, unless `GDK_BACKEND` says otherwise.
pub fn session_kind(env: &dyn Fn(&str) -> Option<String>) -> SessionKind {
    match env("GDK_BACKEND").as_deref() {
        Some("wayland") => return SessionKind::Wayland,
        Some("x11") => return SessionKind::X11,
        _ => {}
    }
    match env("WAYLAND_DISPLAY") {
        Some(display) if !display.is_empty() => SessionKind::Wayland,
        _ => SessionKind::X11,
    }
}

/// The desktop-level command that toggles listening: the quoted path of the
/// running application, followed by `--toggle`.
pub fn fallback_command(executable: &Path) -> String {
    format!("{} --toggle", quote(executable))
}

/// Quote a path for a desktop shortcut command.
fn quote(path: &Path) -> String {
    let path = path.to_string_lossy();
    let mut quoted = String::with_capacity(path.len() + 2);
    quoted.push('"');
    for character in path.chars() {
        match character {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            '`' => quoted.push_str("\\`"),
            '$' => quoted.push_str("\\$"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            _ => quoted.push(character),
        }
    }
    quoted.push('"');
    quoted
}

/// The shortcut state the panel shows. A native grab replaces the fallback
/// command; a Wayland session or a refused grab keeps the command.
pub fn shortcut_state(session: SessionKind, registered: bool, executable: &Path) -> Shortcut {
    if session == SessionKind::Wayland || !registered {
        return Shortcut {
            accelerator: None,
            fallback: Some(fallback_command(executable)),
        };
    }
    Shortcut {
        accelerator: Some(ACCELERATOR.to_string()),
        fallback: None,
    }
}

/// Register the global shortcut. A refused grab is not an error: the
/// desktop-level fallback takes over.
pub fn register(app: &AppHandle) -> Shortcut {
    let executable = crate::platform::executable_path();
    let session = session_kind(&|name| std::env::var(name).ok());
    if session == SessionKind::Wayland {
        return shortcut_state(session, false, &executable);
    }
    let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
    let registered = app.global_shortcut().register(hotkey).is_ok();
    shortcut_state(session, registered, &executable)
}

/// Handle a shortcut press: the same toggle as the tray.
pub fn on_pressed(app: &AppHandle) {
    crate::panel::request_toggle(app);
}

/// True when the event is a key press, not a key release.
pub fn is_pressed(state: ShortcutState) -> bool {
    state == ShortcutState::Pressed
}
