use std::path::Path;

use verbix_shell::shortcut::{
    fallback_command, session_kind, shortcut_state, SessionKind, ACCELERATOR,
};

fn env(pairs: Vec<(&'static str, &'static str)>) -> impl Fn(&str) -> Option<String> {
    move |name: &str| {
        pairs
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    }
}

#[test]
fn the_accelerator_is_fixed() {
    assert_eq!(ACCELERATOR, "Ctrl+Alt+Space");
}

#[test]
fn the_fallback_command_quotes_the_path_and_asks_for_a_toggle() {
    let command = fallback_command(Path::new("/home/user/Verbix.AppImage"));
    assert_eq!(command, "\"/home/user/Verbix.AppImage\" --toggle");
}

#[test]
fn the_fallback_command_keeps_a_path_with_spaces_usable() {
    let command = fallback_command(Path::new("/home/user/My Apps/Verbix 0.1.0.AppImage"));
    assert_eq!(
        command,
        "\"/home/user/My Apps/Verbix 0.1.0.AppImage\" --toggle"
    );
}

#[test]
fn the_fallback_command_escapes_the_shell_reserved_characters() {
    let command = fallback_command(Path::new("/home/user/a\"b$c`d\\e.AppImage"));
    assert_eq!(
        command,
        "\"/home/user/a\\\"b\\$c\\`d\\\\e.AppImage\" --toggle"
    );
}

#[test]
fn a_wayland_session_uses_the_fallback_even_when_the_grab_succeeds() {
    let wayland = env(vec![("WAYLAND_DISPLAY", "wayland-0")]);
    assert_eq!(session_kind(&wayland), SessionKind::Wayland);

    let state = shortcut_state(
        SessionKind::Wayland,
        true,
        Path::new("/tmp/Verbix.AppImage"),
    );
    assert_eq!(state.accelerator, None);
    assert_eq!(
        state.fallback.as_deref(),
        Some("\"/tmp/Verbix.AppImage\" --toggle")
    );
}

#[test]
fn an_x11_session_uses_the_native_grab() {
    let x11 = env(vec![("DISPLAY", ":0")]);
    assert_eq!(session_kind(&x11), SessionKind::X11);

    let state = shortcut_state(SessionKind::X11, true, Path::new("/tmp/Verbix.AppImage"));
    assert_eq!(state.accelerator.as_deref(), Some("Ctrl+Alt+Space"));
    assert_eq!(state.fallback, None);
}

#[test]
fn a_refused_grab_falls_back_to_the_command() {
    let state = shortcut_state(SessionKind::X11, false, Path::new("/tmp/Verbix.AppImage"));
    assert_eq!(state.accelerator, None);
    assert_eq!(
        state.fallback.as_deref(),
        Some("\"/tmp/Verbix.AppImage\" --toggle")
    );
}

#[test]
fn an_explicit_x11_backend_wins_over_the_wayland_session() {
    let xwayland = env(vec![
        ("WAYLAND_DISPLAY", "wayland-0"),
        ("GDK_BACKEND", "x11"),
    ]);
    assert_eq!(session_kind(&xwayland), SessionKind::X11);

    let wayland = env(vec![
        ("WAYLAND_DISPLAY", "wayland-0"),
        ("GDK_BACKEND", "wayland"),
    ]);
    assert_eq!(session_kind(&wayland), SessionKind::Wayland);
}
