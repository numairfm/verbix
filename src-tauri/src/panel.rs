use tauri::{
    App, AppHandle, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent,
};

use crate::service::{Command, Supervisor};
use crate::state::Snapshot;

pub const PANEL_LABEL: &str = "main";

#[tauri::command]
pub fn get_state(state: tauri::State<'_, crate::ShellState>) -> Snapshot {
    match state.snapshot.lock() {
        Ok(snapshot) => snapshot.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

#[tauri::command]
pub fn hide_panel(app: AppHandle) {
    hide(&app);
}

#[tauri::command]
pub fn toggle(app: AppHandle) {
    request_toggle(&app);
}

/// One toggle entry point for the panel, tray, shortcut, and forwarded launch.
/// A toggle is never queued while the service is unavailable: the panel opens
/// with the current state instead.
pub fn request_toggle(app: &AppHandle) {
    let snapshot = app
        .try_state::<crate::ShellState>()
        .and_then(|state| state.snapshot.lock().ok().map(|snapshot| snapshot.clone()));
    if snapshot
        .as_ref()
        .map(|snapshot| snapshot.phase != crate::state::Phase::Connected)
        .unwrap_or(true)
    {
        show(app);
        return;
    }
    if let Some(supervisor) = app.try_state::<Supervisor>() {
        supervisor.command(Command::Toggle);
    }
}

#[tauri::command]
pub fn restart_service(app: AppHandle) {
    if let Some(supervisor) = app.try_state::<Supervisor>() {
        supervisor.command(Command::Restart);
    }
}

pub fn create(app: &App) -> tauri::Result<WebviewWindow> {
    let window = WebviewWindowBuilder::new(app, PANEL_LABEL, WebviewUrl::App("index.html".into()))
        .title("Verbix")
        .inner_size(420.0, 560.0)
        .min_inner_size(360.0, 420.0)
        .resizable(true)
        .visible(false)
        .skip_taskbar(true)
        .build()?;

    let handle = window.clone();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            let _ = handle.hide();
        }
    });
    Ok(window)
}

pub fn show(app: &AppHandle) {
    crate::app_state::refresh_shortcut(app);
    if let Some(window) = app.get_webview_window(PANEL_LABEL) {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

pub fn hide(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(PANEL_LABEL) {
        let _ = window.hide();
    }
}
