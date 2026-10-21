use std::path::Path;

use tauri::{
    App, AppHandle, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent,
};

use crate::app_state::{self, ShellState};
use crate::service::{Command, Supervisor};
use crate::setup;
use crate::state::{DownloadState, Snapshot};

pub const PANEL_LABEL: &str = "main";

#[tauri::command]
pub fn get_state(app: AppHandle) -> Snapshot {
    refresh_autostart(&app);
    app_state::snapshot_of(&app)
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
    let snapshot = app_state::snapshot_of(app);
    if snapshot.setup.required || snapshot.phase != crate::state::Phase::Connected {
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

/// Write the OpenRouter key. The key is never echoed back to the panel.
#[tauri::command]
pub fn save_provider_key(app: AppHandle, key: String) -> Result<Snapshot, String> {
    let state = app
        .try_state::<ShellState>()
        .ok_or_else(|| "The shell state is not ready.".to_string())?;
    let home = state.setup.home.clone();
    setup::write_provider_key(&home, &key)?;
    refresh_setup(&app, true);
    Ok(app_state::snapshot_of(&app))
}

/// Start the bundled model downloader.
#[tauri::command]
pub fn start_model_download(app: AppHandle) -> Result<Snapshot, String> {
    let state = app
        .try_state::<ShellState>()
        .ok_or_else(|| "The shell state is not ready.".to_string())?;
    let home = state.setup.home.clone();
    let resources = state.setup.resources.clone();
    let detection = app_state::detect_setup(&home);

    if detection.model.authoritative && !detection.model.ready {
        return Err(detection
            .model
            .error
            .unwrap_or_else(|| "The configured speech model is missing.".to_string()));
    }
    if detection.model.ready && !detection.model.downloadable {
        return Err(
            "The speech model comes from a configured path. The downloader only manages the voice root."
                .to_string(),
        );
    }
    if state
        .setup
        .download
        .lock()
        .map(|slot| slot.is_some())
        .unwrap_or(false)
    {
        return Err("The model download is already running.".to_string());
    }

    let job = crate::download::DownloadJob {
        program: resources.join("voice/python/bin/python3"),
        args: vec![
            resources
                .join("voice/fetch_voice_models.py")
                .display()
                .to_string(),
            "--root".to_string(),
            setup::voice_dir(&home).display().to_string(),
        ],
        work_dir: setup::work_dir(&home),
        env: app_state::child_env(&resources, &home),
    };
    let handle = crate::download::start(&job)
        .map_err(|error| format!("Cannot start the model download: {error}"))?;
    if let Ok(mut slot) = state.setup.download.lock() {
        *slot = Some(handle.clone());
    }
    if let Ok(mut error) = state.setup.download_error.lock() {
        *error = None;
    }
    refresh_setup(&app, true);

    let app_handle = app.clone();
    let watcher = handle.clone();
    std::thread::Builder::new()
        .name("model-download-events".to_string())
        .spawn(move || follow_download(app_handle, watcher))
        .ok();
    Ok(app_state::snapshot_of(&app))
}

/// Write or remove the autostart entry.
#[tauri::command]
pub fn set_autostart(app: AppHandle, enabled: bool) -> Result<Snapshot, String> {
    let state = app
        .try_state::<ShellState>()
        .ok_or_else(|| "The shell state is not ready.".to_string())?;
    let home = state.setup.home.clone();
    crate::autostart::set_enabled(&home, enabled, &crate::autostart::executable_path())
        .map_err(|error| error.to_string())?;
    app_state::update(&app, |snapshot| snapshot.setup.autostart = enabled);
    Ok(app_state::snapshot_of(&app))
}

/// Write the setup marker, start the service, and switch to the main view.
#[tauri::command]
pub fn finish_setup(app: AppHandle) -> Result<Snapshot, String> {
    let state = app
        .try_state::<ShellState>()
        .ok_or_else(|| "The shell state is not ready.".to_string())?;
    let home = state.setup.home.clone();
    let detection = app_state::detect_setup(&home);
    if !detection.key.ready {
        return Err("The provider key is not ready.".to_string());
    }
    if !detection.model.ready {
        return Err("The speech model is not ready.".to_string());
    }
    setup::write_marker(&home)?;

    let running = state
        .supervisor
        .lock()
        .map(|slot| slot.is_some())
        .unwrap_or(false);
    refresh_setup(&app, false);
    if !running {
        app_state::start_service(&app, false)?;
    }
    Ok(app_state::snapshot_of(&app))
}

/// Open the setup surface for repair, with the current steps pre-marked.
pub fn show_setup(app: &AppHandle) {
    refresh_setup(app, true);
    show(app);
}

/// Re-inspect the setup surface and write the result into the snapshot.
fn refresh_setup(app: &AppHandle, required: bool) {
    let Some(state) = app.try_state::<ShellState>() else {
        return;
    };
    let home = state.setup.home.clone();
    let detection = app_state::detect_setup(&home);
    let download = app_state::snapshot_of(app).setup.download.clone();
    let download_error = state
        .setup
        .download_error
        .lock()
        .ok()
        .and_then(|error| error.clone());
    let autostart = crate::autostart::status(&home) == crate::autostart::Status::Enabled;
    app_state::update(app, |snapshot| {
        snapshot.setup = app_state::setup_fields(&detection, download, download_error);
        snapshot.setup.autostart = autostart;
        snapshot.setup.required = required;
    });
}

fn refresh_autostart(app: &AppHandle) {
    let Some(state) = app.try_state::<ShellState>() else {
        return;
    };
    let home = state.setup.home.clone();
    let enabled = crate::autostart::status(&home) == crate::autostart::Status::Enabled;
    app_state::update(app, |snapshot| snapshot.setup.autostart = enabled);
}

/// Follow the download: report the last line, then re-check the model.
fn follow_download(app: AppHandle, handle: crate::download::DownloadHandle) {
    let mut last_stderr = String::new();
    while let Some(event) = handle.recv() {
        match event {
            crate::download::DownloadEvent::Progress(line) => {
                app_state::update(&app, |snapshot| {
                    snapshot.setup.download = Some(DownloadState {
                        running: true,
                        last_line: line,
                    });
                });
            }
            crate::download::DownloadEvent::Stderr(line) => {
                last_stderr = line;
            }
            crate::download::DownloadEvent::Finished { success } => {
                if let Some(state) = app.try_state::<ShellState>() {
                    if let Ok(mut slot) = state.setup.download.lock() {
                        *slot = None;
                    }
                    if let Ok(mut error) = state.setup.download_error.lock() {
                        *error = if success {
                            None
                        } else {
                            Some(if last_stderr.trim().is_empty() {
                                "The model download failed.".to_string()
                            } else {
                                last_stderr.trim().to_string()
                            })
                        };
                    }
                }
                refresh_setup(&app, true);
                break;
            }
        }
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

/// The path of the credential file, for the setup surface.
pub fn auth_path(home: &Path) -> std::path::PathBuf {
    setup::auth_path(home)
}
