use std::path::{Path, PathBuf};

use tauri::{
    image::Image,
    menu::{CheckMenuItem, Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, AppHandle, Manager, Wry,
};

use crate::autostart::{self, Status};
use crate::panel;
use crate::service::{Command, Supervisor};
use crate::state::{Phase, Snapshot};

const TRAY_ID: &str = "main";

pub struct TrayMenu {
    toggle: MenuItem<Wry>,
    restart: MenuItem<Wry>,
    login: CheckMenuItem<Wry>,
    home: PathBuf,
}

fn icon(name: &str) -> Image<'static> {
    let bytes: &'static [u8] = match name {
        "idle" => include_bytes!("../icons/tray-idle.png"),
        "listening" => include_bytes!("../icons/tray-listening.png"),
        "speaking" => include_bytes!("../icons/tray-speaking.png"),
        "working" => include_bytes!("../icons/tray-working.png"),
        _ => include_bytes!("../icons/tray-error.png"),
    };
    Image::from_bytes(bytes).expect("valid tray icon")
}

pub fn create(app: &App, home: &Path) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open Verbix", true, None::<&str>)?;
    let toggle = MenuItem::with_id(app, "toggle", "Toggle listening", false, None::<&str>)?;
    let restart = MenuItem::with_id(app, "restart", "Restart voice service", false, None::<&str>)?;
    let login_state = autostart::menu_state(autostart::status(home));
    let login = CheckMenuItem::with_id(
        app,
        "autostart",
        "Launch at login",
        login_state.enabled,
        login_state.checked,
        None::<&str>,
    )?;
    let setup = MenuItem::with_id(app, "setup", "Setup…", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &toggle, &restart, &login, &setup, &quit])?;

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon("idle"))
        .tooltip("Verbix — starting")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "open" => panel::show(app),
            "toggle" => panel::request_toggle(app),
            "restart" => {
                if let Some(supervisor) = app.try_state::<Supervisor>() {
                    supervisor.command(Command::Restart);
                }
            }
            "autostart" => {
                if let Some(tray) = app.try_state::<TrayMenu>() {
                    let enabled = autostart::status(&tray.home) == Status::Enabled;
                    let _ =
                        autostart::set_enabled(&tray.home, !enabled, &autostart::executable_path());
                }
                refresh(app);
            }
            "setup" => panel::show_setup(app),
            "quit" => crate::app_state::quit(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                panel::show(tray.app_handle());
            }
        })
        .build(app)?;

    app.manage(TrayMenu {
        toggle,
        restart,
        login,
        home: home.to_path_buf(),
    });
    refresh(app.handle());
    Ok(())
}

pub fn refresh(app: &AppHandle) {
    let Some(state) = app.try_state::<crate::ShellState>() else {
        return;
    };
    let Ok(mut snapshot) = state.snapshot.lock() else {
        return;
    };

    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_icon(Some(icon(snapshot.tray_icon())));
        let _ = tray.set_tooltip(Some(snapshot.tooltip()));
    }
    let mut autostart_changed = false;
    if let Some(menu) = app.try_state::<TrayMenu>() {
        let disconnected = matches!(snapshot.phase, Phase::Restarting | Phase::Failed);
        let _ = menu.toggle.set_enabled(snapshot.phase == Phase::Connected);
        let _ = menu.restart.set_enabled(disconnected);
        // Re-read the entry every refresh so external edits show up.
        let login_state = autostart::menu_state(autostart::status(&menu.home));
        let _ = menu.login.set_enabled(login_state.enabled);
        let _ = menu.login.set_checked(login_state.checked);
        if snapshot.setup.autostart != login_state.checked {
            snapshot.setup.autostart = login_state.checked;
            autostart_changed = true;
        }
    }
    drop(snapshot);
    if autostart_changed {
        crate::app_state::notify(app);
    }
}

pub fn snapshot(app: &AppHandle) -> Snapshot {
    app.state::<crate::ShellState>()
        .snapshot
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| Snapshot::new(Phase::Failed, crate::ipc::Floor::Idle, None))
}
