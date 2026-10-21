pub mod app_state;
pub mod autostart;
pub mod command;
pub mod ipc;
pub mod logging;
pub mod panel;
pub mod platform;
pub mod provider;
pub mod service;
pub mod shortcut;
pub mod state;
pub mod tray;

pub use app_state::ShellState;

use command::{ParsedCommand, Startup};

pub fn run() {
    match command::parse(std::env::args().skip(1)) {
        ParsedCommand::Version => {
            println!("Verbix {}", env!("CARGO_PKG_VERSION"));
        }
        ParsedCommand::Help => {
            print!("{}", command::usage());
        }
        ParsedCommand::Invalid(message) => {
            eprintln!("{message}");
            eprintln!("{}", command::usage());
            std::process::exit(2);
        }
        ParsedCommand::Run(startup) => run_gui(startup),
    }
}

fn run_gui(startup: Startup) {
    let mut builder = tauri::Builder::default();
    // The global shortcut plugin needs an X11 display. On a native Wayland
    // session the desktop-level fallback is the supported path, so the plugin
    // is not loaded at all.
    if shortcut::session_kind(&|name| std::env::var(name).ok()) == shortcut::SessionKind::X11 {
        builder = builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if shortcut::is_pressed(event.state()) {
                        shortcut::on_pressed(app);
                    }
                })
                .build(),
        );
    }

    let result = builder
        .setup(move |app| {
            use tauri::Manager;

            let handle = app.handle().clone();
            let home = platform::home_dir();
            let resources = app.path().resource_dir()?;

            let shortcut_state = shortcut::register(&handle);
            let mut snapshot = app_state::default_snapshot();
            snapshot.shortcut = shortcut_state.clone();
            app_state::manage(app, snapshot, &home);
            panel::create(app)?;

            let shell_log = app
                .try_state::<ShellState>()
                .and_then(|state| state.log.clone());
            let config = app_state::production_config(
                &home,
                &resources,
                startup == Startup::Toggle,
                shell_log,
            );
            app_state::log_shell(
                &handle,
                if shortcut_state.accelerator.is_some() {
                    "global shortcut registered"
                } else {
                    "global shortcut fallback"
                },
            );
            let log_path = config.log_path.clone();
            let supervisor =
                service::spawn_supervisor(config, app_state::observer(&handle, log_path));
            app.manage(supervisor);

            tray::create(app, &home)?;
            if startup == Startup::Show {
                panel::show(&handle);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            panel::get_state,
            panel::hide_panel,
            panel::toggle,
            panel::restart_service
        ])
        .run(tauri::generate_context!());

    if let Err(error) = result {
        eprintln!("Verbix failed: {error}");
        std::process::exit(1);
    }
}
