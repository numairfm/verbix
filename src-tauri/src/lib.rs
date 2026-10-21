pub mod app_state;
pub mod autostart;
pub mod command;
pub mod download;
pub mod ipc;
pub mod logging;
pub mod panel;
pub mod platform;
pub mod provider;
pub mod service;
pub mod setup;
pub mod state;
pub mod tray;

pub use app_state::ShellState;

use command::{Forwarded, ParsedCommand, Startup};
use state::Phase;

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
    let result = tauri::Builder::default()
        // Must be the first plugin: the second process forwards its launch and
        // exits before any GUI, service, or tray can be created.
        .plugin(tauri_plugin_single_instance::init(
            |app, argv, _cwd| match command::forward(argv.into_iter().skip(1)) {
                Forwarded::Show => panel::show(app),
                Forwarded::Nothing => {}
                Forwarded::Toggle => panel::request_toggle(app),
            },
        ))
        .setup(move |app| {
            use tauri::Manager;

            let handle = app.handle().clone();
            let home = platform::home_dir();
            let resources = app.path().resource_dir()?;

            let detection = app_state::detect_setup(&home);
            let plan = app_state::startup_plan(&detection, startup);
            let autostart_enabled = autostart::status(&home) == autostart::Status::Enabled;
            let mut snapshot = app_state::default_snapshot();
            snapshot.setup = app_state::setup_fields(&detection, None, None);
            snapshot.setup.autostart = autostart_enabled;
            snapshot.setup.required = !detection.complete();
            snapshot.phase = if plan.start_service {
                Phase::Starting
            } else {
                Phase::SetupRequired
            };

            app_state::manage(
                app,
                snapshot,
                app_state::SetupContext::new(home.clone(), resources.clone()),
            );
            panel::create(app)?;
            tray::create(app, &home)?;

            if plan.start_service {
                app_state::start_service(&handle, plan.startup_toggle)?;
                app_state::log_shell(&handle, "setup complete");
            } else {
                // An incomplete setup opens the panel, even for a hidden launch.
                app_state::log_shell(&handle, "setup required");
            }
            if plan.show_panel {
                panel::show(&handle);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            panel::get_state,
            panel::hide_panel,
            panel::toggle,
            panel::restart_service,
            panel::save_provider_key,
            panel::start_model_download,
            panel::set_autostart,
            panel::finish_setup
        ])
        .run(tauri::generate_context!());

    if let Err(error) = result {
        eprintln!("Verbix failed: {error}");
        std::process::exit(1);
    }
}
