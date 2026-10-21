pub mod app_state;
pub mod autostart;
pub mod command;
pub mod ipc;
pub mod logging;
pub mod panel;
pub mod platform;
pub mod provider;
pub mod service;
pub mod state;
pub mod tray;

pub use app_state::ShellState;

use command::{Forwarded, ParsedCommand, Startup};

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

            app_state::manage(app, app_state::default_snapshot());
            panel::create(app)?;

            let config =
                app_state::production_config(&home, &resources, startup == Startup::Toggle);
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
