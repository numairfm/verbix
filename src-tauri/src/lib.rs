pub mod app_state;
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
    let result = tauri::Builder::default()
        .setup(move |app| {
            use tauri::Manager;

            let handle = app.handle().clone();
            let home = platform::home_dir();
            let resources = app.path().resource_dir()?;

            app_state::manage(app, app_state::default_snapshot());
            panel::create(app)?;

            let config =
                app_state::production_config(&home, &resources, startup == Startup::Toggle);
            let supervisor = service::spawn_supervisor(config, app_state::observer(&handle));
            app.manage(supervisor);

            tray::create(app)?;
            if startup == Startup::Show {
                panel::show(&handle);
            }
            Ok(())
        })
        .run(tauri::generate_context!());

    if let Err(error) = result {
        eprintln!("Verbix failed: {error}");
        std::process::exit(1);
    }
}
