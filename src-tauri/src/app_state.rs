use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use tauri::{AppHandle, Emitter, Manager};

use crate::ipc::Floor;
use crate::logging::LogSink;
use crate::service::{Command, Observation, Observer, Supervisor};
use crate::setup::{self, Detection};
use crate::state::{Coalescer, EmitDecision, Phase, Snapshot, EMIT_INTERVAL};
use crate::tray;

pub const STATE_EVENT: &str = "shell://state";

/// The paths the setup surface works with, and the running model download.
pub struct SetupContext {
    pub home: PathBuf,
    pub resources: PathBuf,
    pub download: Mutex<Option<crate::download::DownloadHandle>>,
    pub download_error: Mutex<Option<String>>,
}

impl SetupContext {
    pub fn new(home: PathBuf, resources: PathBuf) -> Self {
        Self {
            home,
            resources,
            download: Mutex::new(None),
            download_error: Mutex::new(None),
        }
    }
}

pub struct ShellState {
    pub snapshot: Mutex<Snapshot>,
    pub supervisor: Mutex<Option<Supervisor>>,
    pub setup: SetupContext,
    pub log: Option<LogSink>,
    changes: mpsc::Sender<()>,
}

pub struct TauriObserver {
    pub app: AppHandle,
    changes: mpsc::Sender<()>,
    log_path: PathBuf,
}

impl TauriObserver {
    fn apply(&self, observation: Observation) {
        if let Some(state) = self.app.try_state::<ShellState>() {
            if let Ok(mut snapshot) = state.snapshot.lock() {
                match observation {
                    Observation::Phase(phase) => {
                        snapshot.phase = phase;
                        snapshot.error = (phase == Phase::Failed).then(|| {
                            format!("Voice service failed. See {}.", self.log_path.display())
                        });
                    }
                    Observation::Floor(floor) => {
                        snapshot.floor = floor;
                    }
                    Observation::Event(event) => snapshot.apply_panel_event(event),
                }
            }
        }
        let _ = self.changes.send(());
    }
}

impl Observer for TauriObserver {
    fn observe(&self, observation: Observation) {
        self.apply(observation);
        let app = self.app.clone();
        let _ = self.app.run_on_main_thread(move || tray::refresh(&app));
    }

    fn quit(&self) {
        let app = self.app.clone();
        let _ = self.app.run_on_main_thread(move || app.exit(0));
    }
}

fn emit_loop(app: AppHandle, changes: mpsc::Receiver<()>) {
    let mut coalescer = Coalescer::new(EMIT_INTERVAL);
    while changes.recv().is_ok() {
        loop {
            match coalescer.changed(Instant::now()) {
                EmitDecision::Now => break,
                EmitDecision::Wait(delay) => match changes.recv_timeout(delay) {
                    Ok(()) => {}
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                },
            }
        }
        if let Some(state) = app.try_state::<ShellState>() {
            if let Ok(snapshot) = state.snapshot.lock() {
                let _ = app.emit(STATE_EVENT, snapshot.clone());
            }
        }
        coalescer.mark_emitted(Instant::now());
    }
}

pub fn production_config(
    home: &std::path::Path,
    resources: &std::path::Path,
    startup_toggle: bool,
    shell_log: Option<LogSink>,
) -> crate::service::ServiceConfig {
    let voice_root = resources.join("voice");
    let voice_dir = setup::voice_dir(home);
    let socket = crate::platform::socket_path(home);
    let work_dir = setup::work_dir(home);
    let logs = crate::platform::logs_dir(home);
    let gate_home = home.to_path_buf();

    crate::service::ServiceConfig {
        program: voice_root.join("python/bin/python3"),
        args: vec![
            "-m".to_string(),
            "verbix.cli".to_string(),
            "--serve".to_string(),
            "--voice".to_string(),
        ],
        socket_path: socket,
        work_dir,
        log_path: logs.join("voice-service.log"),
        voice_dir,
        env: child_env(resources, home),
        startup_toggle,
        readiness_timeout: std::time::Duration::from_secs(30),
        ping_interval: std::time::Duration::from_millis(250),
        connected_reset: std::time::Duration::from_secs(60),
        backoff: Default::default(),
        shell_log,
        // A missing key or model is a setup failure, not a crash loop: the
        // service is only spawned while the setup is complete.
        setup_gate: Some(Arc::new(move || detect_setup(&gate_home).complete())),
    }
}

/// The environment the voice service and the model downloader share.
pub fn child_env(
    resources: &std::path::Path,
    home: &std::path::Path,
) -> Vec<(String, Option<String>)> {
    let voice_root = resources.join("voice");
    let voice_dir = setup::voice_dir(home);
    let mut env = vec![
        (
            "PYTHONPATH".to_string(),
            Some(voice_root.join("site-packages").display().to_string()),
        ),
        ("PYTHONHOME".to_string(), None),
        ("PYTHONNOUSERSITE".to_string(), Some("1".to_string())),
        ("PYTHONDONTWRITEBYTECODE".to_string(), Some("1".to_string())),
        (
            "VERBIX_VOICE_DIR".to_string(),
            Some(voice_dir.display().to_string()),
        ),
    ];
    if let Some(provider) =
        crate::provider::openrouter_override(home, &|name| std::env::var(name).ok())
    {
        env.push(("VERBIX_PROVIDER".to_string(), Some(provider)));
    }
    env
}

pub fn default_snapshot() -> Snapshot {
    Snapshot::new(Phase::Starting, Floor::Idle, None)
}

/// Refresh the fallback command before the panel shows it, so a moved
/// AppImage shows the current path.
pub fn refresh_shortcut(app: &AppHandle) {
    let Some(state) = app.try_state::<ShellState>() else {
        return;
    };
    let active = state
        .snapshot
        .lock()
        .map(|snapshot| snapshot.shortcut.accelerator.is_none())
        .unwrap_or(false);
    if !active {
        return;
    }
    let command = crate::shortcut::fallback_command(&crate::platform::executable_path());
    update(app, |snapshot| {
        if snapshot.shortcut.fallback.as_deref() != Some(command.as_str()) {
            snapshot.shortcut.fallback = Some(command);
        }
    });
}

pub fn manage(app: &tauri::App, snapshot: Snapshot, context: SetupContext) {
    let (changes, receiver) = mpsc::channel();
    let logs = crate::platform::logs_dir(&context.home);
    let log = LogSink::open(&logs.join("shell.log")).ok();
    app.manage(ShellState {
        snapshot: Mutex::new(snapshot),
        supervisor: Mutex::new(None),
        setup: context,
        log,
        changes: changes.clone(),
    });
    let emitter_app = app.handle().clone();
    let _ = thread::Builder::new()
        .name("shell-state-emitter".to_string())
        .spawn(move || emit_loop(emitter_app, receiver));
}

pub fn observer(
    app: &AppHandle,
    changes: mpsc::Sender<()>,
    log_path: PathBuf,
) -> Arc<dyn Observer> {
    Arc::new(TauriObserver {
        app: app.clone(),
        changes,
        log_path,
    })
}

/// Ask the panel to re-render after a state change.
pub fn notify(app: &AppHandle) {
    if let Some(state) = app.try_state::<ShellState>() {
        let _ = state.changes.send(());
    }
}

/// The current snapshot, for the panel and the tray.
pub fn snapshot_of(app: &AppHandle) -> Snapshot {
    app.try_state::<ShellState>()
        .and_then(|state| state.snapshot.lock().ok().map(|snapshot| snapshot.clone()))
        .unwrap_or_else(|| Snapshot::new(Phase::Failed, Floor::Idle, None))
}

/// Change the snapshot and let the panel know.
pub fn update(app: &AppHandle, change: impl FnOnce(&mut Snapshot)) {
    if let Some(state) = app.try_state::<ShellState>() {
        if let Ok(mut snapshot) = state.snapshot.lock() {
            change(&mut snapshot);
        }
    }
    notify(app);
}

/// Inspect the setup surface of the current home.
pub fn detect_setup(home: &std::path::Path) -> Detection {
    setup::detect(home, &|name| std::env::var(name).ok())
}

/// What the shell does at start-up. An incomplete setup opens the panel and
/// keeps the voice service from starting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupPlan {
    pub start_service: bool,
    pub show_panel: bool,
    pub startup_toggle: bool,
}

pub fn startup_plan(detection: &Detection, startup: crate::command::Startup) -> StartupPlan {
    if !detection.complete() {
        return StartupPlan {
            start_service: false,
            show_panel: true,
            startup_toggle: false,
        };
    }
    StartupPlan {
        start_service: true,
        show_panel: startup == crate::command::Startup::Show,
        startup_toggle: startup == crate::command::Startup::Toggle,
    }
}

/// Write one line to the shell log.
pub fn log_shell(app: &AppHandle, message: &str) {
    if let Some(state) = app.try_state::<ShellState>() {
        if let Some(sink) = &state.log {
            sink.line(message);
        }
    }
}

/// Start the voice service. The service only starts on a complete setup.
pub fn start_service(app: &AppHandle, startup_toggle: bool) -> Result<(), String> {
    let state = app
        .try_state::<ShellState>()
        .ok_or_else(|| "The shell state is not ready.".to_string())?;
    let detection = detect_setup(&state.setup.home);
    if !detection.complete() {
        update(app, |snapshot| {
            snapshot.phase = Phase::SetupRequired;
            snapshot.setup = setup_fields(&detection, snapshot.setup.download.clone(), None);
            snapshot.setup.required = true;
        });
        crate::panel::show(app);
        return Err("Setup is not complete.".to_string());
    }

    let config = production_config(
        &state.setup.home,
        &state.setup.resources,
        startup_toggle,
        state.log.clone(),
    );
    let log_path = config.log_path.clone();
    let changes = state.changes.clone();
    let supervisor = crate::service::spawn_supervisor(config, observer(app, changes, log_path));
    if let Ok(mut slot) = state.supervisor.lock() {
        *slot = Some(supervisor);
    }
    update(app, |snapshot| snapshot.phase = Phase::Starting);
    Ok(())
}

/// Stop the service and the model download, then leave.
pub fn quit(app: &AppHandle) {
    if let Some(state) = app.try_state::<ShellState>() {
        if let Ok(mut download) = state.setup.download.lock() {
            if let Some(handle) = download.take() {
                handle.kill();
            }
        }
        let supervisor = state
            .supervisor
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        match supervisor {
            Some(supervisor) => supervisor.command(Command::Quit),
            None => {
                let _ = app.exit(0);
            }
        }
    }
}

/// The setup fields of the panel snapshot.
pub fn setup_fields(
    detection: &Detection,
    download: Option<crate::state::DownloadState>,
    download_error: Option<String>,
) -> crate::state::Setup {
    let mut fields = crate::state::Setup {
        required: !detection.complete(),
        key_ready: detection.key.ready,
        key_editable: detection.key.editable,
        model_ready: detection.model.ready,
        model_path: detection
            .model
            .path
            .as_ref()
            .map(|path| path.display().to_string()),
        model_downloadable: detection.model.downloadable,
        autostart: false,
        autostart_supported: crate::autostart::supported(),
        download,
        error: None,
    };
    fields.error = download_error
        .or_else(|| detection.key.error.clone())
        .or_else(|| detection.model.error.clone());
    fields
}
