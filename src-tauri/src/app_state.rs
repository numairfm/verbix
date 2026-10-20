use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use tauri::{AppHandle, Emitter, Manager};

use crate::ipc::Floor;
use crate::logging::LogSink;
use crate::service::{Observation, Observer};
use crate::state::{Coalescer, EmitDecision, Phase, Snapshot, EMIT_INTERVAL};
use crate::tray;

pub const STATE_EVENT: &str = "shell://state";

pub struct ShellState {
    pub snapshot: Mutex<Snapshot>,
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
) -> crate::service::ServiceConfig {
    let voice_root = resources.join("voice");
    let voice_dir = crate::platform::voice_dir(home);
    let socket = crate::platform::socket_path(home);
    let work_dir = crate::platform::work_dir(home);
    let logs = crate::platform::logs_dir(home);
    let shell_log = LogSink::open(&logs.join("shell.log")).ok();

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
        env,
        startup_toggle,
        readiness_timeout: std::time::Duration::from_secs(30),
        ping_interval: std::time::Duration::from_millis(250),
        connected_reset: std::time::Duration::from_secs(60),
        backoff: Default::default(),
        shell_log,
    }
}

pub fn default_snapshot() -> Snapshot {
    Snapshot::new(Phase::Starting, Floor::Idle, None)
}

pub fn manage(app: &tauri::App, snapshot: Snapshot) {
    app.manage(ShellState {
        snapshot: Mutex::new(snapshot),
    });
}

pub fn observer(app: &AppHandle, log_path: PathBuf) -> Arc<dyn Observer> {
    let (changes, receiver) = mpsc::channel();
    let emitter_app = app.clone();
    let _ = thread::Builder::new()
        .name("shell-state-emitter".to_string())
        .spawn(move || emit_loop(emitter_app, receiver));
    Arc::new(TauriObserver {
        app: app.clone(),
        changes,
        log_path,
    })
}
