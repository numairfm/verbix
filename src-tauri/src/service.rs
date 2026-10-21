use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::ipc::{
    floor_from_event, panel_event_from_event, parse_line, request_line, Floor, Incoming, PanelEvent,
};
use crate::logging::{pump_reader, LogSink};
use crate::state::Phase;

const MAX_LINE_BYTES: usize = 1024 * 1024;
const POLL_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Toggle,
    Restart,
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observation {
    Phase(Phase),
    Floor(Floor),
    Event(PanelEvent),
}

pub trait Observer: Send + Sync + 'static {
    fn observe(&self, observation: Observation);

    fn quit(&self) {}
}

#[derive(Debug, Clone)]
pub struct BackoffPolicy {
    pub delays: Vec<Duration>,
    pub max_restarts: u32,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            delays: vec![
                Duration::from_secs(1),
                Duration::from_secs(2),
                Duration::from_secs(5),
                Duration::from_secs(10),
                Duration::from_secs(30),
            ],
            max_restarts: 5,
        }
    }
}

impl BackoffPolicy {
    pub fn delay_for_restart(&self, restarts_done: u32) -> Option<Duration> {
        if restarts_done >= self.max_restarts {
            return None;
        }
        self.delays
            .get(restarts_done as usize)
            .copied()
            .or_else(|| self.delays.last().copied())
    }
}

#[derive(Clone)]
pub struct ServiceConfig {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub socket_path: PathBuf,
    pub work_dir: PathBuf,
    pub log_path: PathBuf,
    pub voice_dir: PathBuf,
    pub env: Vec<(String, Option<String>)>,
    pub startup_toggle: bool,
    pub readiness_timeout: Duration,
    pub ping_interval: Duration,
    pub connected_reset: Duration,
    pub backoff: BackoffPolicy,
    pub shell_log: Option<LogSink>,
    /// Before every spawn the shell checks that the setup is still complete.
    /// A missing key or model is a setup failure, not a crash loop.
    pub setup_gate: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

pub struct Supervisor {
    tx: Sender<Command>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Supervisor {
    pub fn command(&self, command: Command) {
        let _ = self.tx.send(command);
    }

    pub fn join(&self) {
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn_supervisor(config: ServiceConfig, observer: Arc<dyn Observer>) -> Supervisor {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || run_loop(config, observer, rx));
    Supervisor {
        tx,
        handle: Mutex::new(Some(handle)),
    }
}

fn log(config: &ServiceConfig, message: &str) {
    if let Some(sink) = &config.shell_log {
        sink.line(message);
    }
}

fn run_loop(config: ServiceConfig, observer: Arc<dyn Observer>, rx: Receiver<Command>) {
    let mut restarts = 0u32;
    loop {
        observer.observe(Observation::Phase(Phase::Starting));
        log(&config, "phase starting");
        let outcome = run_once(&config, &observer, &rx);
        match outcome {
            Outcome::Quit => {
                observer.observe(Observation::Phase(Phase::Stopping));
                log(&config, "stopping");
                observer.quit();
                return;
            }
            Outcome::Restart => {
                restarts = 0;
                log(&config, "manual restart requested");
            }
            Outcome::SetupRequired => {
                observer.observe(Observation::Phase(Phase::SetupRequired));
                log(&config, "setup is incomplete; the voice service stays away");
                match wait_for_manual(&rx) {
                    Manual::Quit => {
                        observer.observe(Observation::Phase(Phase::Stopping));
                        observer.quit();
                        return;
                    }
                    Manual::Restart => {
                        restarts = 0;
                        continue;
                    }
                }
            }
            Outcome::Failure { detail, stable } => {
                if stable {
                    restarts = 0;
                }
                log(&config, &format!("service failed: {detail}"));
                let Some(delay) = config.backoff.delay_for_restart(restarts) else {
                    observer.observe(Observation::Phase(Phase::Failed));
                    log(&config, "retry budget exhausted");
                    match wait_for_manual(&rx) {
                        Manual::Quit => {
                            observer.observe(Observation::Phase(Phase::Stopping));
                            observer.quit();
                            return;
                        }
                        Manual::Restart => {
                            restarts = 0;
                            continue;
                        }
                    }
                };
                observer.observe(Observation::Phase(Phase::Restarting));
                log(&config, &format!("restarting in {} ms", delay.as_millis()));
                match sleep_with_commands(delay, &rx) {
                    SleepResult::Quit => {
                        observer.observe(Observation::Phase(Phase::Stopping));
                        observer.quit();
                        return;
                    }
                    SleepResult::Restart => {
                        restarts = 0;
                    }
                    SleepResult::Elapsed => {}
                }
                restarts += 1;
            }
        }
    }
}

enum Manual {
    Restart,
    Quit,
}

fn wait_for_manual(rx: &Receiver<Command>) -> Manual {
    loop {
        match rx.recv() {
            Ok(Command::Quit) | Err(_) => return Manual::Quit,
            Ok(Command::Restart) => return Manual::Restart,
            Ok(Command::Toggle) => {}
        }
    }
}

enum SleepResult {
    Elapsed,
    Restart,
    Quit,
}

fn sleep_with_commands(delay: Duration, rx: &Receiver<Command>) -> SleepResult {
    let deadline = Instant::now() + delay;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining.min(Duration::from_millis(50))) {
            Ok(Command::Quit) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return SleepResult::Quit
            }
            Ok(Command::Restart) => return SleepResult::Restart,
            Ok(Command::Toggle) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
    SleepResult::Elapsed
}

enum Outcome {
    Quit,
    Restart,
    SetupRequired,
    Failure { detail: String, stable: bool },
}

fn run_once(
    config: &ServiceConfig,
    observer: &Arc<dyn Observer>,
    rx: &Receiver<Command>,
) -> Outcome {
    if let Some(gate) = &config.setup_gate {
        if !gate() {
            return Outcome::SetupRequired;
        }
    }
    let mut child = match ManagedChild::spawn(config) {
        Ok(child) => child,
        Err(error) => {
            return Outcome::Failure {
                detail: format!("spawn failed: {error}"),
                stable: false,
            }
        }
    };
    log(config, &format!("child pid {}", child.pid()));

    let outcome = run_with_child(config, observer, rx, &mut child);
    child.stop();
    outcome
}

fn run_with_child(
    config: &ServiceConfig,
    observer: &Arc<dyn Observer>,
    rx: &Receiver<Command>,
    child: &mut ManagedChild,
) -> Outcome {
    let mut connection = match wait_ready(config, observer, rx, child) {
        Ok(connection) => connection,
        Err(outcome) => return outcome,
    };

    observer.observe(Observation::Phase(Phase::Connected));
    observer.observe(Observation::Floor(Floor::Idle));
    log(config, "connected");
    if config.startup_toggle {
        if let Err(error) = connection.write_request("toggle") {
            return Outcome::Failure {
                detail: format!("toggle write failed: {error}"),
                stable: false,
            };
        }
        log(config, "startup toggle sent");
    }

    let connected_at = Instant::now();
    let mut stable = false;
    let mut malformed = 0u8;

    loop {
        loop {
            match rx.try_recv() {
                Ok(Command::Quit) | Err(TryRecvError::Disconnected) => return Outcome::Quit,
                Ok(Command::Restart) => return Outcome::Restart,
                Ok(Command::Toggle) => match connection.write_request("toggle") {
                    Ok(id) => log(config, &format!("toggle sent ({id})")),
                    Err(error) => {
                        return Outcome::Failure {
                            detail: format!("toggle write failed: {error}"),
                            stable,
                        }
                    }
                },
                Err(TryRecvError::Empty) => break,
            }
        }

        if !stable && connected_at.elapsed() >= config.connected_reset {
            stable = true;
            log(config, "connection stable");
        }

        if child.has_exited() {
            return Outcome::Failure {
                detail: "child exited unexpectedly".to_string(),
                stable,
            };
        }

        match connection.poll_line() {
            Ok(Some(line)) => match parse_line(&line) {
                Incoming::Event(event) => {
                    malformed = 0;
                    if let Some(floor) = floor_from_event(&event) {
                        observer.observe(Observation::Floor(floor));
                        log(config, &format!("floor {}", floor_name(floor)));
                    }
                    if let Some(panel_event) = panel_event_from_event(&event) {
                        observer.observe(Observation::Event(panel_event));
                    }
                }
                Incoming::Response(response) => {
                    malformed = 0;
                    if response.status != "ok" {
                        log(
                            config,
                            &format!(
                                "request failed: {}",
                                response.error.unwrap_or_else(|| response.status)
                            ),
                        );
                    } else if let Some(action) =
                        response.data.get("action").and_then(|v| v.as_str())
                    {
                        log(config, &format!("toggle action {action}"));
                    }
                }
                Incoming::Malformed => {
                    malformed += 1;
                    log(config, "dropped malformed line");
                    if malformed >= 3 {
                        return Outcome::Failure {
                            detail: "three malformed lines".to_string(),
                            stable,
                        };
                    }
                }
            },
            Ok(None) => {}
            Err(error) => {
                return Outcome::Failure {
                    detail: format!("ipc read failed: {error}"),
                    stable,
                }
            }
        }
    }
}

fn floor_name(floor: Floor) -> &'static str {
    match floor {
        Floor::Idle => "idle",
        Floor::Listening => "listening",
        Floor::Speaking => "speaking",
    }
}

fn wait_ready(
    config: &ServiceConfig,
    _observer: &Arc<dyn Observer>,
    rx: &Receiver<Command>,
    child: &mut ManagedChild,
) -> Result<Connection, Outcome> {
    let deadline = Instant::now() + config.readiness_timeout;
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(Command::Quit) | Err(TryRecvError::Disconnected) => return Err(Outcome::Quit),
            Ok(Command::Restart) => return Err(Outcome::Restart),
            Ok(Command::Toggle) | Err(TryRecvError::Empty) => {}
        }
        if child.has_exited() {
            return Err(Outcome::Failure {
                detail: "child exited before ipc ready".to_string(),
                stable: false,
            });
        }
        if config.socket_path.exists() {
            if let Ok(mut connection) = Connection::connect(&config.socket_path) {
                if ping(&mut connection).is_ok() {
                    return Ok(connection);
                }
            }
        }
        thread::sleep(config.ping_interval.min(Duration::from_millis(250)));
    }
    Err(Outcome::Failure {
        detail: "ipc readiness timed out".to_string(),
        stable: false,
    })
}

fn ping(connection: &mut Connection) -> io::Result<()> {
    let id = connection.write_request("ping")?;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Some(line) = connection.poll_line()? {
            match parse_line(&line) {
                Incoming::Response(response)
                    if response.id.as_deref() == Some(id.as_str())
                        && response.status == "ok"
                        && response.data.get("pong") == Some(&serde_json::Value::Bool(true)) =>
                {
                    return Ok(());
                }
                _ => {}
            }
        }
    }
    Err(io::Error::new(io::ErrorKind::TimedOut, "ping timed out"))
}

struct ManagedChild {
    child: Child,
    pgid: i32,
}

impl ManagedChild {
    fn spawn(config: &ServiceConfig) -> io::Result<Self> {
        std::fs::create_dir_all(&config.work_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ =
                std::fs::set_permissions(&config.work_dir, std::fs::Permissions::from_mode(0o700));
        }
        if let Some(parent) = config.socket_path.parent() {
            std::fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }

        let sink = LogSink::open(&config.log_path)?;
        let mut command = ProcessCommand::new(&config.program);
        command
            .args(&config.args)
            .current_dir(&config.work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in &config.env {
            match value {
                Some(value) => {
                    command.env(key, value);
                }
                None => {
                    command.env_remove(key);
                }
            }
        }

        crate::platform::configure_child(&mut command);

        let mut child = command.spawn()?;
        let pgid = child.id() as i32;
        if let Some(stdout) = child.stdout.take() {
            pump_reader(stdout, sink.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            pump_reader(stderr, sink);
        }
        Ok(Self { child, pgid })
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn has_exited(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(Some(_)))
    }

    fn stop(&mut self) {
        if self.has_exited() {
            return;
        }
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(self.child.id() as i32, libc::SIGINT);
            }
            if self.wait_up_to(Duration::from_secs(5)) {
                return;
            }
            unsafe {
                libc::kill(-self.pgid, libc::SIGTERM);
            }
            if self.wait_up_to(Duration::from_secs(1)) {
                return;
            }
            unsafe {
                libc::kill(-self.pgid, libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }

    fn wait_up_to(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.has_exited() {
                return true;
            }
            thread::sleep(Duration::from_millis(25));
        }
        self.has_exited()
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        if !self.has_exited() {
            #[cfg(unix)]
            unsafe {
                libc::kill(-self.pgid, libc::SIGKILL);
            }
            #[cfg(not(unix))]
            {
                let _ = self.child.kill();
            }
            let _ = self.child.wait();
        }
    }
}

#[cfg(unix)]
struct Connection {
    stream: std::os::unix::net::UnixStream,
    pending: Vec<u8>,
    sequence: u64,
}

#[cfg(unix)]
impl Connection {
    fn connect(path: &Path) -> io::Result<Self> {
        let stream = std::os::unix::net::UnixStream::connect(path)?;
        stream.set_read_timeout(Some(POLL_TIMEOUT))?;
        stream.set_write_timeout(Some(Duration::from_secs(2)))?;
        Ok(Self {
            stream,
            pending: Vec::new(),
            sequence: 0,
        })
    }

    fn write_request(&mut self, action: &str) -> io::Result<String> {
        self.sequence += 1;
        let id = format!("{action}-{}", self.sequence);
        self.stream
            .write_all(format!("{}\n", request_line(&id, action)).as_bytes())?;
        Ok(id)
    }

    fn poll_line(&mut self) -> io::Result<Option<String>> {
        if let Some(position) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=position).collect();
            return Ok(Some(decode_line(&line)));
        }
        let mut buffer = [0u8; 8192];
        loop {
            match self.stream.read(&mut buffer) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "ipc connection closed",
                    ))
                }
                Ok(read) => {
                    self.pending.extend_from_slice(&buffer[..read]);
                    if self.pending.len() > MAX_LINE_BYTES {
                        self.pending.clear();
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "ipc line too long",
                        ));
                    }
                    if let Some(position) = self.pending.iter().position(|byte| *byte == b'\n') {
                        let line: Vec<u8> = self.pending.drain(..=position).collect();
                        return Ok(Some(decode_line(&line)));
                    }
                }
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock
                        || error.kind() == io::ErrorKind::TimedOut =>
                {
                    return Ok(None)
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(unix)]
fn decode_line(line: &[u8]) -> String {
    String::from_utf8_lossy(line)
        .trim_end_matches(['\r', '\n'])
        .to_string()
}

#[cfg(not(unix))]
struct Connection;

#[cfg(not(unix))]
impl Connection {
    fn connect(_path: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "ipc is not implemented on this platform yet",
        ))
    }

    fn write_request(&mut self, _action: &str) -> io::Result<String> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "ipc unsupported",
        ))
    }

    fn poll_line(&mut self) -> io::Result<Option<String>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "ipc unsupported",
        ))
    }
}
