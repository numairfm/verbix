#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use tempfile::tempdir;

use verbix_shell::ipc::{Floor, PanelEvent};
use verbix_shell::service::{
    spawn_supervisor, BackoffPolicy, Command, Observation, Observer, ServiceConfig, Supervisor,
};
use verbix_shell::state::{Phase, Snapshot, TaskState};

struct ChannelObserver(mpsc::Sender<Observation>);

impl Observer for ChannelObserver {
    fn observe(&self, observation: Observation) {
        let _ = self.0.send(observation);
    }
}

fn python() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn test_config(dir: &Path) -> ServiceConfig {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fake_voice_service.py");
    let socket = dir.join("verbix.sock");
    let pid_file = dir.join("service.pid");
    ServiceConfig {
        program: PathBuf::from(python()),
        args: vec![
            script.display().to_string(),
            "--socket".to_string(),
            socket.display().to_string(),
            "--pid-file".to_string(),
            pid_file.display().to_string(),
        ],
        socket_path: socket,
        work_dir: dir.to_path_buf(),
        log_path: dir.join("voice-service.log"),
        voice_dir: dir.join("voice"),
        env: Vec::new(),
        startup_toggle: false,
        readiness_timeout: Duration::from_secs(10),
        ping_interval: Duration::from_millis(50),
        connected_reset: Duration::from_secs(60),
        backoff: BackoffPolicy {
            delays: vec![
                Duration::from_millis(50),
                Duration::from_millis(100),
                Duration::from_millis(150),
                Duration::from_millis(200),
                Duration::from_millis(250),
            ],
            max_restarts: 5,
        },
        shell_log: None,
    }
}

fn pump_until(
    rx: &mpsc::Receiver<Observation>,
    seen: &mut Vec<Observation>,
    timeout: Duration,
    mut predicate: impl FnMut(&[Observation]) -> bool,
) {
    let deadline = Instant::now() + timeout;
    loop {
        while let Ok(observation) = rx.try_recv() {
            seen.push(observation);
            if predicate(seen) {
                return;
            }
        }
        assert!(Instant::now() < deadline, "timed out; seen: {seen:?}");
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(observation) => seen.push(observation),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("observer disconnected; seen: {seen:?}")
            }
        }
        if predicate(seen) {
            return;
        }
    }
}

fn connected_count(seen: &[Observation]) -> usize {
    seen.iter()
        .filter(|observation| **observation == Observation::Phase(Phase::Connected))
        .count()
}

fn read_pid(path: &Path) -> i32 {
    std::fs::read_to_string(path)
        .expect("pid file")
        .trim()
        .parse()
        .expect("pid")
}

fn process_exists(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

fn stop(supervisor: Supervisor) {
    supervisor.command(Command::Quit);
    supervisor.join();
}

fn floor_cycle_complete(seen: &[Observation]) -> bool {
    let Some(listening) = seen
        .iter()
        .position(|observation| *observation == Observation::Floor(Floor::Listening))
    else {
        return false;
    };
    let Some(speaking) = seen
        .iter()
        .enumerate()
        .skip(listening + 1)
        .find(|(_, observation)| **observation == Observation::Floor(Floor::Speaking))
        .map(|(index, _)| index)
    else {
        return false;
    };
    seen.iter()
        .skip(speaking + 1)
        .any(|observation| *observation == Observation::Floor(Floor::Idle))
}

#[test]
fn ipc_toggle_follows_floor_events_and_quit_leaves_no_child() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());
    let pid_file = dir.path().join("service.pid");
    let (tx, rx) = mpsc::channel();
    let supervisor = spawn_supervisor(config, Arc::new(ChannelObserver(tx)));
    let mut seen = Vec::new();

    pump_until(&rx, &mut seen, Duration::from_secs(10), |seen| {
        connected_count(seen) == 1
    });

    supervisor.command(Command::Toggle);
    pump_until(&rx, &mut seen, Duration::from_secs(5), floor_cycle_complete);

    let listening = seen
        .iter()
        .position(|observation| *observation == Observation::Floor(Floor::Listening))
        .unwrap();
    let speaking = seen
        .iter()
        .position(|observation| *observation == Observation::Floor(Floor::Speaking))
        .unwrap();
    let idle = seen
        .iter()
        .enumerate()
        .skip(speaking + 1)
        .find(|(_, observation)| **observation == Observation::Floor(Floor::Idle))
        .map(|(index, _)| index)
        .unwrap();
    assert!(listening < speaking && speaking < idle, "seen: {seen:?}");

    let pid = read_pid(&pid_file);
    assert!(process_exists(pid));
    stop(supervisor);
    assert!(!process_exists(pid), "child {pid} survived quit");
}

fn snapshot_of(seen: &[Observation]) -> Snapshot {
    let mut snapshot = Snapshot::new(Phase::Connected, Floor::Idle, None);
    for observation in seen {
        match observation {
            Observation::Phase(phase) => snapshot.phase = *phase,
            Observation::Floor(floor) => snapshot.floor = *floor,
            Observation::Event(event) => snapshot.apply_panel_event(event.clone()),
        }
    }
    snapshot
}

#[test]
fn panel_events_reach_the_observer_and_tasks_become_terminal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());
    let pid_file = dir.path().join("service.pid");
    let (tx, rx) = mpsc::channel();
    let supervisor = spawn_supervisor(config, Arc::new(ChannelObserver(tx)));
    let mut seen = Vec::new();

    pump_until(&rx, &mut seen, Duration::from_secs(10), |seen| {
        connected_count(seen) == 1
    });

    supervisor.command(Command::Toggle);
    pump_until(&rx, &mut seen, Duration::from_secs(5), |seen| {
        let Some(cancelled) = seen.iter().position(|observation| {
            matches!(
                observation,
                Observation::Event(PanelEvent::TaskCancelled { task_id }) if task_id == "task-3"
            )
        }) else {
            return false;
        };
        seen.iter()
            .skip(cancelled + 1)
            .any(|observation| *observation == Observation::Floor(Floor::Idle))
    });

    let snapshot = snapshot_of(&seen);
    assert_eq!(snapshot.activity.transcript, "what time is it in tokyo");
    assert_eq!(
        snapshot.activity.caption, "Task one is complete.",
        "worker announcement should replace the previous finalized caption"
    );
    assert_eq!(snapshot.tasks.len(), 3, "{:?}", snapshot.tasks);
    assert_eq!(snapshot.tasks[0].id, "task-3");
    assert_eq!(snapshot.tasks[1].id, "task-2");
    assert_eq!(snapshot.tasks[2].id, "task-1");
    assert_eq!(snapshot.tasks[0].state, TaskState::Cancelled);
    assert_eq!(snapshot.tasks[1].state, TaskState::Failed);
    assert_eq!(snapshot.tasks[2].state, TaskState::Completed);
    assert_eq!(
        snapshot.tasks[2].result,
        "The repo is a voice agent with workers."
    );
    assert_eq!(snapshot.floor, Floor::Idle);

    let pid = read_pid(&pid_file);
    stop(supervisor);
    assert!(!process_exists(pid), "child {pid} survived quit");
}

#[test]
fn killed_child_restarts_with_backoff_and_reconnects() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());
    let pid_file = dir.path().join("service.pid");
    let (tx, rx) = mpsc::channel();
    let supervisor = spawn_supervisor(config, Arc::new(ChannelObserver(tx)));
    let mut seen = Vec::new();

    pump_until(&rx, &mut seen, Duration::from_secs(10), |seen| {
        connected_count(seen) == 1
    });
    let first_pid = read_pid(&pid_file);

    unsafe {
        libc::kill(first_pid, libc::SIGKILL);
    }

    pump_until(&rx, &mut seen, Duration::from_secs(10), |seen| {
        connected_count(seen) == 2
    });
    assert!(
        seen.contains(&Observation::Phase(Phase::Restarting))
            || seen.contains(&Observation::Phase(Phase::Starting)),
        "no restart transition in {seen:?}"
    );

    let second_pid = read_pid(&pid_file);
    assert_ne!(first_pid, second_pid);

    stop(supervisor);
    assert!(!process_exists(second_pid));
}
