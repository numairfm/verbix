use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// What the download reports while it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadEvent {
    /// A non-empty line of standard output.
    Progress(String),
    /// A non-empty line of standard error.
    Stderr(String),
    /// The child exited.
    Finished { success: bool },
}

/// The child process the setup runs to fetch the speech model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadJob {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub work_dir: PathBuf,
    pub env: Vec<(String, Option<String>)>,
}

/// A running download. The caller reads the events and can stop the child.
#[derive(Clone)]
pub struct DownloadHandle {
    inner: Arc<DownloadInner>,
}

struct DownloadInner {
    events: Mutex<Receiver<DownloadEvent>>,
    child: Arc<Mutex<Option<Child>>>,
    pgid: i32,
}

impl DownloadHandle {
    /// Wait for the next download event.
    pub fn recv(&self) -> Option<DownloadEvent> {
        let events = self.inner.events.lock().ok()?;
        events.recv().ok()
    }

    /// Stop the download. The child is killed with its process group, so a
    /// shell quit leaves nothing behind.
    pub fn kill(&self) {
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(-self.inner.pgid, libc::SIGTERM);
            }
        }
        if let Ok(mut guard) = self.inner.child.lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.kill();
            }
        }
    }
}

/// Start the downloader and report its output.
pub fn start(job: &DownloadJob) -> std::io::Result<DownloadHandle> {
    crate::setup::create_private_dir(&job.work_dir)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;

    let mut command = Command::new(&job.program);
    command
        .args(&job.args)
        .current_dir(&job.work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in &job.env {
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
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child = Arc::new(Mutex::new(Some(child)));

    let (tx, rx) = channel();
    if let Some(stdout) = stdout {
        spawn_reader(stdout, tx.clone(), true);
    }
    if let Some(stderr) = stderr {
        spawn_reader(stderr, tx.clone(), false);
    }

    let monitor_child = Arc::clone(&child);
    let monitor_tx = tx.clone();
    thread::Builder::new()
        .name("model-download-monitor".to_string())
        .spawn(move || {
            let success = loop {
                let state = monitor_child.lock().ok().and_then(|mut guard| {
                    let child = guard.as_mut()?;
                    child.try_wait().ok()
                });
                match state {
                    Some(Some(status)) => break status.success(),
                    Some(None) => {}
                    None => break false,
                }
                thread::sleep(Duration::from_millis(50));
            };
            let _ = monitor_tx.send(DownloadEvent::Finished { success });
        })?;

    Ok(DownloadHandle {
        inner: Arc::new(DownloadInner {
            events: Mutex::new(rx),
            child,
            pgid,
        }),
    })
}

fn spawn_reader<R>(reader: R, tx: Sender<DownloadEvent>, stdout: bool)
where
    R: std::io::Read + Send + 'static,
{
    thread::Builder::new()
        .name("model-download-reader".to_string())
        .spawn(move || {
            for line in BufReader::new(reader).lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let event = if stdout {
                    DownloadEvent::Progress(line)
                } else {
                    DownloadEvent::Stderr(line)
                };
                if tx.send(event).is_err() {
                    break;
                }
            }
        })
        .ok();
}
