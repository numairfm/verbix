use std::collections::HashSet;
use std::time::{Duration, Instant};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use crate::ipc::{Floor, PanelEvent};

pub const EMIT_INTERVAL: Duration = Duration::from_millis(50);
pub const TEXT_CAP: usize = 16_000;
pub const TASK_ROW_CAP: usize = 20;
pub const OBJECTIVE_CAP: usize = 200;
pub const RESULT_CAP: usize = 280;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    SetupRequired,
    Starting,
    Connected,
    Restarting,
    Failed,
    Stopping,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SetupRequired => "setup_required",
            Self::Starting => "starting",
            Self::Connected => "connected",
            Self::Restarting => "restarting",
            Self::Failed => "failed",
            Self::Stopping => "stopping",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrayState {
    Idle,
    Listening,
    Speaking,
    Working,
    Error,
}

impl TrayState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Listening => "listening",
            Self::Speaking => "speaking",
            Self::Working => "working",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Shortcut {
    pub accelerator: Option<String>,
    pub fallback: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Activity {
    pub transcript: String,
    pub caption: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TaskRow {
    pub id: String,
    pub objective: String,
    pub state: TaskState,
    pub result: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DownloadState {
    pub running: bool,
    pub last_line: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Setup {
    pub required: bool,
    pub key_ready: bool,
    pub key_editable: bool,
    pub model_ready: bool,
    pub model_path: Option<String>,
    pub model_downloadable: bool,
    pub autostart: bool,
    pub autostart_supported: bool,
    pub download: Option<DownloadState>,
    pub error: Option<String>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            required: false,
            key_ready: true,
            key_editable: true,
            model_ready: true,
            model_path: None,
            model_downloadable: true,
            autostart: false,
            autostart_supported: cfg!(target_os = "linux"),
            download: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub phase: Phase,
    pub floor: Floor,
    pub shortcut: Shortcut,
    pub activity: Activity,
    pub tasks: Vec<TaskRow>,
    pub setup: Setup,
    pub error: Option<String>,
    running_tasks: HashSet<String>,
    caption_streamed: bool,
    caption_finalized: bool,
}

impl Snapshot {
    pub fn new(phase: Phase, floor: Floor, error: Option<String>) -> Self {
        Self {
            phase,
            floor,
            shortcut: Shortcut::default(),
            activity: Activity::default(),
            tasks: Vec::new(),
            setup: Setup::default(),
            error,
            running_tasks: HashSet::new(),
            caption_streamed: false,
            caption_finalized: false,
        }
    }

    pub fn tray_state(&self) -> TrayState {
        if !matches!(self.phase, Phase::Connected | Phase::Starting) {
            return TrayState::Error;
        }
        if self.phase == Phase::Starting {
            return TrayState::Idle;
        }
        match self.floor {
            Floor::Listening => TrayState::Listening,
            Floor::Speaking => TrayState::Speaking,
            Floor::Idle => {
                if self.running_tasks.is_empty() {
                    TrayState::Idle
                } else {
                    TrayState::Working
                }
            }
        }
    }

    pub fn tray_icon(&self) -> &'static str {
        self.tray_state().as_str()
    }

    pub fn tooltip(&self) -> String {
        if self.phase == Phase::Starting {
            return "Verbix — starting".to_string();
        }
        format!("Verbix — {}", self.tray_state().as_str())
    }

    pub fn apply_panel_event(&mut self, event: PanelEvent) {
        match event {
            PanelEvent::UserTranscript { text } => {
                self.activity.transcript = cap(&text, TEXT_CAP);
                self.activity.caption.clear();
                self.caption_streamed = false;
                self.caption_finalized = false;
            }
            PanelEvent::TextDelta { text } => {
                if self.caption_finalized {
                    self.activity.caption = cap(&text, TEXT_CAP);
                    self.caption_finalized = false;
                } else {
                    self.activity.caption.push_str(&text);
                    self.activity.caption = cap(&self.activity.caption, TEXT_CAP);
                }
                self.caption_streamed = true;
            }
            PanelEvent::TurnComplete { final_text, .. } => {
                if !self.caption_streamed {
                    self.activity.caption = cap(&final_text, TEXT_CAP);
                }
                self.caption_streamed = false;
                self.caption_finalized = true;
            }
            PanelEvent::TaskRegistered { task_id, objective } => {
                self.running_tasks.insert(task_id.clone());
                let row = TaskRow {
                    id: task_id,
                    objective: cap(&objective, OBJECTIVE_CAP),
                    state: TaskState::Running,
                    result: String::new(),
                };
                self.tasks.insert(0, row);
                self.tasks.truncate(TASK_ROW_CAP);
            }
            PanelEvent::TaskCompleted { task_id, result } => {
                self.finish_task(&task_id, TaskState::Completed, Some(result));
            }
            PanelEvent::TaskFailed { task_id, result } => {
                self.finish_task(&task_id, TaskState::Failed, Some(result));
            }
            PanelEvent::TaskCancelled { task_id } => {
                self.finish_task(&task_id, TaskState::Cancelled, None);
            }
        }
    }

    fn finish_task(&mut self, task_id: &str, state: TaskState, result: Option<String>) {
        self.running_tasks.remove(task_id);
        let Some(row) = self.tasks.iter_mut().find(|row| row.id == task_id) else {
            return;
        };
        row.state = state;
        if let Some(result) = result {
            row.result = cap(&result, RESULT_CAP);
        }
    }
}

fn cap(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

impl Serialize for Snapshot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(8))?;
        map.serialize_entry("phase", &self.phase)?;
        map.serialize_entry("tray_state", &self.tray_state())?;
        map.serialize_entry("floor", &self.floor)?;
        map.serialize_entry("shortcut", &self.shortcut)?;
        map.serialize_entry("activity", &self.activity)?;
        map.serialize_entry("tasks", &self.tasks)?;
        map.serialize_entry("setup", &self.setup)?;
        map.serialize_entry("error", &self.error)?;
        map.end()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitDecision {
    Now,
    Wait(Duration),
}

#[derive(Debug)]
pub struct Coalescer {
    interval: Duration,
    last_emit: Option<Instant>,
    dirty: bool,
}

impl Coalescer {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_emit: None,
            dirty: false,
        }
    }

    pub fn changed(&mut self, now: Instant) -> EmitDecision {
        self.dirty = true;
        let Some(last_emit) = self.last_emit else {
            return EmitDecision::Now;
        };
        let elapsed = now.saturating_duration_since(last_emit);
        if elapsed >= self.interval {
            EmitDecision::Now
        } else {
            EmitDecision::Wait(self.interval - elapsed)
        }
    }

    pub fn mark_emitted(&mut self, now: Instant) {
        self.last_emit = Some(now);
        self.dirty = false;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}
