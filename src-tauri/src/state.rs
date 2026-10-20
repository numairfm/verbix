use serde::Serialize;

use crate::ipc::Floor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Starting,
    Connected,
    Restarting,
    Failed,
    Stopping,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Connected => "connected",
            Self::Restarting => "restarting",
            Self::Failed => "failed",
            Self::Stopping => "stopping",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    pub phase: Phase,
    pub floor: Floor,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Snapshot {
    pub fn new(phase: Phase, floor: Floor, detail: Option<String>) -> Self {
        Self {
            phase,
            floor,
            detail,
        }
    }

    pub fn tray_icon(&self) -> &'static str {
        match self.phase {
            Phase::Starting => "idle",
            Phase::Connected => match self.floor {
                Floor::Idle => "idle",
                Floor::Listening => "listening",
                Floor::Speaking => "speaking",
            },
            Phase::Restarting | Phase::Failed | Phase::Stopping => "error",
        }
    }

    pub fn tooltip(&self) -> String {
        if self.phase == Phase::Starting {
            return "Verbix — starting".to_string();
        }
        format!("Verbix — {}", self.tray_icon())
    }
}
