use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Floor {
    Idle,
    Listening,
    Speaking,
}

impl Floor {
    pub fn from_state(state: &str) -> Option<Self> {
        match state {
            "idle" => Some(Self::Idle),
            "listening" => Some(Self::Listening),
            "speaking" => Some(Self::Speaking),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub id: Option<String>,
    pub status: String,
    pub error: Option<String>,
    pub data: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub kind: String,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanelEvent {
    UserTranscript {
        text: String,
    },
    TextDelta {
        text: String,
    },
    TurnComplete {
        final_text: String,
        exit_reason: Option<String>,
        is_error: bool,
    },
    TaskRegistered {
        task_id: String,
        objective: String,
    },
    TaskCompleted {
        task_id: String,
        result: String,
    },
    TaskFailed {
        task_id: String,
        result: String,
    },
    TaskCancelled {
        task_id: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Response(Response),
    Event(Event),
    Malformed,
}

pub fn parse_line(line: &str) -> Incoming {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return Incoming::Malformed;
    };
    if let Some(status) = value.get("status").and_then(Value::as_str) {
        return Incoming::Response(Response {
            id: value.get("id").and_then(Value::as_str).map(str::to_string),
            status: status.to_string(),
            error: value
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_string),
            data: value.get("data").cloned().unwrap_or(Value::Null),
        });
    }
    if let Some(kind) = value.get("type").and_then(Value::as_str) {
        return Incoming::Event(Event {
            kind: kind.to_string(),
            payload: value.get("payload").cloned().unwrap_or(Value::Null),
        });
    }
    Incoming::Malformed
}

pub fn request_line(id: &str, action: &str) -> String {
    json!({"id": id, "action": action}).to_string()
}

pub fn floor_from_event(event: &Event) -> Option<Floor> {
    if event.kind != "runtime_event" {
        return None;
    }
    if event.payload.get("type")?.as_str()? != "floor_state" {
        return None;
    }
    Floor::from_state(event.payload.get("payload")?.get("state")?.as_str()?)
}

pub fn panel_event_from_event(event: &Event) -> Option<PanelEvent> {
    match event.kind.as_str() {
        "text_delta" => Some(PanelEvent::TextDelta {
            text: event.payload.get("text")?.as_str()?.to_string(),
        }),
        "turn_complete" => Some(PanelEvent::TurnComplete {
            final_text: event.payload.get("final_text")?.as_str()?.to_string(),
            exit_reason: event
                .payload
                .get("exit_reason")
                .and_then(Value::as_str)
                .map(str::to_string),
            is_error: event
                .payload
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }),
        "runtime_event" => {
            let kind = event.payload.get("type")?.as_str()?;
            let payload = event.payload.get("payload")?;
            match kind {
                "user_transcript" => Some(PanelEvent::UserTranscript {
                    text: payload.get("text")?.as_str()?.to_string(),
                }),
                "task_registered" => Some(PanelEvent::TaskRegistered {
                    task_id: payload.get("task_id")?.as_str()?.to_string(),
                    objective: payload.get("objective")?.as_str()?.to_string(),
                }),
                "task_completed" => Some(PanelEvent::TaskCompleted {
                    task_id: payload.get("task_id")?.as_str()?.to_string(),
                    result: payload.get("result")?.as_str()?.to_string(),
                }),
                "task_failed" => Some(PanelEvent::TaskFailed {
                    task_id: payload.get("task_id")?.as_str()?.to_string(),
                    result: payload.get("result")?.as_str()?.to_string(),
                }),
                "task_cancelled" => Some(PanelEvent::TaskCancelled {
                    task_id: payload.get("task_id")?.as_str()?.to_string(),
                }),
                _ => None,
            }
        }
        _ => None,
    }
}
