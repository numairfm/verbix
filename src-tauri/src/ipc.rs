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
