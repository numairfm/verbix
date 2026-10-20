use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::json;
use tempfile::tempdir;

use verbix_shell::command::{parse, ParsedCommand, Startup};
use verbix_shell::ipc::{
    floor_from_event, panel_event_from_event, parse_line, request_line, Event, Floor, Incoming,
    PanelEvent,
};
use verbix_shell::logging::redact_line;
use verbix_shell::provider::openrouter_override;
use verbix_shell::service::BackoffPolicy;
use verbix_shell::state::{Coalescer, EmitDecision, Phase, Snapshot, TaskState, TrayState};

fn write(home: &Path, relative: &str, contents: &str) {
    let path = home.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

#[test]
fn command_parses_supported_arguments() {
    assert_eq!(
        parse(Vec::<String>::new()),
        ParsedCommand::Run(Startup::Hidden)
    );
    assert_eq!(
        parse(["--hidden".to_string()]),
        ParsedCommand::Run(Startup::Hidden)
    );
    assert_eq!(
        parse(["--show".to_string()]),
        ParsedCommand::Run(Startup::Show)
    );
    assert_eq!(
        parse(["--toggle".to_string()]),
        ParsedCommand::Run(Startup::Toggle)
    );
    assert_eq!(parse(["--version".to_string()]), ParsedCommand::Version);
    assert_eq!(parse(["--help".to_string()]), ParsedCommand::Help);
}

#[test]
fn command_rejects_unknown_and_repeated_arguments() {
    assert!(matches!(
        parse(["--nope".to_string()]),
        ParsedCommand::Invalid(_)
    ));
    assert!(matches!(
        parse(["--show".to_string(), "--toggle".to_string()]),
        ParsedCommand::Invalid(_)
    ));
}

#[test]
fn parse_line_reads_a_ping_response() {
    let parsed = parse_line(r#"{"id":"ping-1","status":"ok","data":{"pong":true}}"#);
    match parsed {
        Incoming::Response(response) => {
            assert_eq!(response.id.as_deref(), Some("ping-1"));
            assert_eq!(response.status, "ok");
            assert_eq!(response.data["pong"], true);
        }
        other => panic!("expected response, got {other:?}"),
    }
}

#[test]
fn parse_line_reads_runtime_floor_event() {
    let parsed = parse_line(
        r#"{"type":"runtime_event","payload":{"type":"floor_state","payload":{"state":"speaking"}}}"#,
    );
    match parsed {
        Incoming::Event(event) => {
            assert_eq!(event.kind, "runtime_event");
            assert_eq!(floor_from_event(&event), Some(Floor::Speaking));
        }
        other => panic!("expected event, got {other:?}"),
    }
}

#[test]
fn parse_line_drops_malformed_and_unknown_shapes() {
    assert_eq!(parse_line("not json"), Incoming::Malformed);
    assert_eq!(parse_line(""), Incoming::Malformed);
    assert_eq!(parse_line(r#"{"hello":"world"}"#), Incoming::Malformed);
}

#[test]
fn request_line_is_compact_jsonl() {
    let line = request_line("toggle-4", "toggle");
    assert!(!line.contains('\n'));
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(value["id"], "toggle-4");
    assert_eq!(value["action"], "toggle");
}

#[test]
fn floor_from_event_ignores_other_events() {
    let event = Event {
        kind: "text_delta".to_string(),
        payload: json!({"text": "hello"}),
    };
    assert_eq!(floor_from_event(&event), None);
}

#[test]
fn provider_override_uses_openrouter_key_when_nothing_selects_a_provider() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":"secret"}}"#,
    );
    assert_eq!(
        openrouter_override(home.path(), &|_| None),
        Some("openrouter".to_string())
    );
}

#[test]
fn explicit_provider_wins_over_auth_key() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":"secret"}}"#,
    );
    assert_eq!(
        openrouter_override(home.path(), &|name| {
            (name == "VERBIX_PROVIDER").then(|| "openai".to_string())
        }),
        None
    );
    write(
        home.path(),
        ".config/verbix/.env",
        "VERBIX_PROVIDER=openai\n",
    );
    assert_eq!(openrouter_override(home.path(), &|_| None), None);
    fs::remove_file(home.path().join(".config/verbix/.env")).unwrap();
    write(home.path(), ".verbix/config.yaml", "provider: openai\n");
    assert_eq!(openrouter_override(home.path(), &|_| None), None);
}

#[test]
fn empty_provider_values_do_not_count_as_selection() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":"secret"}}"#,
    );
    write(
        home.path(),
        ".verbix/config.yaml",
        "provider: null\nstt_model: ~/.verbix/models/x.gguf\n",
    );
    assert_eq!(
        openrouter_override(home.path(), &|_| None),
        Some("openrouter".to_string())
    );
}

#[test]
fn missing_or_empty_auth_key_does_not_force_provider() {
    let home = tempdir().unwrap();
    assert_eq!(openrouter_override(home.path(), &|_| None), None);
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":""}}"#,
    );
    assert_eq!(openrouter_override(home.path(), &|_| None), None);
}

fn connected_snapshot() -> Snapshot {
    Snapshot::new(Phase::Connected, Floor::Idle, None)
}

fn event(kind: &str, payload: serde_json::Value) -> Event {
    Event {
        kind: kind.to_string(),
        payload,
    }
}

#[test]
fn snapshot_serializes_the_full_contract_with_inert_setup_defaults() {
    let value = serde_json::to_value(connected_snapshot()).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(object.len(), 8, "{value}");

    for key in [
        "phase",
        "tray_state",
        "floor",
        "shortcut",
        "activity",
        "tasks",
        "setup",
        "error",
    ] {
        assert!(object.contains_key(key), "missing {key}: {value}");
    }

    assert_eq!(value["phase"], "connected");
    assert_eq!(value["tray_state"], "idle");
    assert_eq!(value["floor"], "idle");
    assert_eq!(
        value["shortcut"],
        json!({"accelerator": null, "fallback": null})
    );
    assert_eq!(value["activity"], json!({"transcript": "", "caption": ""}));
    assert_eq!(value["tasks"], json!([]));
    assert_eq!(value["error"], serde_json::Value::Null);
    assert_eq!(
        value["setup"],
        json!({
            "required": false,
            "key_ready": true,
            "key_editable": true,
            "model_ready": true,
            "model_path": null,
            "model_downloadable": true,
            "autostart": false,
            "autostart_supported": cfg!(target_os = "linux"),
            "download": null,
            "error": null
        })
    );

    let text = value.to_string();
    assert!(!text.contains("provider_key"), "{text}");
    assert!(!text.contains("child_output"), "{text}");
}

#[test]
fn tray_state_precedence_follows_the_contract() {
    let starting = Snapshot::new(Phase::Starting, Floor::Idle, None);
    assert_eq!(starting.tray_state(), TrayState::Idle);
    assert_eq!(starting.tray_icon(), "idle");
    assert_eq!(starting.tooltip(), "Verbix — starting");

    for phase in [
        Phase::Restarting,
        Phase::Failed,
        Phase::Stopping,
        Phase::SetupRequired,
    ] {
        let mut snapshot = connected_snapshot();
        snapshot.phase = phase;
        snapshot.floor = Floor::Listening;
        assert_eq!(snapshot.tray_state(), TrayState::Error, "{phase:?}");
        assert_eq!(snapshot.tray_icon(), "error");
        assert_eq!(snapshot.tooltip(), "Verbix — error");
    }

    let listening = Snapshot::new(Phase::Connected, Floor::Listening, None);
    assert_eq!(listening.tray_state(), TrayState::Listening);
    assert_eq!(listening.tray_icon(), "listening");
    assert_eq!(listening.tooltip(), "Verbix — listening");

    let idle = Snapshot::new(Phase::Connected, Floor::Idle, None);
    assert_eq!(idle.tray_state(), TrayState::Idle);
    assert_eq!(idle.tray_icon(), "idle");
}

#[test]
fn tray_state_uses_working_while_a_task_runs_at_an_idle_floor() {
    let mut snapshot = connected_snapshot();
    snapshot.apply_panel_event(PanelEvent::TaskRegistered {
        task_id: "task-1".into(),
        objective: "summarize this repo".into(),
    });
    assert_eq!(snapshot.tray_state(), TrayState::Working);
    assert_eq!(snapshot.tray_icon(), "working");
    assert_eq!(snapshot.tooltip(), "Verbix — working");

    snapshot.floor = Floor::Speaking;
    assert_eq!(snapshot.tray_state(), TrayState::Speaking);
    snapshot.floor = Floor::Idle;

    snapshot.apply_panel_event(PanelEvent::TaskCompleted {
        task_id: "task-1".into(),
        result: "done".into(),
    });
    assert_eq!(snapshot.tray_state(), TrayState::Idle);
}

#[test]
fn tray_stays_working_when_a_running_row_falls_out_of_the_visible_cap() {
    let mut snapshot = connected_snapshot();
    snapshot.apply_panel_event(PanelEvent::TaskRegistered {
        task_id: "task-old".into(),
        objective: "long running".into(),
    });
    for index in 1..=20 {
        let task_id = format!("task-{index}");
        snapshot.apply_panel_event(PanelEvent::TaskRegistered {
            task_id: task_id.clone(),
            objective: "newer".into(),
        });
        snapshot.apply_panel_event(PanelEvent::TaskCompleted {
            task_id,
            result: "done".into(),
        });
    }

    assert_eq!(snapshot.tasks.len(), 20);
    assert!(!snapshot.tasks.iter().any(|row| row.id == "task-old"));
    assert_eq!(snapshot.tray_state(), TrayState::Working);

    snapshot.apply_panel_event(PanelEvent::TaskCompleted {
        task_id: "task-old".into(),
        result: "finally done".into(),
    });
    assert_eq!(snapshot.tray_state(), TrayState::Idle);
}

#[test]
fn activity_starts_new_turns_and_announcements_replace_finalized_captions() {
    let mut snapshot = connected_snapshot();

    snapshot.apply_panel_event(PanelEvent::UserTranscript {
        text: "first question".into(),
    });
    snapshot.apply_panel_event(PanelEvent::TextDelta {
        text: "First ".into(),
    });
    snapshot.apply_panel_event(PanelEvent::TextDelta {
        text: "answer".into(),
    });
    snapshot.apply_panel_event(PanelEvent::TurnComplete {
        final_text: "ignored because deltas arrived".into(),
        exit_reason: Some("completed".into()),
        is_error: false,
    });
    assert_eq!(snapshot.activity.transcript, "first question");
    assert_eq!(snapshot.activity.caption, "First answer");

    snapshot.apply_panel_event(PanelEvent::UserTranscript {
        text: "second question".into(),
    });
    assert_eq!(snapshot.activity.transcript, "second question");
    assert_eq!(snapshot.activity.caption, "");

    snapshot.apply_panel_event(PanelEvent::TurnComplete {
        final_text: "Second answer from final text".into(),
        exit_reason: Some("completed".into()),
        is_error: false,
    });
    assert_eq!(snapshot.activity.caption, "Second answer from final text");

    snapshot.apply_panel_event(PanelEvent::TextDelta {
        text: "Task one is complete".into(),
    });
    assert_eq!(snapshot.activity.caption, "Task one is complete");
    snapshot.apply_panel_event(PanelEvent::TextDelta {
        text: " and announced".into(),
    });
    assert_eq!(
        snapshot.activity.caption,
        "Task one is complete and announced"
    );
}

#[test]
fn snapshot_caps_text_and_keeps_the_newest_twenty_terminal_rows() {
    let mut snapshot = connected_snapshot();

    snapshot.apply_panel_event(PanelEvent::UserTranscript {
        text: "x".repeat(20_000),
    });
    assert_eq!(snapshot.activity.transcript.chars().count(), 16_000);

    for _ in 0..3 {
        snapshot.apply_panel_event(PanelEvent::TextDelta {
            text: "y".repeat(10_000),
        });
    }
    assert_eq!(snapshot.activity.caption.chars().count(), 16_000);

    for index in 1..=25 {
        snapshot.apply_panel_event(PanelEvent::TaskRegistered {
            task_id: format!("task-{index}"),
            objective: "o".repeat(500),
        });
    }
    assert_eq!(snapshot.tasks.len(), 20);
    assert_eq!(snapshot.tasks[0].id, "task-25");
    assert_eq!(snapshot.tasks[19].id, "task-6");
    assert_eq!(snapshot.tasks[0].objective.chars().count(), 200);

    snapshot.apply_panel_event(PanelEvent::TaskCompleted {
        task_id: "task-25".into(),
        result: "r".repeat(500),
    });
    assert_eq!(snapshot.tasks[0].state, TaskState::Completed);
    assert_eq!(snapshot.tasks[0].result.chars().count(), 280);

    snapshot.apply_panel_event(PanelEvent::TaskFailed {
        task_id: "task-24".into(),
        result: "worker failed".into(),
    });
    snapshot.apply_panel_event(PanelEvent::TaskCancelled {
        task_id: "task-23".into(),
    });
    assert_eq!(snapshot.tasks[1].state, TaskState::Failed);
    assert_eq!(snapshot.tasks[1].result, "worker failed");
    assert_eq!(snapshot.tasks[2].state, TaskState::Cancelled);
    assert_eq!(snapshot.tasks.len(), 20);

    snapshot.apply_panel_event(PanelEvent::TaskRegistered {
        task_id: "task-26".into(),
        objective: "newest".into(),
    });
    assert_eq!(snapshot.tasks.len(), 20);
    assert_eq!(snapshot.tasks[0].id, "task-26");
    assert!(snapshot
        .tasks
        .iter()
        .any(|row| row.id == "task-25" && row.state == TaskState::Completed));
}

#[test]
fn panel_decoder_reads_direct_streaming_events() {
    let delta = event("text_delta", json!({"text": "Hello"}));
    assert_eq!(
        panel_event_from_event(&delta),
        Some(PanelEvent::TextDelta {
            text: "Hello".into()
        })
    );

    let done = event(
        "turn_complete",
        json!({
            "final_text": "Hello there.",
            "exit_reason": "completed",
            "is_error": false
        }),
    );
    assert_eq!(
        panel_event_from_event(&done),
        Some(PanelEvent::TurnComplete {
            final_text: "Hello there.".into(),
            exit_reason: Some("completed".into()),
            is_error: false,
        })
    );
}

#[test]
fn panel_decoder_reads_runtime_transcript_and_task_events() {
    let cases = [
        (
            event(
                "runtime_event",
                json!({"type": "user_transcript", "payload": {"text": "what time is it", "duration": 1.5}}),
            ),
            Some(PanelEvent::UserTranscript {
                text: "what time is it".into(),
            }),
        ),
        (
            event(
                "runtime_event",
                json!({"type": "task_registered", "payload": {"task_id": "task-1", "objective": "summarize this repo"}}),
            ),
            Some(PanelEvent::TaskRegistered {
                task_id: "task-1".into(),
                objective: "summarize this repo".into(),
            }),
        ),
        (
            event(
                "runtime_event",
                json!({"type": "task_completed", "payload": {"task_id": "task-1", "result": "done"}}),
            ),
            Some(PanelEvent::TaskCompleted {
                task_id: "task-1".into(),
                result: "done".into(),
            }),
        ),
        (
            event(
                "runtime_event",
                json!({"type": "task_failed", "payload": {"task_id": "task-2", "result": "boom"}}),
            ),
            Some(PanelEvent::TaskFailed {
                task_id: "task-2".into(),
                result: "boom".into(),
            }),
        ),
        (
            event(
                "runtime_event",
                json!({"type": "task_cancelled", "payload": {"task_id": "task-3"}}),
            ),
            Some(PanelEvent::TaskCancelled {
                task_id: "task-3".into(),
            }),
        ),
    ];

    for (event, expected) in cases {
        assert_eq!(panel_event_from_event(&event), expected, "{event:?}");
    }
}

#[test]
fn panel_decoder_ignores_other_events_and_incomplete_payloads() {
    for event in [
        event(
            "runtime_event",
            json!({"type": "floor_state", "payload": {"state": "speaking"}}),
        ),
        event("wake_event", json!({"text": "duplicate"})),
        event("thought_delta", json!({"text": "private"})),
        event(
            "runtime_event",
            json!({"type": "task_registered", "payload": {"task_id": "task-1"}}),
        ),
        event("text_delta", json!({})),
    ] {
        assert_eq!(panel_event_from_event(&event), None, "{event:?}");
    }
}

#[test]
fn coalescer_emits_first_change_then_at_most_once_per_interval_with_the_final_state() {
    let base = Instant::now();
    let interval = Duration::from_millis(50);
    let mut coalescer = Coalescer::new(interval);

    assert_eq!(coalescer.changed(base), EmitDecision::Now);
    coalescer.mark_emitted(base);
    assert!(!coalescer.is_dirty());

    assert_eq!(
        coalescer.changed(base + Duration::from_millis(10)),
        EmitDecision::Wait(Duration::from_millis(40))
    );
    assert!(coalescer.is_dirty());
    assert_eq!(
        coalescer.changed(base + Duration::from_millis(30)),
        EmitDecision::Wait(Duration::from_millis(20))
    );
    assert_eq!(
        coalescer.changed(base + Duration::from_millis(50)),
        EmitDecision::Now
    );
    coalescer.mark_emitted(base + Duration::from_millis(50));
    assert!(!coalescer.is_dirty());

    assert_eq!(
        coalescer.changed(base + Duration::from_millis(75)),
        EmitDecision::Wait(Duration::from_millis(25))
    );
    assert_eq!(
        coalescer.changed(base + Duration::from_millis(100)),
        EmitDecision::Now
    );
    coalescer.mark_emitted(base + Duration::from_millis(100));
    assert!(!coalescer.is_dirty());
}

#[test]
fn panel_script_renders_with_text_content_only() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("panel/panel.js");
    let script = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));

    assert!(script.contains("textContent"), "{script}");
    assert!(!script.contains("innerHTML"), "{script}");
    assert!(!script.contains("insertAdjacentHTML"), "{script}");
    assert!(script.contains("shell://state"), "{script}");
    assert!(script.contains("get_state"), "{script}");
    assert!(script.contains("Escape"), "{script}");
    assert!(script.contains("hide_panel"), "{script}");
    assert!(script.contains("restart_service"), "{script}");
}

#[test]
fn panel_markup_is_offline_and_uses_external_script_only() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("panel/index.html");
    let html = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));

    assert!(
        html.contains(r#"<script src="panel.js"></script>"#),
        "{html}"
    );
    assert!(!html.contains("<script>"), "{html}");
    assert!(!html.contains("<input"), "{html}");
    assert!(!html.contains("http://"), "{html}");
    assert!(!html.contains("https://"), "{html}");
    assert!(html.contains(">Hide<"), "{html}");
    assert!(html.contains(">Retry<"), "{html}");
}

#[test]
fn backoff_uses_the_published_delays_then_gives_up() {
    let policy = BackoffPolicy::default();
    assert_eq!(policy.delay_for_restart(0), Some(Duration::from_secs(1)));
    assert_eq!(policy.delay_for_restart(1), Some(Duration::from_secs(2)));
    assert_eq!(policy.delay_for_restart(2), Some(Duration::from_secs(5)));
    assert_eq!(policy.delay_for_restart(3), Some(Duration::from_secs(10)));
    assert_eq!(policy.delay_for_restart(4), Some(Duration::from_secs(30)));
    assert_eq!(policy.delay_for_restart(5), None);
}

#[test]
fn redaction_removes_keys_and_authorization_values() {
    for line in [
        "OPENROUTER_API_KEY=sk-abc123",
        "api_key=sk-abc123",
        "Authorization: Bearer abc123",
        "token is sk-abc123",
    ] {
        let redacted = redact_line(line);
        assert!(!redacted.contains("sk-abc123"), "{redacted}");
        assert!(!redacted.contains("abc123"), "{redacted}");
    }
    assert_eq!(redact_line("Verbix — starting"), "Verbix — starting");
}
