use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use tempfile::tempdir;

use verbix_shell::command::{parse, ParsedCommand, Startup};
use verbix_shell::ipc::{floor_from_event, parse_line, request_line, Event, Floor, Incoming};
use verbix_shell::logging::redact_line;
use verbix_shell::provider::openrouter_override;
use verbix_shell::service::BackoffPolicy;
use verbix_shell::state::{Phase, Snapshot};

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

#[test]
fn tray_state_precedence_follows_the_contract() {
    let starting = Snapshot::new(Phase::Starting, Floor::Idle, None);
    assert_eq!(starting.tray_icon(), "idle");
    assert_eq!(starting.tooltip(), "Verbix — starting");

    let failed = Snapshot::new(Phase::Failed, Floor::Idle, Some("boom".into()));
    assert_eq!(failed.tray_icon(), "error");

    let listening = Snapshot::new(Phase::Connected, Floor::Listening, None);
    assert_eq!(listening.tray_icon(), "listening");
    assert_eq!(listening.tooltip(), "Verbix — listening");

    let idle = Snapshot::new(Phase::Connected, Floor::Idle, None);
    assert_eq!(idle.tray_icon(), "idle");
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
