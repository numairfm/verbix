use std::fs;
use std::path::Path;

use tempfile::tempdir;
use verbix_shell::app_state::startup_plan;
use verbix_shell::command::Startup;
use verbix_shell::setup::{
    auth_path, detect, marker_complete, marker_path, validate_key, write_marker, write_provider_key,
};

fn write(home: &Path, relative: &str, contents: &str) {
    let path = home.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

fn none_env(_name: &str) -> Option<String> {
    None
}

fn model_file(home: &Path) {
    let path = home.join(".verbix/voice/stt/models/nemotron-gguf/model.gguf");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"model").unwrap();
}

#[test]
fn a_fresh_home_is_incomplete_and_offers_the_key_and_the_model() {
    let home = tempdir().unwrap();
    let detection = detect(home.path(), &none_env);

    assert!(!detection.marker_complete);
    assert!(!detection.complete());
    assert!(!detection.key.ready);
    assert!(detection.key.editable);
    assert_eq!(detection.key.selected_provider, None);
    assert!(!detection.model.ready);
    assert!(detection.model.downloadable);
    assert!(!detection.model.authoritative);
    assert!(detection
        .model
        .path
        .as_ref()
        .unwrap()
        .starts_with(home.path().join(".verbix/voice")));
}

#[test]
fn an_existing_key_and_model_are_detected_and_the_key_is_not_rewritten() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":"sk-existing","model":"m","base_url":"u"}}"#,
    );
    model_file(home.path());
    write_marker(home.path()).unwrap();

    let detection = detect(home.path(), &none_env);
    assert!(detection.key.ready);
    assert!(detection.model.ready);
    assert!(detection.complete());

    let before = fs::read_to_string(auth_path(home.path())).unwrap();
    let _ = detect(home.path(), &none_env);
    assert_eq!(fs::read_to_string(auth_path(home.path())).unwrap(), before);
}

#[test]
fn a_key_in_the_environment_or_the_environment_file_counts_as_ready() {
    let home = tempdir().unwrap();
    model_file(home.path());
    let detection = detect(home.path(), &|name| {
        (name == "OPENROUTER_API_KEY").then(|| "sk-env".to_string())
    });
    assert!(detection.key.ready);

    write(
        home.path(),
        ".config/verbix/.env",
        "VERBIX_API_KEY=sk-file\n",
    );
    let detection = detect(home.path(), &none_env);
    assert!(detection.key.ready);
}

#[test]
fn a_selected_provider_without_a_key_shows_the_credential_path() {
    let home = tempdir().unwrap();
    write(home.path(), ".verbix/config.yaml", "provider: openai\n");
    model_file(home.path());

    let detection = detect(home.path(), &none_env);
    assert!(!detection.key.ready);
    assert!(!detection.key.editable);
    let error = detection
        .key
        .error
        .expect("an error names the credential file");
    assert!(error.contains("openai"), "{error}");
    assert!(
        error.contains(&auth_path(home.path()).display().to_string()),
        "{error}"
    );
}

#[test]
fn writing_the_key_preserves_other_entries_and_fields() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openai":{"type":"api","key":"other"},"openrouter":{"type":"api","key":"old","model":"m","base_url":"u"}}"#,
    );

    write_provider_key(home.path(), "sk-new").unwrap();

    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(auth_path(home.path())).unwrap()).unwrap();
    assert_eq!(value["openrouter"]["key"], "sk-new");
    assert_eq!(value["openrouter"]["model"], "m");
    assert_eq!(value["openrouter"]["base_url"], "u");
    assert_eq!(value["openrouter"]["type"], "api");
    assert_eq!(value["openai"]["key"], "other");
}

#[test]
fn writing_the_key_creates_a_private_state_folder_and_file() {
    let home = tempdir().unwrap();
    write_provider_key(home.path(), "sk-new").unwrap();

    let file = auth_path(home.path());
    let metadata = fs::symlink_metadata(&file).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        let folder = fs::symlink_metadata(home.path().join(".verbix")).unwrap();
        assert_eq!(folder.permissions().mode() & 0o777, 0o700);
    }
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(value["openrouter"]["key"], "sk-new");
    assert_eq!(value["openrouter"]["type"], "api");
}

#[cfg(unix)]
#[test]
fn writing_the_key_refuses_to_follow_a_symlink() {
    let home = tempdir().unwrap();
    let target = home.path().join("elsewhere.json");
    fs::write(&target, b"{}").unwrap();
    fs::create_dir_all(home.path().join(".verbix")).unwrap();
    std::os::unix::fs::symlink(&target, auth_path(home.path())).unwrap();

    let error = write_provider_key(home.path(), "sk-new").unwrap_err();
    assert!(error.contains("symlink"), "{error}");
    assert_eq!(fs::read_to_string(&target).unwrap(), "{}");
}

#[test]
fn the_key_is_rejected_when_it_is_empty_long_or_has_a_newline() {
    assert!(validate_key("").is_err());
    assert!(validate_key("   ").is_err());
    assert!(validate_key("sk-ok").is_ok());
    assert!(validate_key(&"x".repeat(4096)).is_ok());
    assert!(validate_key(&"x".repeat(4097)).is_err());
    assert!(validate_key("sk-one\nsk-two").is_err());
}

#[test]
fn a_configured_model_path_that_is_missing_blocks_the_step() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/config.yaml",
        "stt_model: /nowhere/model.gguf\n",
    );

    let detection = detect(home.path(), &none_env);
    assert!(!detection.model.ready);
    assert!(detection.model.authoritative);
    assert!(!detection.model.downloadable);
    assert_eq!(
        detection.model.path.as_ref().unwrap().to_string_lossy(),
        "/nowhere/model.gguf"
    );
    assert!(detection
        .model
        .error
        .as_ref()
        .unwrap()
        .contains("/nowhere/model.gguf"));
}

#[test]
fn a_relative_model_path_resolves_against_the_child_working_directory() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/config.yaml",
        "stt_model: models/voice.gguf\n",
    );
    let work = home.path().join(".config/verbix/models");
    fs::create_dir_all(&work).unwrap();
    fs::write(work.join("voice.gguf"), b"model").unwrap();

    let detection = detect(home.path(), &none_env);
    assert!(detection.model.ready);
    assert!(detection.model.authoritative);
    assert_eq!(
        detection.model.path.as_ref().unwrap(),
        &home.path().join(".config/verbix/models/voice.gguf")
    );
}

#[test]
fn an_environment_model_path_is_authoritative_too() {
    let home = tempdir().unwrap();
    let detection = detect(home.path(), &|name| {
        (name == "VERBIX_STT_MODEL").then(|| "/nowhere/env.gguf".to_string())
    });
    assert!(!detection.model.ready);
    assert!(detection.model.authoritative);
    assert!(!detection.model.downloadable);
}

#[test]
fn a_model_outside_the_voice_root_cannot_be_re_downloaded() {
    let home = tempdir().unwrap();
    let path = home.path().join(".verbix/models/legacy.gguf");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"model").unwrap();

    let detection = detect(home.path(), &none_env);
    assert!(detection.model.ready);
    assert!(!detection.model.downloadable);
    assert_eq!(detection.model.path.as_ref().unwrap(), &path);
}

#[test]
fn a_model_in_the_voice_root_can_be_re_downloaded() {
    let home = tempdir().unwrap();
    model_file(home.path());

    let detection = detect(home.path(), &none_env);
    assert!(detection.model.ready);
    assert!(detection.model.downloadable);
}

#[test]
fn a_missing_or_corrupt_marker_means_the_setup_is_incomplete() {
    let home = tempdir().unwrap();
    assert!(!marker_complete(home.path()));

    write(
        home.path(),
        ".verbix/shell.json",
        "{\"setup_complete\": true}",
    );
    assert!(marker_complete(home.path()));

    write(home.path(), ".verbix/shell.json", "not json");
    assert!(!marker_complete(home.path()));

    write(
        home.path(),
        ".verbix/shell.json",
        "{\"setup_complete\": false}",
    );
    assert!(!marker_complete(home.path()));
}

#[test]
fn the_marker_is_written_atomically_with_a_private_mode() {
    let home = tempdir().unwrap();
    write_marker(home.path()).unwrap();

    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(marker_path(home.path())).unwrap()).unwrap();
    assert_eq!(value["setup_complete"], true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = fs::symlink_metadata(marker_path(home.path())).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    }
}

#[test]
fn the_global_models_folder_wins_over_the_voice_folder() {
    let home = tempdir().unwrap();
    let legacy = home.path().join(".verbix/models/legacy.gguf");
    fs::create_dir_all(legacy.parent().unwrap()).unwrap();
    fs::write(&legacy, b"model").unwrap();
    model_file(home.path());

    let detection = detect(home.path(), &none_env);
    assert_eq!(detection.model.path.as_ref().unwrap(), &legacy);
}

#[test]
fn an_incomplete_setup_opens_the_panel_and_keeps_the_service_away() {
    let home = tempdir().unwrap();
    let detection = detect(home.path(), &none_env);

    for startup in [Startup::Hidden, Startup::Show, Startup::Toggle] {
        let plan = startup_plan(&detection, startup);
        assert!(!plan.start_service, "{startup:?}");
        assert!(plan.show_panel, "{startup:?}");
        assert!(!plan.startup_toggle, "{startup:?}");
    }
}

#[test]
fn a_complete_setup_starts_the_service_and_hides_the_panel_by_default() {
    let home = tempdir().unwrap();
    write(
        home.path(),
        ".verbix/auth.json",
        r#"{"openrouter":{"type":"api","key":"sk-key"}}"#,
    );
    model_file(home.path());
    write_marker(home.path()).unwrap();
    let detection = detect(home.path(), &none_env);
    assert!(detection.complete());

    let hidden = startup_plan(&detection, Startup::Hidden);
    assert!(hidden.start_service);
    assert!(!hidden.show_panel);
    assert!(!hidden.startup_toggle);

    let show = startup_plan(&detection, Startup::Show);
    assert!(show.start_service);
    assert!(show.show_panel);

    let toggle = startup_plan(&detection, Startup::Toggle);
    assert!(toggle.start_service);
    assert!(!toggle.show_panel);
    assert!(toggle.startup_toggle);
}
