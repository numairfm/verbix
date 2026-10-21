use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::provider;

pub const MARKER_FILE: &str = "shell.json";
pub const AUTH_FILE: &str = "auth.json";
pub const MAX_KEY_BYTES: usize = 4096;
pub const DEFAULT_MODEL: &str =
    "stt/models/nemotron-gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf";
pub const OPENROUTER_MODEL: &str = "openrouter/google/gemini-3.5-flash-lite";
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// The global state folder, `~/.verbix`.
pub fn state_dir(home: &Path) -> PathBuf {
    home.join(".verbix")
}

/// The setup marker, `~/.verbix/shell.json`.
pub fn marker_path(home: &Path) -> PathBuf {
    state_dir(home).join(MARKER_FILE)
}

/// The credential file, `~/.verbix/auth.json`.
pub fn auth_path(home: &Path) -> PathBuf {
    state_dir(home).join(AUTH_FILE)
}

/// The child's working directory, `~/.config/verbix`.
pub fn work_dir(home: &Path) -> PathBuf {
    home.join(".config/verbix")
}

/// The voice root the shell pins for the child, `~/.verbix/voice`.
pub fn voice_dir(home: &Path) -> PathBuf {
    home.join(".verbix/voice")
}

/// What the setup step knows about the provider key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyStatus {
    pub ready: bool,
    pub editable: bool,
    pub selected_provider: Option<String>,
    pub error: Option<String>,
}

/// What the setup step knows about the speech model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStatus {
    pub ready: bool,
    pub path: Option<PathBuf>,
    /// A config or environment path is authoritative: the shell never
    /// downloads over it.
    pub authoritative: bool,
    /// The model can be downloaded or re-downloaded by the bundled script.
    pub downloadable: bool,
    pub error: Option<String>,
}

/// The result of one setup inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detection {
    pub marker_complete: bool,
    pub key: KeyStatus,
    pub model: ModelStatus,
}

impl Detection {
    /// Setup is complete when the marker exists and the key and model are
    /// usable. The voice service only starts on a complete setup.
    pub fn complete(&self) -> bool {
        self.marker_complete && self.key.ready && self.model.ready
    }
}

/// Read the setup marker. A missing or corrupt marker means "not complete".
pub fn marker_complete(home: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(marker_path(home)) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(&contents) else {
        return false;
    };
    value.get("setup_complete").and_then(Value::as_bool) == Some(true)
}

/// Inspect the provider key. A ready key is never rewritten.
pub fn detect_key(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> KeyStatus {
    let selected = provider::selected_provider(home, env);
    let editable = matches!(selected.as_deref(), None | Some("openrouter"));
    let env_key = ["OPENROUTER_API_KEY", "VERBIX_API_KEY"]
        .iter()
        .any(|name| provider::env_value(home, env, name).is_some());

    let auth_key = match selected.as_deref() {
        Some(name) => provider::provider_key(home, name),
        None => provider::provider_key(home, "openrouter"),
    };
    let (auth_key, auth_error) = match auth_key {
        Ok(key) => (key, None),
        Err(error) => (None, Some(error)),
    };

    let ready = env_key || auth_key.is_some();
    let error = match (&selected, auth_error) {
        (Some(name), Some(error)) if !ready => Some(error),
        (Some(name), _) if !ready && name != "openrouter" => Some(format!(
            "Provider '{name}' has no key. Add it to {}.",
            auth_path(home).display()
        )),
        _ => None,
    };

    KeyStatus {
        ready,
        editable,
        selected_provider: selected,
        error,
    }
}

/// Resolve the speech model the same way Python does: the config scalar
/// first, then the environment, then the default roots.
pub fn detect_model(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> ModelStatus {
    let work = work_dir(home);
    let voice = voice_dir(home);

    let authoritative = provider::config_value(home, "stt_model")
        .or_else(|| provider::env_value(home, env, "VERBIX_STT_MODEL"));
    if let Some(raw) = authoritative {
        let path = resolve_path(&work, &raw);
        let ready = path.is_file();
        return ModelStatus {
            ready,
            path: Some(path.clone()),
            authoritative: true,
            downloadable: false,
            error: (!ready).then(|| format!("Speech model not found at {}.", path.display())),
        };
    }

    if let Some(path) = find_default_model(home) {
        let downloadable = path.starts_with(&voice);
        return ModelStatus {
            ready: true,
            path: Some(path),
            authoritative: false,
            downloadable,
            error: None,
        };
    }

    let path = voice.join(DEFAULT_MODEL);
    ModelStatus {
        ready: false,
        path: Some(path.clone()),
        authoritative: false,
        downloadable: true,
        error: Some(format!("Speech model not found at {}.", path.display())),
    }
}

/// A relative path from the config or the environment resolves against the
/// child's working directory, like the Python side does.
fn resolve_path(work: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        work.join(path)
    }
}

/// The first speech model under the default roots, in the Python order.
fn find_default_model(home: &Path) -> Option<PathBuf> {
    let voice = voice_dir(home);
    let legacy = home.join(".verbix/models");
    let stt_models = voice.join("stt/models");
    let legacy_models = voice.join("models");

    if let Some(path) = first_gguf(&legacy) {
        return Some(path);
    }
    if let Some(path) = first_gguf(&stt_models.join("nemotron-gguf")) {
        return Some(path);
    }
    if let Some(path) = first_gguf(&stt_models) {
        return Some(path);
    }
    if let Some(path) = first_gguf(&legacy_models.join("nemotron-gguf")) {
        return Some(path);
    }
    if let Some(path) = first_gguf(&legacy_models) {
        return Some(path);
    }
    None
}

/// The first `*.gguf` file under a directory, sorted by path.
fn first_gguf(root: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    collect_gguf(root, &mut found);
    found.sort();
    found.into_iter().next()
}

fn collect_gguf(root: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, found);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("gguf")
            && path.is_file()
        {
            found.push(path);
        }
    }
}

/// Inspect the whole setup surface.
pub fn detect(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Detection {
    let key = detect_key(home, env);
    let model = detect_model(home, env);
    Detection {
        marker_complete: marker_complete(home),
        key,
        model,
    }
}

/// Reject a key the shell must not write.
pub fn validate_key(key: &str) -> Result<(), String> {
    if key.trim().is_empty() {
        return Err("The provider key cannot be empty.".to_string());
    }
    if key.len() > MAX_KEY_BYTES {
        return Err(format!(
            "The provider key cannot be longer than {MAX_KEY_BYTES} bytes."
        ));
    }
    if key.contains('\n') || key.contains('\r') {
        return Err("The provider key cannot contain a newline.".to_string());
    }
    Ok(())
}

/// Write or update the OpenRouter key, preserving every other entry and
/// field. The write goes through a same-directory temporary file.
pub fn write_provider_key(home: &Path, key: &str) -> Result<(), String> {
    validate_key(key)?;
    let path = auth_path(home);
    refuse_symlink(&path)?;

    let mut auth = match provider::read_auth(home)? {
        Some(value) => value.as_object().cloned().unwrap_or_default(),
        None => Map::new(),
    };
    match auth.get("openrouter") {
        Some(Value::Object(entry)) => {
            let mut entry = entry.clone();
            entry.insert("key".to_string(), Value::String(key.to_string()));
            auth.insert("openrouter".to_string(), Value::Object(entry));
        }
        Some(_) => {
            return Err(format!(
                "Invalid entry for provider 'openrouter' in {}: expected a mapping",
                path.display()
            ))
        }
        None => {
            auth.insert(
                "openrouter".to_string(),
                json!({
                    "type": "api",
                    "key": key,
                    "model": OPENROUTER_MODEL,
                    "base_url": OPENROUTER_BASE_URL,
                }),
            );
        }
    }

    let mut contents = serde_json::to_string_pretty(&Value::Object(auth))
        .map_err(|error| format!("Cannot serialise {}: {error}", path.display()))?;
    contents.push('\n');
    write_atomic(&path, contents.as_bytes())
}

/// Write the setup marker. A missing or corrupt marker is overwritten.
pub fn write_marker(home: &Path) -> Result<(), String> {
    let path = marker_path(home);
    refuse_symlink(&path)?;
    let contents = serde_json::to_string_pretty(&json!({"setup_complete": true}))
        .map_err(|error| format!("Cannot serialise {}: {error}", path.display()))?;
    write_atomic(&path, contents.as_bytes())
}

/// Refuse to follow a symlink at the target path.
fn refuse_symlink(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "Refusing to write {}: the path is a symlink.",
            path.display()
        )),
        Ok(metadata) if !metadata.is_file() => Err(format!(
            "Refusing to write {}: the path is not a regular file.",
            path.display()
        )),
        _ => Ok(()),
    }
}

/// Write a file through a same-directory temporary file, then rename it over
/// the target. The state folder is created as `0700`, the file as `0600`.
fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    create_private_dir(parent)?;

    let temp = temp_path(path);
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp)
            .map_err(|error| format!("Cannot write {}: {error}", temp.display()))?;
        file.write_all(contents)
            .map_err(|error| format!("Cannot write {}: {error}", temp.display()))?;
        file.sync_all()
            .map_err(|error| format!("Cannot write {}: {error}", temp.display()))?;
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temp, fs::Permissions::from_mode(0o600))
                .map_err(|error| format!("Cannot write {}: {error}", temp.display()))?;
        }
        fs::rename(&temp, path)
            .map_err(|error| format!("Cannot replace {}: {error}", path.display()))
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Create a directory that only the user can read.
pub fn create_private_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("Cannot create {}: {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
    }
    Ok(())
}

fn temp_path(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{nanos}-{counter}", std::process::id()));
    path.with_file_name(name)
}
