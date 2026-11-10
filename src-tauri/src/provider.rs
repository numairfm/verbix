use std::path::Path;

use serde_json::Value;

fn unquote(value: &str) -> &str {
    let value = value.trim();
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'')
        {
            return &value[1..value.len() - 1];
        }
    }
    value
}

/// A value that counts as set. `null` and `~` are the two spellings the
/// Python side treats as "not configured".
pub fn meaningful(value: &str) -> Option<String> {
    let value = unquote(value);
    if value.is_empty() || value.eq_ignore_ascii_case("null") || value == "~" {
        None
    } else {
        Some(value.to_string())
    }
}

/// Read one variable from the child's environment file. The parsing follows the
/// Python `load_env` rules: the first `=` splits the line, the key is trimmed,
/// and the value is trimmed and unquoted.
pub fn dotenv_value(home: &Path, name: &str) -> Option<String> {
    let path = home.join(".config/verbix/.env");
    let contents = std::fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != name {
            continue;
        }
        let value = value.trim().trim_matches(['\'', '"']);
        if value.is_empty() {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

/// Read one top-level scalar from the global config file. The rest of the
/// document is left to Python.
pub fn config_value(home: &Path, name: &str) -> Option<String> {
    let path = home.join(".verbix/config.yaml");
    let contents = std::fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        if line.starts_with(char::is_whitespace) || line.trim_start().starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() == name {
            return meaningful(value);
        }
    }
    None
}

/// A variable from the process environment, falling back to the child's
/// environment file. The process environment always wins.
pub fn env_value(home: &Path, env: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env(name)
        .as_deref()
        .and_then(meaningful)
        .or_else(|| dotenv_value(home, name))
}

/// The provider the user selected: environment first, then the environment
/// file, then the config file.
pub fn selected_provider(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    env_value(home, env, "VERBIX_PROVIDER").or_else(|| config_value(home, "provider"))
}

/// Read `auth.json`. A missing file is not an error; an unreadable or invalid
/// one is, so the panel can refuse to overwrite it.
pub fn read_auth(home: &Path) -> Result<Option<Value>, String> {
    let path = home.join(".verbix/auth.json");
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Cannot read {}: {error}", path.display())),
    };
    let value: Value = serde_json::from_str(&contents)
        .map_err(|error| format!("Invalid JSON in {}: {error}", path.display()))?;
    if !value.is_object() {
        return Err(format!(
            "Invalid auth config in {}: expected a mapping",
            path.display()
        ));
    }
    Ok(Some(value))
}

/// The key of one provider entry, when the entry is usable: a mapping with
/// `type: api` and a non-empty `key`.
pub fn provider_key(home: &Path, provider: &str) -> Result<Option<String>, String> {
    let Some(auth) = read_auth(home)? else {
        return Ok(None);
    };
    let Some(entry) = auth.get(provider) else {
        return Ok(None);
    };
    let Some(entry) = entry.as_object() else {
        return Err(format!(
            "Invalid entry for provider '{provider}' in {}: expected a mapping",
            home.join(".verbix/auth.json").display()
        ));
    };
    match entry.get("type").and_then(Value::as_str) {
        Some("api") => {}
        Some(other) => {
            return Err(format!(
                "Invalid entry for provider '{provider}' in {}: 'type' must be \"api\", found \"{other}\"",
                home.join(".verbix/auth.json").display()
            ))
        }
        None => {
            return Err(format!(
                "Invalid entry for provider '{provider}' in {}: 'type' must be \"api\"",
                home.join(".verbix/auth.json").display()
            ))
        }
    }
    match entry.get("key").and_then(Value::as_str) {
        Some(key) if !key.trim().is_empty() => Ok(Some(key.to_string())),
        _ => Ok(None),
    }
}

fn openrouter_key(home: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(home.join(".verbix/auth.json")).ok()?;
    let auth: Value = serde_json::from_str(&contents).ok()?;
    let key = auth.get("openrouter")?.get("key")?.as_str()?;
    let key = key.trim();
    if key.is_empty() {
        None
    } else {
        Some(key.to_string())
    }
}

/// Returns the provider value the shell must add for the child, if any.
///
/// The OpenRouter key is usable without editing config: the shell selects that
/// provider only when nothing else selects one.
pub fn openrouter_override(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    if selected_provider(home, env).is_some() {
        return None;
    }
    openrouter_key(home).map(|_| "openrouter".to_string())
}
