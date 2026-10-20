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

fn meaningful(value: &str) -> Option<String> {
    let value = unquote(value);
    if value.is_empty() || value.eq_ignore_ascii_case("null") || value == "~" {
        None
    } else {
        Some(value.to_string())
    }
}

fn dotenv_value(home: &Path, name: &str) -> Option<String> {
    let path = home.join(".config/verbix/.env");
    let contents = std::fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == name {
            let value = value.split(" #").next().unwrap_or(value);
            return meaningful(value);
        }
    }
    None
}

fn config_value(home: &Path, name: &str) -> Option<String> {
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
            let value = value.split(" #").next().unwrap_or(value);
            return meaningful(value);
        }
    }
    None
}

fn explicit_provider(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    env("VERBIX_PROVIDER")
        .as_deref()
        .and_then(meaningful)
        .or_else(|| dotenv_value(home, "VERBIX_PROVIDER"))
        .or_else(|| config_value(home, "provider"))
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
    if explicit_provider(home, env).is_some() {
        return None;
    }
    openrouter_key(home).map(|_| "openrouter".to_string())
}
