use std::io;
use std::path::{Path, PathBuf};

pub const ENTRY_FILE: &str = "verbix.desktop";

/// Live state of the desktop entry. A missing or unreadable entry is `Disabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Enabled,
    Disabled,
    Unsupported,
}

/// What the tray check item should show for a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MenuState {
    pub enabled: bool,
    pub checked: bool,
}

pub fn menu_state(status: Status) -> MenuState {
    match status {
        Status::Enabled => MenuState {
            enabled: true,
            checked: true,
        },
        Status::Disabled => MenuState {
            enabled: true,
            checked: false,
        },
        Status::Unsupported => MenuState {
            enabled: false,
            checked: false,
        },
    }
}

/// Path of the entry. `$XDG_CONFIG_HOME/autostart/verbix.desktop`, falling back
/// to `~/.config/autostart/verbix.desktop`. Relative XDG paths are ignored per
/// the XDG Base Directory specification.
pub fn entry_path(home: &Path) -> PathBuf {
    let xdg = std::env::var_os("XDG_CONFIG_HOME");
    entry_path_with(home, xdg.as_deref().map(Path::new))
}

pub fn entry_path_with(home: &Path, xdg_config_home: Option<&Path>) -> PathBuf {
    match xdg_config_home.filter(|path| path.is_absolute()) {
        Some(config_home) => config_home.join("autostart").join(ENTRY_FILE),
        None => home.join(".config/autostart").join(ENTRY_FILE),
    }
}

/// The executable to put in the entry: `$APPIMAGE` when it names a real file,
/// otherwise the running executable. A mounted AppImage runtime (`.../.mount_*`)
/// is never preferred over the resolved AppImage path.
pub fn executable_path() -> PathBuf {
    crate::platform::executable_path()
}

pub fn executable_path_with(appimage: Option<&Path>, current_exe: PathBuf) -> PathBuf {
    crate::platform::executable_path_with(appimage, current_exe)
}

/// Render the XDG desktop entry. Quotes the executable, escapes the characters
/// the desktop-entry grammar reserves inside quoted arguments, and doubles `%`
/// so it is not read as a field code.
pub fn render_entry(executable: &Path) -> io::Result<String> {
    let path = executable.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "autostart executable path is not valid UTF-8",
        )
    })?;
    Ok(format!(
        concat!(
            "[Desktop Entry]\n",
            "Type=Application\n",
            "Name=Verbix\n",
            "Comment=Voice-operated desktop assistant\n",
            "Exec=\"{}\" --hidden\n",
            "Terminal=false\n",
            "X-GNOME-Autostart-enabled=true\n",
        ),
        escape_exec(path)
    ))
}

fn escape_exec(path: &str) -> String {
    let mut escaped = String::with_capacity(path.len());
    for character in path.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '`' => escaped.push_str("\\`"),
            '$' => escaped.push_str("\\$"),
            '%' => escaped.push_str("%%"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(character),
        }
    }
    escaped
}

#[cfg(target_os = "linux")]
pub fn supported() -> bool {
    true
}

#[cfg(not(target_os = "linux"))]
pub fn supported() -> bool {
    false
}

#[cfg(target_os = "linux")]
pub fn status(home: &Path) -> Status {
    match std::fs::read_to_string(entry_path(home)) {
        Ok(contents) => status_from_entry(&contents),
        Err(_) => Status::Disabled,
    }
}

#[cfg(not(target_os = "linux"))]
pub fn status(_home: &Path) -> Status {
    Status::Unsupported
}

#[cfg(target_os = "linux")]
fn status_from_entry(contents: &str) -> Status {
    let mut in_desktop_entry = false;
    let mut enabled = true;

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_desktop_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_desktop_entry {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "X-GNOME-Autostart-enabled" {
            let value = value.trim().trim_matches('"').trim();
            enabled = !matches!(value.to_ascii_lowercase().as_str(), "false" | "0" | "no");
        }
    }

    if enabled {
        Status::Enabled
    } else {
        Status::Disabled
    }
}

#[cfg(target_os = "linux")]
pub fn set_enabled(home: &Path, enabled: bool, executable: &Path) -> io::Result<()> {
    let path = entry_path(home);
    if enabled {
        write_entry(&path, executable)
    } else {
        remove_entry(&path)
    }
}

#[cfg(not(target_os = "linux"))]
pub fn set_enabled(_home: &Path, _enabled: bool, _executable: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "autostart is unsupported on this platform",
    ))
}

#[cfg(target_os = "linux")]
fn write_entry(path: &Path, executable: &Path) -> io::Result<()> {
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let contents = render_entry(executable)?;
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "autostart entry has no parent directory",
        )
    })?;
    fs::create_dir_all(parent)?;

    let temp = temp_path(path);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o644)
            .open(&temp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::set_permissions(&temp, fs::Permissions::from_mode(0o644))?;
        fs::rename(&temp, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(target_os = "linux")]
fn remove_entry(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "linux")]
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
