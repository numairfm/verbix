use std::path::{Component, Path, PathBuf};

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn socket_path(home: &Path) -> PathBuf {
    home.join(".local/share/verbix/verbix.sock")
}

pub fn voice_dir(home: &Path) -> PathBuf {
    home.join(".verbix/voice")
}

pub fn work_dir(home: &Path) -> PathBuf {
    home.join(".config/verbix")
}

pub fn logs_dir(home: &Path) -> PathBuf {
    home.join(".verbix/logs")
}

/// The executable to point a desktop shortcut at: `$APPIMAGE` when it names a
/// real file, otherwise the running executable. A mounted AppImage runtime
/// (`.../.mount_*`) is never preferred over the resolved AppImage path.
pub fn executable_path() -> PathBuf {
    let appimage = std::env::var_os("APPIMAGE");
    let current = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("verbix-shell"));
    executable_path_with(appimage.as_deref().map(Path::new), current)
}

pub fn executable_path_with(appimage: Option<&Path>, current_exe: PathBuf) -> PathBuf {
    match appimage {
        Some(path) if !path.as_os_str().is_empty() && !is_mounted_runtime(path) => {
            path.to_path_buf()
        }
        _ => current_exe,
    }
}

fn is_mounted_runtime(path: &Path) -> bool {
    path.components().any(|component| match component {
        Component::Normal(name) => name.to_string_lossy().starts_with(".mount_"),
        _ => false,
    })
}
