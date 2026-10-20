use std::path::{Path, PathBuf};

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
