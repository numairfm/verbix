use std::path::{Component, Path, PathBuf};

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn socket_path(home: &Path) -> PathBuf {
    home.join(".local/share/verbix/verbix.sock")
}

pub fn logs_dir(home: &Path) -> PathBuf {
    home.join(".verbix/logs")
}

/// Put the child in its own process group and, on Linux, make the kernel
/// stop it when the shell dies. The voice service and the model downloader
/// both use this.
pub fn configure_child(command: &mut std::process::Command) {
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            libc::setpgid(0, 0);
            #[cfg(target_os = "linux")]
            {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            }
            Ok(())
        });
    }
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
