use std::path::{Path, PathBuf};

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
