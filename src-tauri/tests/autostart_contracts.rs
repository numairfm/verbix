#![cfg(target_os = "linux")]

use std::ffi::{OsStr, OsString};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use tempfile::tempdir;

use verbix_shell::autostart::{self, MenuState, Status};

/// Environment changes are process-wide, so env-mutating tests take turns.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVar {
    name: &'static str,
    previous: Option<OsString>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvVar {
    fn set(name: &'static str, value: Option<&OsStr>) -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(name);
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
        Self {
            name,
            previous,
            _lock: lock,
        }
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var(self.name, value),
            None => std::env::remove_var(self.name),
        }
    }
}

fn default_entry(home: &Path) -> PathBuf {
    home.join(".config/autostart/verbix.desktop")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

fn file_mode(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

fn enable(home: &Path, executable: &Path) {
    autostart::set_enabled(home, true, executable).expect("enable autostart");
}

fn disable(home: &Path, executable: &Path) {
    autostart::set_enabled(home, false, executable).expect("disable autostart");
}

#[test]
fn linux_reports_autostart_as_supported() {
    assert!(autostart::supported());
}

#[test]
fn tray_menu_state_is_checked_only_for_a_live_enabled_entry() {
    assert_eq!(
        autostart::menu_state(Status::Enabled),
        MenuState {
            enabled: true,
            checked: true,
        }
    );
    assert_eq!(
        autostart::menu_state(Status::Disabled),
        MenuState {
            enabled: true,
            checked: false,
        }
    );
    assert_eq!(
        autostart::menu_state(Status::Unsupported),
        MenuState {
            enabled: false,
            checked: false,
        }
    );
}

#[test]
fn absent_entry_means_disabled_and_status_never_writes_one() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);

    assert_eq!(autostart::status(home.path()), Status::Disabled);
    assert!(!home.path().join(".config").exists());
}

#[test]
fn enable_writes_a_valid_entry_for_the_appimage_with_hidden_flag() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let executable = home.path().join("Applications/Verbix 1.0.AppImage");
    fs::create_dir_all(executable.parent().unwrap()).unwrap();
    fs::write(&executable, b"appimage").unwrap();

    enable(home.path(), &executable);

    let entry = autostart::entry_path(home.path());
    assert_eq!(entry, default_entry(home.path()));
    assert!(entry.is_file());
    assert_eq!(
        read(&entry),
        concat!(
            "[Desktop Entry]\n",
            "Type=Application\n",
            "Name=Verbix\n",
            "Comment=Voice-operated desktop assistant\n",
            "Exec=\"/home/user/placeholder\" --hidden\n",
            "Terminal=false\n",
            "X-GNOME-Autostart-enabled=true\n",
        )
        .replace("/home/user/placeholder", &executable.display().to_string())
    );
    assert_eq!(autostart::status(home.path()), Status::Enabled);
    assert_eq!(file_mode(&entry), 0o644);
}

#[test]
fn enable_replaces_an_existing_entry_and_leaves_no_temporary_files() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let old = home.path().join("Old Place/Verbix.AppImage");
    let new = home.path().join("New Place/Verbix.AppImage");

    enable(home.path(), &old);
    enable(home.path(), &new);

    let entry = default_entry(home.path());
    let text = read(&entry);
    assert!(text.contains(&format!("Exec=\"{}\" --hidden", new.display())));
    assert!(!text.contains(&old.display().to_string()));

    let mut names: Vec<OsString> = fs::read_dir(entry.parent().unwrap())
        .unwrap()
        .map(|item| item.unwrap().file_name())
        .collect();
    names.sort();
    assert_eq!(names, vec![OsString::from("verbix.desktop")]);
}

#[test]
fn disable_removes_only_the_verbix_entry() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let executable = home.path().join("Verbix.AppImage");
    let autostart_dir = home.path().join(".config/autostart");
    fs::create_dir_all(&autostart_dir).unwrap();
    fs::write(
        autostart_dir.join("other.desktop"),
        "[Desktop Entry]\nName=Other\n",
    )
    .unwrap();

    enable(home.path(), &executable);
    assert!(default_entry(home.path()).is_file());

    disable(home.path(), &executable);

    assert!(!default_entry(home.path()).exists());
    assert!(autostart_dir.join("other.desktop").is_file());
    assert_eq!(autostart::status(home.path()), Status::Disabled);
}

#[test]
fn disabling_a_missing_entry_is_ok() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);

    disable(home.path(), &home.path().join("Verbix.AppImage"));

    assert_eq!(autostart::status(home.path()), Status::Disabled);
}

#[test]
fn moving_the_appimage_and_toggling_off_and_on_rewrites_the_entry() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let old = home.path().join("Old Place/Verbix.AppImage");
    let new = home.path().join("New Place/Verbix.AppImage");

    enable(home.path(), &old);
    assert!(read(&default_entry(home.path())).contains(&old.display().to_string()));

    disable(home.path(), &old);
    assert!(!default_entry(home.path()).exists());

    enable(home.path(), &new);
    let text = read(&default_entry(home.path()));
    assert!(text.contains(&format!("Exec=\"{}\" --hidden", new.display())));
    assert!(!text.contains(&old.display().to_string()));
    assert_eq!(autostart::status(home.path()), Status::Enabled);
}

#[test]
fn paths_with_spaces_quotes_backslashes_dollars_backticks_and_percent_render_a_valid_exec_line() {
    let path = Path::new("/opt/Verbix 1.0/a\"b\\c/$HOME/`tick`/100%/Verbix.AppImage");

    let entry = autostart::render_entry(path).expect("render entry");

    let exec = entry
        .lines()
        .find(|line| line.starts_with("Exec="))
        .expect("Exec line");
    assert_eq!(
        exec,
        r#"Exec="/opt/Verbix 1.0/a\"b\\c/\$HOME/\`tick\`/100%%/Verbix.AppImage" --hidden"#
    );
    assert_eq!(
        entry.matches('%').count(),
        2,
        "one literal percent became two"
    );
}

#[test]
fn a_tricky_path_round_trips_to_an_enabled_entry() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let executable = Path::new("/opt/Verbix 1.0/a\"b\\c/$HOME/`tick`/100%/Verbix.AppImage");

    enable(home.path(), executable);

    let text = read(&default_entry(home.path()));
    assert!(text.contains(
        r#"Exec="/opt/Verbix 1.0/a\"b\\c/\$HOME/\`tick\`/100%%/Verbix.AppImage" --hidden"#
    ));
    assert_eq!(autostart::status(home.path()), Status::Enabled);
}

#[test]
fn status_reflects_external_edits_to_the_entry() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let entry = default_entry(home.path());
    fs::create_dir_all(entry.parent().unwrap()).unwrap();

    for disabled in ["false", "FALSE", "0", "no", " No "] {
        fs::write(
            &entry,
            format!("[Desktop Entry]\nX-GNOME-Autostart-enabled={disabled}\n"),
        )
        .unwrap();
        assert_eq!(
            autostart::status(home.path()),
            Status::Disabled,
            "value {disabled:?}"
        );
    }

    for enabled in ["true", "TRUE", "1", "yes", "enabled"] {
        fs::write(
            &entry,
            format!("[Desktop Entry]\nX-GNOME-Autostart-enabled={enabled}\n"),
        )
        .unwrap();
        assert_eq!(
            autostart::status(home.path()),
            Status::Enabled,
            "value {enabled:?}"
        );
    }

    fs::write(&entry, "[Desktop Entry]\nName=Verbix\n").unwrap();
    assert_eq!(autostart::status(home.path()), Status::Enabled);

    fs::remove_file(&entry).unwrap();
    assert_eq!(autostart::status(home.path()), Status::Disabled);
}

#[test]
fn status_reads_the_key_only_from_the_desktop_entry_group() {
    let home = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", None);
    let entry = default_entry(home.path());
    fs::create_dir_all(entry.parent().unwrap()).unwrap();
    fs::write(
        &entry,
        concat!(
            "[Desktop Entry]\n",
            "Name=Verbix\n",
            "[Other]\n",
            "X-GNOME-Autostart-enabled=false\n",
        ),
    )
    .unwrap();

    assert_eq!(autostart::status(home.path()), Status::Enabled);
}

#[test]
fn xdg_config_home_chooses_the_entry_location() {
    let home = tempdir().unwrap();
    let config = tempdir().unwrap();
    let _xdg = EnvVar::set("XDG_CONFIG_HOME", Some(config.path().as_os_str()));
    let executable = home.path().join("Verbix.AppImage");
    let expected = config.path().join("autostart/verbix.desktop");

    assert_eq!(autostart::entry_path(home.path()), expected);

    enable(home.path(), &executable);
    assert!(expected.is_file());
    assert!(!default_entry(home.path()).exists());
    assert_eq!(autostart::status(home.path()), Status::Enabled);
}

#[test]
fn relative_xdg_config_home_is_ignored() {
    let home = tempdir().unwrap();

    assert_eq!(
        autostart::entry_path_with(home.path(), Some(Path::new("relative/config"))),
        default_entry(home.path())
    );
    assert_eq!(
        autostart::entry_path_with(home.path(), None),
        default_entry(home.path())
    );
}

#[test]
fn executable_path_prefers_the_appimage_over_the_mounted_runtime() {
    let appimage = Path::new("/home/user/Apps/Verbix.AppImage");
    let mounted = PathBuf::from("/tmp/.mount_VerbixAbC123/usr/bin/verbix-shell");

    assert_eq!(
        autostart::executable_path_with(Some(appimage), mounted),
        appimage
    );
}

#[test]
fn executable_path_ignores_an_appimage_value_that_is_a_mounted_runtime() {
    let mounted_appimage = Path::new("/tmp/.mount_VerbixAbC123/Verbix.AppImage");
    let current = PathBuf::from("/home/user/.local/bin/verbix-shell");

    assert_eq!(
        autostart::executable_path_with(Some(mounted_appimage), current.clone()),
        current
    );
}

#[test]
fn executable_path_falls_back_to_the_current_executable() {
    let current = PathBuf::from("/home/user/.local/bin/verbix-shell");

    assert_eq!(
        autostart::executable_path_with(None, current.clone()),
        current
    );
    assert_eq!(
        autostart::executable_path_with(Some(Path::new("")), current.clone()),
        current
    );
}

#[test]
fn executable_path_reads_the_appimage_from_the_environment() {
    let _appimage = EnvVar::set(
        "APPIMAGE",
        Some(OsStr::new("/home/user/Apps/Verbix.AppImage")),
    );

    assert_eq!(
        autostart::executable_path(),
        PathBuf::from("/home/user/Apps/Verbix.AppImage")
    );
}
