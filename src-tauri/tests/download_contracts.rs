use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::tempdir;
use verbix_shell::download::{start, DownloadEvent, DownloadJob};

fn python() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fake_model_downloader.py")
}

fn job(dir: &Path, extra: &[&str]) -> DownloadJob {
    let mut args = vec![
        script().display().to_string(),
        "--root".to_string(),
        dir.join("voice").display().to_string(),
    ];
    args.extend(extra.iter().map(|value| value.to_string()));
    DownloadJob {
        program: PathBuf::from(python()),
        args,
        work_dir: dir.join("work").to_path_buf(),
        env: Vec::new(),
    }
}

fn collect(handle: &verbix_shell::download::DownloadHandle) -> Vec<DownloadEvent> {
    let mut events = Vec::new();
    while let Some(event) = handle.recv() {
        let finished = matches!(event, DownloadEvent::Finished { .. });
        events.push(event);
        if finished {
            break;
        }
    }
    events
}

#[test]
fn a_finished_download_reports_its_last_line_and_writes_the_model() {
    let dir = tempdir().unwrap();
    let handle = start(&job(dir.path(), &[])).expect("the fake downloader starts");
    let events = collect(&handle);

    assert!(
        events.iter().any(
            |event| matches!(event, DownloadEvent::Progress(line) if line.contains("Fetching"))
        ),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, DownloadEvent::Finished { success: true })),
        "{events:?}"
    );
    assert!(dir
        .path()
        .join("voice/stt/models/nemotron-gguf/fake.gguf")
        .is_file());
}

#[test]
fn a_failed_download_reports_the_last_stderr_line() {
    let dir = tempdir().unwrap();
    let handle = start(&job(dir.path(), &["--fail"])).expect("the fake downloader starts");
    let events = collect(&handle);

    assert!(
        events.iter().any(|event| matches!(
            event,
            DownloadEvent::Stderr(line) if line.contains("Checksum mismatch")
        )),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, DownloadEvent::Finished { success: false })),
        "{events:?}"
    );
}

#[test]
fn a_killed_download_stops_the_child() {
    let dir = tempdir().unwrap();
    let handle = start(&job(dir.path(), &["--sleep", "60"])).expect("the fake downloader starts");
    std::thread::sleep(Duration::from_millis(200));
    handle.kill();

    let events = collect(&handle);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, DownloadEvent::Finished { success: false })),
        "{events:?}"
    );
}
