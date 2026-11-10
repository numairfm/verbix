use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone)]
pub struct LogSink {
    file: Arc<Mutex<File>>,
}

impl LogSink {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut options = OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn line(&self, text: &str) {
        let redacted = redact_line(text);
        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(file, "{redacted}");
        }
    }
}

pub fn pump_reader<R>(reader: R, sink: LogSink) -> JoinHandle<()>
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => sink.line(&line),
                Err(_) => break,
            }
        }
    })
}

pub fn redact_line(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    if let Some(index) = lower.find("authorization:") {
        return format!("{} [redacted]", &line[..index + "authorization:".len()]);
    }
    if let Some(index) = lower.find("bearer ") {
        let start = index + "bearer ".len();
        return format!("{}[redacted]", &line[..start]);
    }
    for key in ["api_key=", "apikey=", "token=", "secret=", "password="] {
        if let Some(index) = lower.find(key) {
            let start = index + key.len();
            return format!("{}[redacted]", &line[..start]);
        }
    }
    let mut redacted = String::with_capacity(line.len());
    for (index, token) in line.split(' ').enumerate() {
        if index > 0 {
            redacted.push(' ');
        }
        if token.starts_with("sk-") || token.starts_with("sk_") {
            redacted.push_str("[redacted]");
        } else {
            redacted.push_str(token);
        }
    }
    redacted
}
