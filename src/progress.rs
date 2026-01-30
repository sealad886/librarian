//! Shared progress and logging helpers to keep progress bars pinned.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget};
use std::io::{self, Write};
use std::sync::OnceLock;
use tracing_subscriber::fmt::MakeWriter;

static MULTI_PROGRESS: OnceLock<MultiProgress> = OnceLock::new();

fn multi_progress() -> &'static MultiProgress {
    MULTI_PROGRESS.get_or_init(|| {
        let mp = MultiProgress::new();
        mp.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        mp
    })
}

pub fn add_progress_bar(len: u64) -> ProgressBar {
    multi_progress().add(ProgressBar::new(len))
}

#[derive(Default, Clone)]
pub struct LogWriterFactory;

pub struct LogWriter {
    buffer: String,
}

impl LogWriter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let line = self.buffer.trim_end_matches('\n').trim_end_matches('\r');
        if line.is_empty() {
            let _ = multi_progress().println(String::new());
        } else {
            let _ = multi_progress().println(line.to_string());
        }
        self.buffer.clear();
    }
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let chunk = String::from_utf8_lossy(buf);
        self.buffer.push_str(&chunk);

        while let Some(idx) = self.buffer.find('\n') {
            let line = self.buffer[..idx].trim_end_matches('\r');
            if line.is_empty() {
                let _ = multi_progress().println(String::new());
            } else {
                let _ = multi_progress().println(line.to_string());
            }
            self.buffer.drain(..idx + 1);
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer();
        Ok(())
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl<'a> MakeWriter<'a> for LogWriterFactory {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter::new()
    }
}
