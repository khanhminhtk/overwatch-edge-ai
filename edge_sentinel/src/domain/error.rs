use std::fmt;

#[derive(Debug)]
pub enum VideoSourceError {
    ConnectionFailed(String),
    InvalidConfiguration(String),
    FrameCaptureError(String),
    InvalidResolution { width: u32, height: u32 },
}

impl fmt::Display for VideoSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoSourceError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            VideoSourceError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            VideoSourceError::FrameCaptureError(msg) => write!(f, "Frame capture error: {}", msg),
            VideoSourceError::InvalidResolution { width, height } => {
                write!(f, "Invalid resolution: {}x{}", width, height)
            }
        }
    }
}

impl std::error::Error for VideoSourceError {}

pub type Result<T> = std::result::Result<T, VideoSourceError>;
