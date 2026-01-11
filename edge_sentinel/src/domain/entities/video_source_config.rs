#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum VideoSourceType {
    UsbCamera(i32),
    NetworkStream(String),
}

#[derive(Debug, Clone)]
pub struct VideoSourceConfig {
    source_type: VideoSourceType,
    expected_fps: Option<u8>,
    retry_attempts: u8,
}

impl VideoSourceConfig {
    pub fn new(source_type: VideoSourceType) -> Self {
        Self {
            source_type,
            expected_fps: None,
            retry_attempts: 3,
        }
    }

    pub fn with_fps(mut self, fps: u8) -> Self {
        self.expected_fps = Some(fps);
        self
    }

    pub fn with_retry(mut self, attempts: u8) -> Self {
        self.retry_attempts = attempts;
        self
    }

    pub fn is_valid_fps(&self) -> bool {
        match self.expected_fps {
            Some(fps) => fps > 0 && fps <= 120,
            None => true,
        }
    }

    pub fn source_type(&self) -> &VideoSourceType {
        &self.source_type
    }

    pub fn expected_fps(&self) -> Option<u8> {
        self.expected_fps
    }

    pub fn retry_attempts(&self) -> u8 {
        self.retry_attempts
    }
}
