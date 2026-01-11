use crate::{
    application::ports::videoprocess::VideoSource,
    domain::{
        entities::{VideoFrame, VideoSourceConfig},
        Result, VideoSourceError,
    },
};
use std::time::Duration;

pub struct VideoStreamUseCase<V: VideoSource> {
    video_source: V,
}

impl<V: VideoSource> VideoStreamUseCase<V> {
    pub fn new(video_source: V) -> Self {
        Self { video_source }
    }

    pub fn initialize(&mut self, config: &VideoSourceConfig) -> Result<()> {
        let max_retries = config.retry_attempts();
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.video_source.connect(config) {
                Ok(_) => {
                    println!("Connected (attempt {})", attempt + 1);
                    return Ok(());
                }
                Err(e) => {
                    println!("Attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                    if attempt < max_retries - 1 {
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            VideoSourceError::ConnectionFailed("All retries failed".to_string())
        }))
    }

    pub fn process_stream<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(&VideoFrame) -> Result<bool>,
    {
        loop {
            match self.video_source.read_frame()? {
                Some(frame) => {
                    if !frame.is_valid_resolution() {
                        return Err(VideoSourceError::InvalidResolution {
                            width: frame.width(),
                            height: frame.height(),
                        });
                    }

                    let should_continue = callback(&frame)?;
                    if !should_continue {
                        break;
                    }
                }
                None => {
                    println!("Stream ended");
                    break;
                }
            }
        }

        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.video_source.disconnect()
    }
}
