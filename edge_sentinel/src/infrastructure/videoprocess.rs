use opencv::prelude::*;
use crate::{application::ports::videoprocess::VideoSource, domain::entities::VideoSourceType};
use crate::domain::entities::{VideoSourceConfig, VideoFrame};
use crate::domain::{Result, VideoSourceError};
use opencv::{
    videoio::{self, VideoCapture, CAP_ANY},
};

pub struct OpencvVideoSource {
    capture: Option<VideoCapture>,
    config: Option<VideoSourceConfig>,
    frame_count: u64,
}

impl OpencvVideoSource {
    pub fn new() -> Self {
        Self {
            capture: None,
            config: None,
            frame_count: 0,
        }
    }
    
    fn open_capture(source_type: &VideoSourceType) -> Result<VideoCapture> {
        let capture = match source_type {
            VideoSourceType::UsbCamera(index) => {
                VideoCapture::new(*index, CAP_ANY)
                .map_err(|e| VideoSourceError::ConnectionFailed(format!("USB Camera Error: {}", e)))?
            }
            VideoSourceType::NetworkStream(url) => {
                VideoCapture::from_file(url.as_str(), videoio::CAP_ANY)
                .map_err(|e| VideoSourceError::ConnectionFailed(format!("Stream URL Error: {}", e)))?
            }
        };
        
        if !capture.is_opened()
            .map_err(|e| VideoSourceError::ConnectionFailed(e.to_string()))?
        {
            return Err(VideoSourceError::ConnectionFailed(
                "Failed to open video source".to_string(),
            ));
        }
        
        Ok(capture)
    }
    
    fn get_frame_size(&self) -> Result<(u32, u32)> {
        let capture = self
            .capture
            .as_ref()
            .ok_or_else(|| VideoSourceError::ConnectionFailed("Not connected".to_string()))?;

        let width = capture
            .get(videoio::CAP_PROP_FRAME_WIDTH)
            .map_err(|e| VideoSourceError::FrameCaptureError(e.to_string()))? as u32;

        let height = capture
            .get(videoio::CAP_PROP_FRAME_HEIGHT)
            .map_err(|e| VideoSourceError::FrameCaptureError(e.to_string()))? as u32;

        Ok((width, height))
    }
}

impl Default for OpencvVideoSource {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoSource for OpencvVideoSource {
    fn connect(&mut self, config: &VideoSourceConfig) -> Result<()> {
        if !config.is_valid_fps() {
            return Err(VideoSourceError::InvalidConfiguration(
                "Invalid FPS configuration".to_string(),
            ));
        }
        
        let mut capture = Self::open_capture(config.source_type())?;

        if let Some(fps) = config.expected_fps() {
            let _ = capture.set(videoio::CAP_PROP_FPS, fps as f64);
        }

        self.capture = Some(capture);
        self.config = Some(config.clone());
        self.frame_count = 0;

        Ok(())
    }
    
    fn read_frame(&mut self) -> Result<Option<VideoFrame>> {
        let capture = self
            .capture
            .as_mut()
            .ok_or_else(|| VideoSourceError::ConnectionFailed("Not connected".to_string()))?;

        let mut mat = opencv::core::Mat::default();
        let success = capture
            .read(&mut mat)
            .map_err(|e| VideoSourceError::FrameCaptureError(e.to_string()))?;

        if !success || mat.empty() {
            return Ok(None);
        }

        let (width, height) = self.get_frame_size()?;

        let frame = VideoFrame::new(self.frame_count, width, height);
        if !frame.is_valid_resolution() {
            return Err(VideoSourceError::InvalidResolution { width, height });
        }

        self.frame_count += 1;

        Ok(Some(frame))
    }
    
    fn disconnect(&mut self) -> Result<()> {
        if let Some(mut capture) = self.capture.take() {
            capture
                .release()
                .map_err(|e| VideoSourceError::ConnectionFailed(e.to_string()))?;
        }
        self.config = None;
        self.frame_count = 0;
        Ok(())
    }
}
