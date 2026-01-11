use crate::domain::{entities::{VideoFrame, VideoSourceConfig}, Result};

pub trait VideoSource {
    fn connect(&mut self, config: &VideoSourceConfig) -> Result<()>;
    fn read_frame(&mut self) -> Result<Option<VideoFrame>>;
    fn disconnect(&mut self) -> Result<()>;
}