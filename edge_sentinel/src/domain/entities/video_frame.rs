#[derive(Debug, Clone)]
pub struct VideoFrame {
    frame_number: u64,
    width: u32,
    height: u32,
}

impl VideoFrame {
    pub fn new(frame_number: u64, width: u32, height: u32) -> Self {
        Self {
            frame_number,
            width,
            height,
        }
    }

    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn is_valid_resolution(&self) -> bool {
        self.width > 0 && self.height > 0 && self.width <= 7680 && self.height <= 4320
    }

    pub fn is_hd_quality(&self) -> bool {
        self.width >= 1280 && self.height >= 720
    }
}