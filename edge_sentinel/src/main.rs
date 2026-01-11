mod application;
mod domain;
mod infrastructure;
mod adapter;

use application::VideoStreamUseCase;
use domain::entities::{VideoSourceConfig, VideoSourceType};
use infrastructure::OpencvVideoSource;

fn main() {
    println!("=== Video Streaming with Clean Architecture ===\n");

    let config = VideoSourceConfig::new(
        VideoSourceType::NetworkStream("http://192.168.1.100:8080/video".to_string())
    )
    .with_fps(30)
    .with_retry(3);

    let video_source = OpencvVideoSource::new();
    let mut use_case = VideoStreamUseCase::new(video_source);

    match use_case.initialize(&config) {
        Ok(_) => {
            println!("✓ Stream initialized\n");

            let mut frame_count = 0;
            let result = use_case.process_stream(|frame| {
                frame_count += 1;
                println!(
                    "Frame #{}: {}x{} {}p",
                    frame.frame_number(),
                    frame.width(),
                    frame.height(),
                    if frame.is_hd_quality() { "HD" } else { "SD" }
                );

                Ok(frame_count < 10)
            });

            if let Err(e) = result {
                eprintln!("✗ Error: {}", e);
            }

            let _ = use_case.shutdown();
            println!("\n✓ Stream closed");
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize: {}", e);
        }
    }
}
