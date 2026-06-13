#include <iostream>
#include <cctype>
#include <string>
#include <variant>

#include "runtime_config_loader.hpp"
#include "triton_grpc_client.hpp"
#include "videosource.hpp"

int main(int argc, char** argv)
{
    try {
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
        const std::string config_path = argc > 1 ? argv[1] : "config/runtime.yaml";

        RuntimeConfigLoader loader(config_path);
        const RuntimeConfig config = loader.load();

        TritonGrpcClient     triton_client(config.triton_url);
        triton_client.connect();

        std::cout << "[INFO] Triton live: " << triton_client.is_server_live() << '\n';
        std::cout << "[INFO] Triton ready: " << triton_client.is_server_ready() << '\n';
        std::cout << "[INFO] Detector ready: " << triton_client.is_model_ready(config.detector_model_name) << '\n';
        std::cout << "[INFO] Recognizer ready: " << triton_client.is_model_ready(config.recognizer_model_name) << '\n';
#else
        std::cout << "[INFO] Triton gRPC is disabled for this build" << '\n';

        const std::string raw_source = argc > 1 ? argv[1] : "0";
        const bool is_camera_index =
            !raw_source.empty() &&
            std::all_of(raw_source.begin(), raw_source.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            });
#endif

        const std::variant<int, std::string> source =
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
            0;
#else
            is_camera_index ? std::variant<int, std::string>(std::stoi(raw_source))
                            : std::variant<int, std::string>(raw_source);
        std::cout << "[INFO] Using source: " << raw_source << '\n';
#endif
        VideoSource video_source(source);
        cv::Mat frame;
        const bool has_frame = video_source.getFrame(frame);

        std::cout << "[INFO] First frame available: " << has_frame << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << '\n';
        return 1;
    }
}
