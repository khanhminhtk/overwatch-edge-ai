#pragma once
#include <opencv2/opencv.hpp>
#include "videosource.hpp"
#include <string>
#include <variant>

VideoSource::VideoSource(const std::variant<int, std::string>& source){
    if (std::holds_alternative<int>(source)) {
        int index = std::get<int>(source);
        cap.open(index=index);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    } else if (std::holds_alternative<std::string>(source))
    {
        std::string path = std::get<std::string>(source);
        cap.open(path, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            std::cout << "[WARN] FFMPEG failed, trying GSTREAMER..." << std::endl;
            cap.open(path, cv::CAP_GSTREAMER);
        }
        if (!cap.isOpened()) {
            throw std::runtime_error("ERROR: Could not open video source!");
        }
        std::cout << "[INFO] Video source opened successfully!" << std::endl;
    }
};

bool VideoSource::getFrame(cv::Mat& frame) {
    cap.read(frame);
    if (!frame.empty()) {
        return true;
    } else if (frame.empty())
    {
        std::cout << "Frame is empty";
        return false;
    } else {
        std::cout << "ERROR for getFrame";
        return false;
    }
}

VideoSource::~VideoSource() {
    std::cout << "[INFO] Destructor called. Cleaning up..." << std::endl;
        if (cap.isOpened()) {
            cap.release();
            std::cout << "[INFO] Camera released successfully." << std::endl;
        }
}