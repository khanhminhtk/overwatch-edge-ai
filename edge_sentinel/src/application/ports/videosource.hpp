#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <variant>

class VideoSource{
    public:
        VideoSource(const std::variant<int, std::string>& source);
        bool getFrame(cv::Mat& frame);
        ~VideoSource();
    private:
        cv::VideoCapture cap;
};