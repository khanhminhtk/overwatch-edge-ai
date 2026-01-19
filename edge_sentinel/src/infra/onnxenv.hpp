#pragma once
#include "edge_sentinel/OnnxCpuDev/onnxruntime/include/onnxruntime_cxx_api.h"
#include "edge_sentinel/src/application/ports/onnx.hpp"
#include <fstream>
#include <memory>
#include <mutex>
#include <iostream>

class OnnxEnvironment : public IOnnxEnvironment {
    public:
        OnnxEnvironment(const LogConfig& config, OrtLoggingLevel level, const char* logid, std::ofstream* logFile = nullptr);
        
        void* getNativeEnv() override;

    private:
        struct LogParams {
            std::string deviceID;
            bool isDebugMode;
            std::ofstream* logFile;
            std::mutex mutex;
        };

        static void ORT_API_CALL LogCallback(
            void* param,
            OrtLoggingLevel severity,
            const char* category,
            const char* logid,
            const char* code_location,
            const char* message
        );

        LogConfig config_;
        std::ofstream* logFile_;
        LogParams logParams_;
        std::unique_ptr<Ort::Env> env_;
};
