#include "onnxenv.hpp"

OnnxEnvironment::OnnxEnvironment(const LogConfig& config, OrtLoggingLevel level, const char* logid, std::ofstream* logFile)
    : config_(config), logFile_(logFile) {
    
    logParams_.deviceID = config.deviceID;
    logParams_.isDebugMode = config.isDebugMode;
    logParams_.logFile = logFile;
    
    env_ = std::make_unique<Ort::Env>(
        level,
        logid,
        OnnxEnvironment::LogCallback,
        &logParams_
    );
}

void* OnnxEnvironment::getNativeEnv() {
    if (!env_) {
        throw std::runtime_error("ONNX Runtime environment not initialized.");
    }
    return env_.get();
}

void ORT_API_CALL OnnxEnvironment::LogCallback(
    void* param,
    OrtLoggingLevel severity,
    const char* category,
    const char* logid,
    const char* code_location,
    const char* message
) {
    LogParams* ctx = static_cast<LogParams*>(param);
    std::lock_guard<std::mutex> lock(ctx->mutex);

    std::string levelStr;
    switch (severity) {
        case ORT_LOGGING_LEVEL_VERBOSE: levelStr = "[VERBOSE]"; break;
        case ORT_LOGGING_LEVEL_INFO: levelStr = "[INFO]"; break;
        case ORT_LOGGING_LEVEL_WARNING: levelStr = "[WARN]"; break;
        case ORT_LOGGING_LEVEL_ERROR: levelStr = "[ERROR]"; break;
        case ORT_LOGGING_LEVEL_FATAL: levelStr = "[FATAL]"; break;
        default: levelStr = "[UNKNOWN]"; break;
    }

    std::string finalLog = "[" + ctx->deviceID + "]" + levelStr + "[" + std::string(logid) + "] " + std::string(message);

    if (ctx->isDebugMode) {
        if (severity >= ORT_LOGGING_LEVEL_ERROR) {
            std::cerr << finalLog << std::endl;
        } else {
            std::cout << finalLog << std::endl;
        }
    }

    if (ctx->logFile && ctx->logFile->is_open()) {
        *ctx->logFile << finalLog << std::endl;
        if (severity >= ORT_LOGGING_LEVEL_ERROR) {
            ctx->logFile->flush();
        }
    }
}