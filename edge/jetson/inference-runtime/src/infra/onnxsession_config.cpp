#pragma once
#include "onnxsession.hpp"
#include "edge_sentinel/OnnxCpuDev/onnxruntime/include/onnxruntime_cxx_api.h"
#include "edge_sentinel/src/application/dto/onnx_config_session.hpp"
#include <memory>
#include <stdexcept>
#include <iostream>

void OnnxSession::configureExecutionMode(ExecutionModeType mode) {
    switch (mode) {
        case ExecutionModeType::Parallel:
            sessionOptions_->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
            break;
        case ExecutionModeType::Sequential:
            sessionOptions_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            break;
        default:
            throw std::invalid_argument("Unknown ExecutionModeType provided to SessionOptions.");
    }
}

void OnnxSession::configureGraphOptimization(GraphOptimizationLevelType level) {
    switch (level) {
        case GraphOptimizationLevelType::DisableAll:
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            break;
        case GraphOptimizationLevelType::EnableBasic:
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            break;
        case GraphOptimizationLevelType::EnableExtended:
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            break;
        case GraphOptimizationLevelType::EnableAll:
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            break;
        default:
            throw std::invalid_argument("Unknown GraphOptimizationLevelType provided to SessionOptions.");
    }
}

void OnnxSession::configureExecutionProvider(const ExecutionProvider& provider) {
    switch (provider.type) {
        case ExecutionProviderType::CPU:
            std::cout << "[INFO] Using CPU execution provider" << std::endl;
            break;
            
        case ExecutionProviderType::CUDA: {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = provider.device;
            sessionOptions_->AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[INFO] CUDA execution provider configured on device " 
                      << provider.device << std::endl;
            break;
        }
            
        case ExecutionProviderType::TensorRT: {
            OrtTensorRTProviderOptionsV2* trt_options;
            auto& api = Ort::GetApi();
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
            
            try {
                std::vector<const char*> keys;
                std::vector<const char*> values;
                std::string device_id_str = std::to_string(provider.device);
                keys.push_back("device_id");
                values.push_back(device_id_str.c_str());
                std::vector<std::string> option_values; 
                for (const auto& [key, value] : provider.options) {
                    keys.push_back(key.c_str());
                    option_values.push_back(value);
                    values.push_back(option_values.back().c_str());
                }
                if (!keys.empty()) {
                    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
                        trt_options, 
                        keys.data(), 
                        values.data(), 
                        keys.size()
                    ));
                }
                Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                    sessionOptions_->operator OrtSessionOptions*(), 
                    trt_options
                ));
                
                std::cout << "[INFO] TensorRT execution provider V2 configured on device " 
                          << provider.device << std::endl;
                api.ReleaseTensorRTProviderOptions(trt_options);
                
            } catch (...) {
                api.ReleaseTensorRTProviderOptions(trt_options);
                throw;
            }
            break;
        }
            
        default:
            throw std::invalid_argument("Unknown ExecutionProviderType provided.");
    }
}

void OnnxSession::configureMemory(const MemoryConfigurationType& memConfig) {
    if (memConfig.memPatternType == MemoryPatternType::Enable) {
        sessionOptions_->EnableMemPattern();
    } else {
        sessionOptions_->DisableMemPattern();
    }

    if (memConfig.cpuMemArenaType == CpuMemArenaType::Enable) {
        sessionOptions_->EnableCpuMemArena();
    } else {
        sessionOptions_->DisableCpuMemArena();
    }
}

void OnnxSession::configureModelOptimizer(const ModelOptimizerSaving& optimizer) {
    sessionOptions_->SetOptimizedModelFilePath(optimizer.optimizedModelFilePath.c_str());
    if (!optimizer.keyConfigEntry.empty() && !optimizer.valueConfigEntry.empty()) {
        sessionOptions_->AddConfigEntry(
            optimizer.keyConfigEntry.c_str(), 
            optimizer.valueConfigEntry.c_str()
        );
    }
}

void OnnxSession::configureProfiling(const ProfilingConfig& profiling) {
    if (profiling.enabled) {
        sessionOptions_->EnableProfiling(profiling.profileFilePrefix.c_str());
    } else {
        sessionOptions_->DisableProfiling();
    }
}

void OnnxSession::configureLogLevel(LogSeverityLevel level) {
    switch (level) {
        case LogSeverityLevel::Verbose:
            sessionOptions_->SetLogSeverityLevel(0);
            break;
        case LogSeverityLevel::Info:
            sessionOptions_->SetLogSeverityLevel(1);
            break;
        case LogSeverityLevel::Warning:
            sessionOptions_->SetLogSeverityLevel(2);
            break;
        case LogSeverityLevel::Error:
            sessionOptions_->SetLogSeverityLevel(3);
            break;
        case LogSeverityLevel::Fatal:
            sessionOptions_->SetLogSeverityLevel(4);
            break;
        default:
            throw std::invalid_argument("Unknown LogSeverityLevel provided.");
    }
}
