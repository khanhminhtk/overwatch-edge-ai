#pragma once
#include "onnxsession.hpp"
#include "edge_sentinel/OnnxCpuDev/onnxruntime/include/onnxruntime_cxx_api.h"
#include "edge_sentinel/src/application/dto/onnx_config_session.hpp"
#include <memory>
#include <stdexcept>
#include <iostream>

OnnxSession::OnnxSession(IOnnxEnvironment* env) 
    : env_(env), 
      sessionOptions_(nullptr), 
      session_(nullptr),
      memory_info_cpu_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
    if (!env_) {
        throw std::invalid_argument("OnnxEnvironment cannot be null");
    }
}

void OnnxSession::initialize(const OnnxSessionParams& params) {
    sessionOptions_ = std::make_unique<Ort::SessionOptions>();
    config_ = params;
    sessionOptions_->SetInterOpNumThreads(params.interOpNumThread);
    sessionOptions_->SetIntraOpNumThreads(params.intraOpNumThread);
    configureExecutionMode(params.executionMode);
    configureGraphOptimization(params.graphOptimizationLevel);
    
    if (params.executionProvider.has_value()) {
        configureExecutionProvider(params.executionProvider.value());
    }
    
    if (params.memoryConfig.has_value()) {
        configureMemory(params.memoryConfig.value());
    }
    
    if (params.modelOptimizerSaving.has_value()) {
        configureModelOptimizer(params.modelOptimizerSaving.value());
    }
    
    if (params.profilingConfig.has_value()) {
        configureProfiling(params.profilingConfig.value());
    }
    
    if (params.logSeverityLevel.has_value()) {
        configureLogLevel(params.logSeverityLevel.value());
    }
}

void OnnxSession::loadModel(const char* modelPath) {
    if (!sessionOptions_) {
        throw std::runtime_error("Session options not initialized. Call initialize() first.");
    }
    
    if (!modelPath) {
        throw std::invalid_argument("Model path cannot be null.");
    }
    
    auto* ortEnv = static_cast<Ort::Env*>(env_->getNativeEnv());
    if (!ortEnv) {
        throw std::runtime_error("Invalid ONNX Runtime environment.");
    }
    
    session_ = std::make_unique<Ort::Session>(
        *ortEnv,  
        modelPath, 
        *sessionOptions_
    );

    io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
    
    Ort::AllocatorWithDefaultOptions allocator;
    size_t input_count = session_->GetInputCount();
    size_t output_count = session_->GetOutputCount();
    
    input_names_storage_.clear();
    output_names_storage_.clear();
    input_names_.clear();
    output_names_.clear();
    
    for (size_t i = 0; i < input_count; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_storage_.push_back(input_name.get());
        input_names_.push_back(input_names_storage_.back().c_str());
    }
    
    for (size_t i = 0; i < output_count; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.push_back(output_name.get());
        output_names_.push_back(output_names_storage_.back().c_str());
    }
    
    std::cout << "[INFO] Model loaded successfully from: " << modelPath << std::endl;
}

void OnnxSession::loadModelFromMemory(const void* modelData, size_t modelDataLength) {
    if (!sessionOptions_) {
        throw std::runtime_error("Session options not initialized. Call initialize() first.");
    }
    
    if (!modelData) {
        throw std::invalid_argument("Model data cannot be null.");
    }
    
    if (modelDataLength == 0) {
        throw std::invalid_argument("Model data length must be greater than 0.");
    }
    
    auto* ortEnv = static_cast<Ort::Env*>(env_->getNativeEnv());
    if (!ortEnv) {
        throw std::runtime_error("Invalid ONNX Runtime environment.");
    }
    
    session_ = std::make_unique<Ort::Session>(
        *ortEnv,
        modelData, 
        modelDataLength,
        *sessionOptions_
    );

    io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
    
    Ort::AllocatorWithDefaultOptions allocator;
    size_t input_count = session_->GetInputCount();
    size_t output_count = session_->GetOutputCount();
    
    input_names_storage_.clear();
    output_names_storage_.clear();
    input_names_.clear();
    output_names_.clear();
    
    for (size_t i = 0; i < input_count; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_storage_.push_back(input_name.get());
        input_names_.push_back(input_names_storage_.back().c_str());
    }
    
    for (size_t i = 0; i < output_count; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.push_back(output_name.get());
        output_names_.push_back(output_names_storage_.back().c_str());
    }
    
    std::cout << "[INFO] Model loaded successfully from memory (" 
              << modelDataLength << " bytes)" << std::endl;
}

size_t OnnxSession::getInputCount() const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    return session_->GetInputCount();
}

size_t OnnxSession::getOutputCount() const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    return session_->GetOutputCount();
}

std::string OnnxSession::getInputName(size_t index) const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = session_->GetInputNameAllocated(index, allocator);
    return std::string(name.get());
}

std::string OnnxSession::getOutputName(size_t index) const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = session_->GetOutputNameAllocated(index, allocator);
    return std::string(name.get());
}

std::vector<int64_t> OnnxSession::getInputShape(size_t index) const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<int64_t> OnnxSession::getOutputShape(size_t index) const {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

void* OnnxSession::getNativeSession() {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    return static_cast<void*>(session_.get());
}


void OnnxSession::run(
    const std::vector<const char*>& inputNames,
    const std::vector<void*>& inputTensors,
    const std::vector<const char*>& outputNames,
    std::vector<void*>& outputTensors
) {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    if (inputNames.size() != inputTensors.size()) {
        throw std::invalid_argument("Input names and tensors size mismatch.");
    }
    
    if (outputNames.empty()) {
        throw std::invalid_argument("Output names cannot be empty.");
    }
    
    std::vector<Ort::Value> inputOrtTensors;
    inputOrtTensors.reserve(inputTensors.size());
    for (auto* tensorPtr : inputTensors) {
        auto* ortValue = static_cast<Ort::Value*>(tensorPtr);
        inputOrtTensors.push_back(std::move(*ortValue));
    }
    
    Ort::RunOptions runOptions;
    auto outputOrtTensors = session_->Run(
        runOptions,
        inputNames.data(),
        inputOrtTensors.data(),
        inputNames.size(),
        outputNames.data(),
        outputNames.size()
    );

    outputTensors.clear();
    outputTensors.reserve(outputOrtTensors.size());
    for (auto& tensor : outputOrtTensors) {
        auto* tensorPtr = new Ort::Value(std::move(tensor));
        outputTensors.push_back(static_cast<void*>(tensorPtr));
    }
}

void OnnxSession::Inference(const std::vector<float>& input_data, 
                           const std::vector<int64_t>& input_shape,
                           std::vector<float>& output_buffer,
                           const std::vector<int64_t>& output_shape) {
    if (!session_) {
        throw std::runtime_error("Session not initialized. Call loadModel() first.");
    }
    
    if (!io_binding_) {
        throw std::runtime_error("I/O Binding not initialized. Call loadModel() first.");
    }
    
    io_binding_->ClearBoundInputs();
    io_binding_->ClearBoundOutputs();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cpu_, 
        const_cast<float*>(input_data.data()), 
        input_data.size(), 
        input_shape.data(), 
        input_shape.size()
    );
    
    io_binding_->BindInput(input_names_[0], input_tensor);

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cpu_, 
        output_buffer.data(), 
        output_buffer.size(), 
        output_shape.data(), 
        output_shape.size()
    );
    
    io_binding_->BindOutput(output_names_[0], output_tensor);

    session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
}

size_t OnnxSession::GetInputSize() const {
    return input_names_.size();
}
