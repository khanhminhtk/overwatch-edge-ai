#pragma once
#include "onnxsession.hpp"
#include "edge_sentinel/OnnxCpuDev/onnxruntime/include/onnxruntime_cxx_api.h"
#include <memory>
#include <stdexcept>

void* OnnxSession::createInputTensor(
    const std::vector<int64_t>& shape,
    void* data,
    size_t dataSize
) {
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null.");
    }
    
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.");
    }
    
    size_t totalElements = 1;
    for (auto dim : shape) {
        totalElements *= dim;
    }
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );
    
    auto* tensor = new Ort::Value(Ort::Value::CreateTensor<float>(
        memoryInfo,
        static_cast<float*>(data),
        totalElements,
        shape.data(),
        shape.size()
    ));
    
    return static_cast<void*>(tensor);
}

void* OnnxSession::createOutputTensor(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.");
    }
    
    size_t totalElements = 1;
    for (auto dim : shape) {
        totalElements *= dim;
    }
    
    auto* outputData = new float[totalElements];
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );
    
    auto* tensor = new Ort::Value(Ort::Value::CreateTensor<float>(
        memoryInfo,
        outputData,
        totalElements,
        shape.data(),
        shape.size()
    ));
    
    return static_cast<void*>(tensor);
}

void OnnxSession::releaseTensor(void* tensor) {
    if (tensor) {
        auto* ortValue = static_cast<Ort::Value*>(tensor);
        delete ortValue;
    }
}

void* OnnxSession::getTensorData(void* tensor) {
    if (!tensor) {
        throw std::invalid_argument("Tensor pointer cannot be null.");
    }
    
    auto* ortValue = static_cast<Ort::Value*>(tensor);
    
    if (!ortValue->IsTensor()) {
        throw std::runtime_error("Value is not a tensor.");
    }
    
    return ortValue->GetTensorMutableData<float>();
}

std::vector<int64_t> OnnxSession::getTensorShape(void* tensor) {
    if (!tensor) {
        throw std::invalid_argument("Tensor pointer cannot be null.");
    }
    
    auto* ortValue = static_cast<Ort::Value*>(tensor);
    
    if (!ortValue->IsTensor()) {
        throw std::runtime_error("Value is not a tensor.");
    }
    
    auto tensorInfo = ortValue->GetTensorTypeAndShapeInfo();
    return tensorInfo.GetShape();
}

size_t OnnxSession::getTensorElementCount(void* tensor) {
    if (!tensor) {
        throw std::invalid_argument("Tensor pointer cannot be null.");
    }
    
    auto* ortValue = static_cast<Ort::Value*>(tensor);
    
    if (!ortValue->IsTensor()) {
        throw std::runtime_error("Value is not a tensor.");
    }
    
    auto tensorInfo = ortValue->GetTensorTypeAndShapeInfo();
    return tensorInfo.GetElementCount();
}
