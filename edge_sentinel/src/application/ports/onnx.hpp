#pragma once
#include "edge_sentinel/src/domain/entities/onnxlog.hpp"
#include "edge_sentinel/src/application/dto/onnx_config_session.hpp"
#include <vector>
#include <string>
#include <cstddef>

class IOnnxEnvironment {
    public:
        virtual ~IOnnxEnvironment() = default;
        virtual void* getNativeEnv() = 0;
};

class IOnnxSession {
    public:
        virtual ~IOnnxSession() = default;
        
        virtual void initialize(const OnnxSessionParams& params) = 0;
        
        virtual void loadModel(const char* modelPath) = 0;
        virtual void loadModelFromMemory(const void* modelData, size_t modelDataLength) = 0;
        
        virtual size_t getInputCount() const = 0;
        virtual size_t getOutputCount() const = 0;
        
        virtual std::string getInputName(size_t index) const = 0;
        virtual std::string getOutputName(size_t index) const = 0;
        
        virtual std::vector<int64_t> getInputShape(size_t index) const = 0;
        virtual std::vector<int64_t> getOutputShape(size_t index) const = 0;
        
        virtual void* getNativeSession() = 0;
        virtual void run(
            const std::vector<const char*>& inputNames,
            const std::vector<void*>& inputTensors,
            const std::vector<const char*>& outputNames,
            std::vector<void*>& outputTensors
        ) = 0;
        
        virtual void* createInputTensor(
            const std::vector<int64_t>& shape,
            void* data,
            size_t dataSize
        ) = 0;
        
        virtual void* createOutputTensor(
            const std::vector<int64_t>& shape
        ) = 0;
        
        virtual void releaseTensor(void* tensor) = 0;
        
        virtual void* getTensorData(void* tensor) = 0;
        virtual std::vector<int64_t> getTensorShape(void* tensor) = 0;
        virtual size_t getTensorElementCount(void* tensor) = 0;
};