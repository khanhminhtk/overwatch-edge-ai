#pragma once
#include "edge_sentinel/src/application/ports/onnx.hpp"
#include "edge_sentinel/src/application/dto/onnx_config_session.hpp"
#include "edge_sentinel/OnnxCpuDev/onnxruntime/include/onnxruntime_cxx_api.h"
#include <memory>
#include <string>
#include <vector>

class OnnxSession : public IOnnxSession {
    public:
        OnnxSession(IOnnxEnvironment* env);
        ~OnnxSession() override = default;
        
        void initialize(const OnnxSessionParams& params) override;
        
        void loadModel(const char* modelPath) override;
        void loadModelFromMemory(const void* modelData, size_t modelDataLength) override;
        
        size_t getInputCount() const override;
        size_t getOutputCount() const override;
        
        std::string getInputName(size_t index) const override;
        std::string getOutputName(size_t index) const override;
        
        std::vector<int64_t> getInputShape(size_t index) const override;
        std::vector<int64_t> getOutputShape(size_t index) const override;
        
        void* getNativeSession() override;
        
        void run(
            const std::vector<const char*>& inputNames,
            const std::vector<void*>& inputTensors,
            const std::vector<const char*>& outputNames,
            std::vector<void*>& outputTensors
        ) override;
        
        void* createInputTensor(
            const std::vector<int64_t>& shape,
            void* data,
            size_t dataSize
        ) override;
        
        void* createOutputTensor(
            const std::vector<int64_t>& shape
        ) override;
        
        void releaseTensor(void* tensor) override;
        
        void* getTensorData(void* tensor) override;
        std::vector<int64_t> getTensorShape(void* tensor) override;
        size_t getTensorElementCount(void* tensor) override;
        
        void Inference(const std::vector<float>& input_data, 
                      const std::vector<int64_t>& input_shape,
                      std::vector<float>& output_buffer,
                      const std::vector<int64_t>& output_shape);
        
        size_t GetInputSize() const;
    
    private:
        void configureExecutionMode(ExecutionModeType mode);
        void configureGraphOptimization(GraphOptimizationLevelType level);
        void configureExecutionProvider(const ExecutionProvider& provider);
        void configureMemory(const MemoryConfigurationType& memConfig);
        void configureModelOptimizer(const ModelOptimizerSaving& optimizer);
        void configureProfiling(const ProfilingConfig& profiling);
        void configureLogLevel(LogSeverityLevel level);
        
        IOnnxEnvironment* env_;
        std::unique_ptr<Ort::SessionOptions> sessionOptions_;
        std::unique_ptr<Ort::Session> session_;
        OnnxSessionParams config_;
        
        std::unique_ptr<Ort::IoBinding> io_binding_;
        Ort::MemoryInfo memory_info_cpu_;
        std::vector<const char*> input_names_;
        std::vector<const char*> output_names_;
        std::vector<std::string> input_names_storage_;
        std::vector<std::string> output_names_storage_;
};