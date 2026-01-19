#pragma once
#include <string>
#include <optional>
#include <map>

enum class ExecutionModeType {
    Parallel,
    Sequential
};

enum class GraphOptimizationLevelType {
    DisableAll,
    EnableBasic,
    EnableExtended,
    EnableAll
};

enum class ExecutionProviderType {
    CPU,
    CUDA,
    TensorRT
};

enum class MemoryPatternType {
    Enable,
    Disable
};

enum class CpuMemArenaType {
    Enable,
    Disable
};

enum class LogSeverityLevel {
    Verbose,
    Info,
    Warning,
    Error,
    Fatal
};

struct ArenaCfg {
    int initialChunkSizeBytes;
    int maxMem;
    int arenaExtendStrategy;
};

struct ProfilingConfig {
    bool enabled = false;
    std::string profileFilePrefix;
};

struct ModelOptimizerSaving {
    std::string optimizedModelFilePath;
    std::string keyConfigEntry;
    std::string valueConfigEntry;
};

struct MemoryConfigurationType {
    MemoryPatternType memPatternType;
    CpuMemArenaType cpuMemArenaType;
    ArenaCfg arenaCfg;
};

struct ExecutionProvider {
    ExecutionProviderType type;
    std::map<std::string, std::string> options;
    int device = 0;
};

struct OnnxSessionParams {
    int intraOpNumThread;
    int interOpNumThread;
    ExecutionModeType executionMode;
    GraphOptimizationLevelType graphOptimizationLevel;
    std::optional<ExecutionProvider> executionProvider;
    std::optional<MemoryConfigurationType> memoryConfig; 
    std::optional<ModelOptimizerSaving> modelOptimizerSaving;
    std::optional<ProfilingConfig> profilingConfig;
    std::optional<LogSeverityLevel> logSeverityLevel;
};
