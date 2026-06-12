#pragma once

#include <string>

struct RuntimeConfig {
    std::string triton_url = "localhost:8001";
    std::string detector_model_name = "detector";
    std::string recognizer_model_name = "recognizer";
    std::string source = "0";
};

class RuntimeConfigLoader {
  public:
    RuntimeConfig load(const std::string& config_path) const;
};

