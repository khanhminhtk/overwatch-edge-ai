#pragma once

#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>


struct RuntimeConfig {
    std::string triton_url = "localhost:8001";
    std::string detector_model_name = "detector";
    std::string recognizer_model_name = "recognizer";
    std::string source = "0";
};

class RuntimeConfigLoader {
  public:
    explicit RuntimeConfigLoader(
        std::string yaml_path,
        std::string env_path = "config/.env");

    RuntimeConfig load() const;

  private:
    static std::unordered_map<std::string, std::string> loadEnvFile(const std::string& env_path);
    static std::string trim(const std::string& value);

    std::string resolveStringValue(
        const std::string& value,
        const std::string& field_name) const;

    std::string resolveNodeValue(
        const YAML::Node& node,
        const std::string& field_name,
        const std::string& fallback) const;

    std::string yaml_path_;
    std::string env_path_;
    std::unordered_map<std::string, std::string> env_values_;
};
