#include "runtime_config_loader.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

RuntimeConfig RuntimeConfigLoader::load(const std::string& config_path) const
{
    const YAML::Node root = YAML::LoadFile(config_path);
    RuntimeConfig config;

    if (const auto triton = root["triton"]) {
        config.triton_url = triton["grpc_url"] ? triton["grpc_url"].as<std::string>() : config.triton_url;
        config.detector_model_name = triton["detector_model_name"] ? triton["detector_model_name"].as<std::string>() : config.detector_model_name;
        config.recognizer_model_name = triton["recognizer_model_name"] ? triton["recognizer_model_name"].as<std::string>() : config.recognizer_model_name;
    }

    if (const auto source = root["source"]) {
        config.source = source.as<std::string>();
    }

    if (config.triton_url.empty()) {
        throw std::runtime_error("triton.grpc_url must not be empty");
    }

    return config;
}

