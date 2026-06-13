#include "runtime_config_loader.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {

std::string extractPlaceholderName(const std::string& value, const std::string& field_name)
{
    if (value.rfind("${", 0) != 0) {
        return "";
    }

    if (value.back() != '}') {
        throw std::runtime_error(
            "Malformed placeholder for " + field_name + ": " + value);
    }

    const std::string placeholder_name = value.substr(2, value.size() - 3);
    if (placeholder_name.empty()) {
        throw std::runtime_error(
            "Empty placeholder for " + field_name + ": " + value);
    }

    return placeholder_name;
}

}  // namespace

RuntimeConfigLoader::RuntimeConfigLoader(std::string yaml_path, std::string env_path)
    : yaml_path_(std::move(yaml_path)),
      env_path_(std::move(env_path)),
      env_values_(loadEnvFile(env_path_))
{
}

RuntimeConfig RuntimeConfigLoader::load() const
{
    const YAML::Node root = YAML::LoadFile(yaml_path_);
    RuntimeConfig config;

    if (const auto triton = root["triton"]) {
        config.triton_url = resolveNodeValue(
            triton["grpc_url"],
            "triton.grpc_url",
            config.triton_url);
        config.detector_model_name = resolveNodeValue(
            triton["detector_model_name"],
            "triton.detector_model_name",
            config.detector_model_name);
        config.recognizer_model_name = resolveNodeValue(
            triton["recognizer_model_name"],
            "triton.recognizer_model_name",
            config.recognizer_model_name);
    }

    config.source = resolveNodeValue(root["source"], "source", config.source);

    if (config.triton_url.empty()) {
        throw std::runtime_error("triton.grpc_url must not be empty");
    }

    return config;
}

std::unordered_map<std::string, std::string> RuntimeConfigLoader::loadEnvFile(
    const std::string& env_path)
{
    std::unordered_map<std::string, std::string> env_values;
    std::ifstream input(env_path);
    if (!input) {
        return env_values;
    }

    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed_line = trim(line);
        if (trimmed_line.empty() || trimmed_line.front() == '#') {
            continue;
        }

        const auto separator_pos = trimmed_line.find('=');
        if (separator_pos == std::string::npos) {
            throw std::runtime_error(
                "Invalid .env entry in " + env_path + ": " + trimmed_line);
        }

        const std::string key = trim(trimmed_line.substr(0, separator_pos));
        const std::string value = trim(trimmed_line.substr(separator_pos + 1));
        if (key.empty()) {
            throw std::runtime_error(
                "Invalid .env entry with empty key in " + env_path + ": " + trimmed_line);
        }

        env_values[key] = value;
    }

    return env_values;
}

std::string RuntimeConfigLoader::trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string RuntimeConfigLoader::resolveStringValue(
    const std::string& value,
    const std::string& field_name) const
{
    const std::string placeholder_name = extractPlaceholderName(value, field_name);
    if (placeholder_name.empty()) {
        return value;
    }

    if (const char* process_value = std::getenv(placeholder_name.c_str())) {
        return process_value;
    }

    const auto env_it = env_values_.find(placeholder_name);
    if (env_it != env_values_.end()) {
        return env_it->second;
    }

    throw std::runtime_error(
        "Missing environment variable for " + field_name + ": " + placeholder_name);
}

std::string RuntimeConfigLoader::resolveNodeValue(
    const YAML::Node& node,
    const std::string& field_name,
    const std::string& fallback) const
{
    if (!node) {
        return fallback;
    }

    return resolveStringValue(node.as<std::string>(), field_name);
}
