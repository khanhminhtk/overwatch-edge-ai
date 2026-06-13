#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime_config_loader.hpp"

namespace {

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write file: " + path);
    }
    out << content;
}

void unset_env(const char* name)
{
    unsetenv(name);
}

void set_env(const char* name, const char* value)
{
    setenv(name, value, 1);
}

void test_loads_plain_yaml_values()
{
    const std::string yaml_path = "/tmp/inference_runtime_plain.yaml";
    const std::string env_path = "/tmp/inference_runtime_plain.env";

    write_file(yaml_path, "triton:\n  grpc_url: localhost:8001\n  detector_model_name: detector\n  recognizer_model_name: recognizer\nsource: camera0\n");
    write_file(env_path, "# empty\n");

    RuntimeConfigLoader loader(yaml_path, env_path);
    const RuntimeConfig config = loader.load();

    expect(config.triton_url == "localhost:8001", "plain YAML triton_url mismatch");
    expect(config.detector_model_name == "detector", "plain YAML detector mismatch");
    expect(config.recognizer_model_name == "recognizer", "plain YAML recognizer mismatch");
    expect(config.source == "camera0", "plain YAML source mismatch");
}

void test_prefers_process_env_for_placeholder()
{
    const std::string yaml_path = "/tmp/inference_runtime_process.yaml";
    const std::string env_path = "/tmp/inference_runtime_process.env";

    write_file(yaml_path, "triton:\n  grpc_url: ${TRITON_GRPC_URL}\n  detector_model_name: detector\n  recognizer_model_name: recognizer\nsource: 0\n");
    write_file(env_path, "TRITON_GRPC_URL=env-file:8001\n");

    set_env("TRITON_GRPC_URL", "process-env:8001");
    RuntimeConfigLoader loader(yaml_path, env_path);
    const RuntimeConfig config = loader.load();
    unset_env("TRITON_GRPC_URL");

    expect(config.triton_url == "process-env:8001", "process env should override env file");
}

void test_falls_back_to_env_file_placeholder()
{
    const std::string yaml_path = "/tmp/inference_runtime_file.yaml";
    const std::string env_path = "/tmp/inference_runtime_file.env";

    write_file(yaml_path, "triton:\n  grpc_url: ${TRITON_GRPC_URL}\n  detector_model_name: ${DETECTOR_MODEL}\n  recognizer_model_name: recognizer\nsource: 0\n");
    write_file(env_path, "TRITON_GRPC_URL=file-env:8001\nDETECTOR_MODEL=detector-from-env\n");

    unset_env("TRITON_GRPC_URL");
    unset_env("DETECTOR_MODEL");

    RuntimeConfigLoader loader(yaml_path, env_path);
    const RuntimeConfig config = loader.load();

    expect(config.triton_url == "file-env:8001", "env file fallback triton_url mismatch");
    expect(config.detector_model_name == "detector-from-env", "env file fallback detector mismatch");
}

void test_throws_for_missing_placeholder_variable()
{
    const std::string yaml_path = "/tmp/inference_runtime_missing.yaml";
    const std::string env_path = "/tmp/inference_runtime_missing.env";

    write_file(yaml_path, "triton:\n  grpc_url: ${TRITON_GRPC_URL}\n  detector_model_name: detector\n  recognizer_model_name: recognizer\nsource: 0\n");
    write_file(env_path, "# empty\n");

    unset_env("TRITON_GRPC_URL");

    bool threw = false;
    try {
        RuntimeConfigLoader loader(yaml_path, env_path);
        static_cast<void>(loader.load());
    } catch (const std::exception& ex) {
        threw = std::string(ex.what()).find("TRITON_GRPC_URL") != std::string::npos;
    }

    expect(threw, "missing placeholder variable should throw");
}

}  // namespace

int main()
{
    test_loads_plain_yaml_values();
    test_prefers_process_env_for_placeholder();
    test_falls_back_to_env_file_placeholder();
    test_throws_for_missing_placeholder_variable();

    std::cout << "runtime_config_loader_test: PASS\n";
    return 0;
}
