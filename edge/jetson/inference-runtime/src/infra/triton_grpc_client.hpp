#pragma once

#include <memory>
#include <string>

namespace triton::client {
class InferenceServerGrpcClient;
}

class TritonGrpcClient {
  public:
    explicit TritonGrpcClient(std::string server_url);

    void connect();
    bool is_server_live() const;
    bool is_server_ready() const;
    bool is_model_ready(const std::string& model_name) const;

  private:
    std::string server_url_;
    std::shared_ptr<triton::client::InferenceServerGrpcClient> client_;
};
