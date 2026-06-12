#include "triton_grpc_client.hpp"

#include <stdexcept>
#include <utility>

#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
#include "grpc_client.h"
#endif

TritonGrpcClient::TritonGrpcClient(std::string server_url) : server_url_(std::move(server_url)) {}

void TritonGrpcClient::connect()
{
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
    triton::client::InferenceServerGrpcClient* raw_client = nullptr;
    const auto err = triton::client::InferenceServerGrpcClient::Create(&raw_client, server_url_, false);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to create Triton gRPC client: " + err.Message());
    }
    client_ = std::shared_ptr<triton::client::InferenceServerGrpcClient>(raw_client);
#else
    throw std::runtime_error("Triton gRPC client support is disabled at build time");
#endif
}

bool TritonGrpcClient::is_server_live() const
{
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
    if (!client_) {
        throw std::runtime_error("Triton client is not connected");
    }
    bool is_live = false;
    const auto err = client_->IsServerLive(&is_live);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to query Triton live state: " + err.Message());
    }
    return is_live;
#else
    return false;
#endif
}

bool TritonGrpcClient::is_server_ready() const
{
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
    if (!client_) {
        throw std::runtime_error("Triton client is not connected");
    }
    bool is_ready = false;
    const auto err = client_->IsServerReady(&is_ready);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to query Triton ready state: " + err.Message());
    }
    return is_ready;
#else
    return false;
#endif
}

bool TritonGrpcClient::is_model_ready(const std::string& model_name) const
{
#if defined(INFERENCE_RUNTIME_HAS_TRITON_GRPC)
    if (!client_) {
        throw std::runtime_error("Triton client is not connected");
    }
    bool is_ready = false;
    const auto err = client_->IsModelReady(&is_ready, model_name);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to query model ready state for " + model_name + ": " + err.Message());
    }
    return is_ready;
#else
    (void)model_name;
    return false;
#endif
}
