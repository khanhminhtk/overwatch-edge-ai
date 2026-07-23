# Deploy model code detail

## Danh sách thay đổi code

- File/Module: `runner.go`, `deployment_lock.go`.
- Chức năng thay đổi: lock model và telemetry latency pipeline.
- Mục đích thay đổi: ngăn concurrent deployment.

- File/Module: `domain/value_objects/deployment.go`, `application/dto/deploy_request.go`, `application/ports/deployment_runner.go`.
- Chức năng thay đổi: domain hóa format/policy, gom input `Run` vào DTO và khai báo inbound port.
- Mục đích thay đổi: tách mutable deployment parameters khỏi use case và tạo contract rõ ràng.

- File/Module: `move_model.go`.
- Chức năng thay đổi: copy staging, fsync, atomic rename, SHA-256 verification.
- Mục đích thay đổi: artifact immutable theo version.

- File/Module: `triton_lifecycle.go`, `triton_client.go`, `version_policy.go`.
- Chức năng thay đổi: poll readiness, rollback config + runtime, atomic config activation.
- Mục đích thay đổi: nhất quán config/runtime khi deploy lỗi.

## Chi tiết kỹ thuật quan trọng

- I/O contract được giữ/đổi: public `Run` giữ nguyên; HTTP client có timeout 10 giây, poll timeout 30 giây.
- Instrumentation log/độ trễ: log bắt đầu/kết thúc pipeline với `duration_ms`.
- Chiến lược xử lý lỗi: fail fast, giữ backup tới khi health pass, rollback config rồi unload/load runtime khi failure.

## Validation đã chạy

- Syntax/compile: `gofmt`.
- Unit tests: `go test ./edge/jetson/orchestrators/internal/modules/deploy_model/application/use_cases/runner`.
- Integration/API smoke: integration test thật trước thay đổi safeguards đã pass; không chạy lại tự động vì thay đổi repository Triton thật.
- Kết quả tổng: unit pass.
