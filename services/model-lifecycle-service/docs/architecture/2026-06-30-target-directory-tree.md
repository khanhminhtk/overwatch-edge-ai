# Model Lifecycle Service Current Target Directory Tree

Tài liệu này cập nhật theo source hiện tại dưới `services/model-lifecycle-service/src`.

Mục tiêu của cây hiện tại:

- tách rõ `entrypoints`, `bootstrap`, `modules`, `platform`
- gom orchestration theo business module thay vì theo vendor
- để `job_control` làm shared workflow core cho mọi daemon
- để bootstrap/container là nơi wire dependency dùng chung

## Current Source Tree

```text
services/model-lifecycle-service/
├── config/
├── docs/
├── proto/
├── src/
│   ├── bootstrap/
│   │   ├── __init__.py
│   │   └── container.py
│   ├── entrypoints/
│   │   ├── continual_learning_daemon/
│   │   │   └── main.py
│   │   ├── dataset_daemon/
│   │   │   └── main.py
│   │   ├── export_onnx_daemon/
│   │   │   └── main.py
│   │   ├── kafka_ingestion/
│   │   │   └── main.py
│   │   ├── mlflow_download_daemon/
│   │   │   └── main.py
│   │   ├── mlflow_tracking_daemon/
│   │   │   └── main.py
│   │   └── training_daemon/
│   │       └── main.py
│   ├── modules/
│   │   ├── continual_learning/
│   │   │   ├── adapters/
│   │   │   ├── application/
│   │   │   ├── domain/
│   │   │   └── tests/
│   │   ├── dataset/
│   │   │   ├── adapters/
│   │   │   ├── application/
│   │   │   ├── domain/
│   │   │   └── tests/
│   │   ├── exports/
│   │   │   ├── adapters/
│   │   │   ├── application/
│   │   │   ├── domain/
│   │   │   └── tests/
│   │   ├── job_control/
│   │   │   ├── adapters/
│   │   │   │   ├── inbound/
│   │   │   │   └── outbound/
│   │   │   ├── application/
│   │   │   │   ├── dto/
│   │   │   │   ├── ports/
│   │   │   │   └── use_case/
│   │   │   ├── domain/
│   │   │   │   ├── entity_objects/
│   │   │   │   ├── errors/
│   │   │   │   └── value_objects/
│   │   │   ├── config.py
│   │   │   └── tests/
│   │   ├── tracking/
│   │   │   ├── adapters/
│   │   │   ├── application/
│   │   │   ├── domain/
│   │   │   └── tests/
│   │   └── training/
│   │       ├── adapters/
│   │       ├── application/
│   │       ├── domain/
│   │       └── tests/
│   ├── platform/
│   │   ├── config/
│   │   ├── logger/
│   │   ├── messaging/
│   │   │   └── kafka/
│   │   ├── persistence/
│   │   │   └── postgres/
│   │   ├── tracking/
│   │   │   └── mlflow/
│   │   └── vision/
│   └── shared_kernel/
├── src_origin/
└── tests_origin/
```

## Role Of Each Layer

- `bootstrap/`
  - chứa DI/runtime container dùng chung cho worker daemon
  - load config từ `.env` + yaml
  - start `PostgresPool`
  - build success-event publisher
  - resolve reclaim timeout theo config
  - handle shutdown signal và fail in-flight job

- `entrypoints/`
  - mỗi daemon là một process boundary rõ ràng
  - parse CLI args
  - gọi bootstrap/container
  - tạo use case/handler theo module
  - start polling loop hoặc ingestion consumer

- `modules/job_control/`
  - workflow core dùng chung
  - ingest Kafka message vào `kafka_events`
  - claim job từ SQL
  - persist `RECEIVED / PROCESSING / PROCESSED / FAILED`
  - publish success event dùng chung
  - chứa scheduler, repository adapter, DTO, port, query logic

- `modules/training/`
  - xử lý training jobs
  - decision theo `train_requested_recognizer` và `train_requested_detection`

- `modules/dataset/`
  - build detection/recognizer dataset jobs
  - tạo output đúng format cho `ml/training`

- `modules/continual_learning/`
  - OCR strong-model assisted processing
  - biến raw image thành detection/recognizer samples

- `modules/tracking/`
  - MLflow tracking workflow
  - MLflow artifact/registry/download orchestration

- `modules/exports/`
  - export ONNX cho detection và recognizer

- `platform/`
  - implementation concrete cho config, logging, Kafka, Postgres, MLflow, OCR
  - không chứa orchestration business theo job

- `shared_kernel/`
  - reserved shared primitives dùng cross-module
  - hiện còn rất mỏng

## Runtime Topology

Source hiện tại chạy theo topology này:

1. `kafka_ingestion.main`
   - consume Kafka request topics
   - validate payload
   - insert job vào bảng `kafka_events`

2. Worker daemons
   - `training_daemon`
   - `dataset_daemon`
   - `continual_learning_daemon`
   - `mlflow_tracking_daemon`
   - `mlflow_download_daemon`
   - `export_onnx_daemon`
   - các daemon này không consume request Kafka trực tiếp
   - chúng claim job từ SQL bằng `job_control`

3. Success event
   - sau khi job `PROCESSED`, daemon publish một success domain event vào topic dùng chung

## Key Architectural Notes

- `job_control` là shared workflow engine thực tế của service hiện tại.
- `bootstrap/container.py` là điểm wire dependency dùng chung, thay cho việc mỗi daemon tự lặp lại startup logic.
- Reclaim timeout của stale `PROCESSING` jobs đã được đưa vào config `job_control.reclaim`.
- Trạng thái lỗi chuẩn của hệ hiện tại vẫn là `FAILED`, không có enum `ERROR` riêng.
- Cây cũ mô tả `domain/application/adapter/infra` theo kiểu target refactor tổng quát; source hiện tại đã ổn định hơn theo `modules/* + platform/* + bootstrap/* + entrypoints/*`.

## What This Tree Does Not Claim

Tài liệu này không còn coi các flow sau là entrypoint/runtime hiện tại của source:

- `main_kafka_consumer.py`
- `main_training_worker.py`
- `main_export_worker.py`
- `pseudo_label_requested`
- `gold_data_requested`
- `evaluation_requested`
- deploy-topic orchestration end-to-end

Các flow đó có thể từng là hướng thiết kế hoặc legacy idea, nhưng không phản ánh entrypoint/source tree hiện tại trong `src/`.
