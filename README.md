# Overwatch Edge AI — V1 Design README

> Tài liệu này mô tả **ý tưởng, kiến trúc và kế hoạch triển khai V1** cho một hệ thống OCR Edge AI realtime.  
> Các bảng benchmark, metric và kết quả chi tiết sẽ được cập nhật sau khi có code, training run và đo đạc thực tế.

---

## 1. Tổng quan

**Overwatch Edge AI** là project xây dựng hệ thống OCR Edge AI cho nhận diện vùng chữ / biển số theo hướng realtime.

Mục tiêu của V1 là tạo một pipeline end-to-end đủ thực tế để học và chứng minh năng lực về:

- Computer Vision
- OCR
- CTC sequence modeling
- Edge AI deployment
- TensorRT
- ONNX
- Model optimization
- MLOps cơ bản
- MLflow experiment tracking
- Go backend
- Kafka / MinIO
- Monitoring
- CI/CD ở mức phù hợp cho project

Hệ thống chạy inference chính trên **Jetson Orin Nano**. Camera chụp ảnh, YOLO phát hiện vùng chữ, OpenCV crop và chia vùng chữ thành các lát ảnh, sau đó ViT-CTC recognizer decode ra chuỗi ký tự.

Teacher model chỉ dùng cho **offline labeling**, không dùng làm fallback runtime.

---

## 2. Trạng thái tài liệu

README này là **bản thiết kế V1**, không phải báo cáo kết quả cuối cùng.

Vì project chưa hoàn thiện toàn bộ pipeline nên:

- Các thông số latency, accuracy, FPS, CER, mAP chưa được điền số cụ thể.
- Các bảng benchmark dùng `TBD` để chờ kết quả thực nghiệm.
- Các metric sẽ được cập nhật sau khi training, export ONNX, build TensorRT và benchmark trên Jetson.
- Model card và experiment report sẽ được sinh sau từng training run.

Quy ước:

```text
TBD = sẽ cập nhật sau khi có kết quả thực nghiệm thật
```

---

## 3. Ý tưởng chính

Hệ thống tách rõ detection và recognition:

```text
YOLO11n = phát hiện vùng chữ
ViT-CTC = nhận diện chuỗi ký tự trong vùng chữ
```

YOLO không nhận diện từng ký tự.

YOLO chỉ trả về bounding box của vùng chữ, ví dụ vùng chứa:

```text
59A12345
```

Sau đó OpenCV sẽ:

```text
crop vùng chữ
resize / normalize
chia crop thành 16 lát ảnh theo chiều ngang
```

Mỗi lát ảnh là một visual timestep. Recognizer encode các lát này thành chuỗi embedding, đưa qua Transformer, rồi dùng CTC để decode chuỗi ký tự.

---

## 4. Scope V1

V1 ưu tiên xây dựng một pipeline hoàn chỉnh nhưng vẫn đủ gọn để triển khai.

### Có trong V1

```text
[ ] Camera / ESP32-CAM input
[ ] Jetson Orin Nano inference
[ ] YOLO11n text-region detector
[ ] OpenCV crop và slicing
[ ] ViT-CTC sliced recognizer
[ ] CTC greedy decoding
[ ] PyTorch training
[ ] MLflow experiment tracking
[ ] ONNX export
[ ] ONNX validation
[ ] TensorRT FP16 build trên Jetson
[ ] Pruning experiment cơ bản
[ ] Quantization experiment cơ bản
[ ] Go backend ingest API
[ ] MinIO image/model storage
[ ] Kafka event pipeline
[ ] Offline teacher-labeling service
[ ] Retraining workflow cơ bản
[ ] Monitoring cơ bản bằng Prometheus/Grafana
[ ] CI/CD cơ bản cho lint, test, build, export check
```

### Chưa có trong V1

Các phần này để sang V2:

```text
[ ] Zero-downtime rollout
[ ] GPU temporary inference khi Jetson update model
[ ] Multi-device fleet rollout
[ ] Canary deployment
[ ] Shadow inference production
[ ] Automatic rollback
[ ] Full human review UI
[ ] Kubernetes production deployment
[ ] Advanced model registry service
[ ] Multi-camera orchestration
[ ] Full distributed tracing
```

---

## 5. Kiến trúc V1

```text
┌────────────────────┐
│ ESP32-CAM / Camera │
└─────────┬──────────┘
          ↓
┌────────────────────────────┐
│ Jetson Orin Nano           │
│                            │
│  Capture Service           │
│  YOLO11n Detector          │
│  OpenCV Crop/Slice         │
│  ViT-CTC OCR               │
│  TensorRT Runtime          │
│  Local TensorRT Builder    │
│  Metrics Exporter          │
└─────────┬──────────────────┘
          ↓
┌────────────────────┐
│ Go Backend         │
│                    │
│  Ingest API        │
│  Confidence Router │
│  Device Metadata   │
│  Model Metadata    │
└──────┬───────┬─────┘
       ↓       ↓
┌──────────┐ ┌────────────┐
│ MinIO    │ │ Kafka      │
│          │ │            │
│ Images   │ │ Events     │
│ Models   │ │ Review     │
│ Metadata │ │ Retrain    │
└────┬─────┘ └──────┬─────┘
     ↓              ↓
┌─────────────────────────┐
│ Teacher Labeling Service│
│                         │
│ LLM/VLM Teacher         │
│ Rule Validator          │
│ Label Export            │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│ GPU Training Server     │
│                         │
│ Train Detector          │
│ Train Recognizer        │
│ Evaluate                │
│ MLflow Tracking         │
│ Prune                   │
│ Quantize                │
│ Export ONNX             │
│ Validate ONNX           │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│ Model Artifact Storage  │
│                         │
│ PyTorch .pt             │
│ ONNX .onnx              │
│ Metrics JSON            │
│ Vocab                   │
│ Preprocess Config       │
│ Model Card              │
└─────────────────────────┘
```

---

## 6. Runtime Inference Flow

```text
1. Camera chụp ảnh.
2. Ảnh được gửi sang Jetson Orin Nano.
3. YOLO11n phát hiện bbox vùng chữ.
4. OpenCV crop vùng bbox.
5. OpenCV resize và slice crop thành nhiều lát theo chiều ngang.
6. ViT-CTC nhận sequence ảnh đã slice.
7. Recognizer xuất CTC logits.
8. CTC greedy decoder sinh chuỗi ký tự.
9. OpenCV vẽ bbox và text dự đoán lên ảnh.
10. Kết quả được gửi sang Go backend.
11. Backend route kết quả theo confidence.
12. Sample confidence cao được lưu vào accepted.
13. Sample confidence thấp được gửi vào Kafka để xử lý offline.
```

---

## 7. Detector: YOLO11n Text Region Detector

YOLO11n chỉ dùng để phát hiện vùng chữ.

YOLO output:

```json
{
  "bbox": [120, 80, 420, 160],
  "class": "text_region",
  "confidence": 0.96
}
```

Trong V1, detector có thể chỉ có một class:

```text
text_region
```

### Detector metrics cần đo sau khi train

```text
mAP50
mAP50-95
Precision
Recall
IoU
Crop success rate
```

Ghi chú:

```text
Các giá trị cụ thể sẽ được cập nhật sau khi detector được train và evaluate trên validation set.
```

---

## 8. OpenCV Crop và Slice

Sau khi YOLO detect bbox, OpenCV xử lý:

```text
1. Crop bbox từ ảnh gốc.
2. Thêm margin để tránh crop sát mất chữ.
3. Resize crop về kích thước cố định.
4. Normalize ảnh.
5. Chia crop thành N lát theo chiều ngang.
```

V1 mặc định thử nghiệm:

```text
N = 16 slices
```

Các biến thể sẽ được benchmark:

```text
16 slices
24 slices
32 slices
```

Input recognizer:

```text
[B, N, 3, H, W]
```

Ví dụ:

```text
[B, 16, 3, 128, 128]
```

hoặc:

```text
[B, 16, 3, 224, 224]
```

Kích thước input tốt nhất sẽ được quyết định bằng benchmark thực tế.

---

## 9. Recognizer: ViT-CTC Sliced Recognizer

Recognizer nhận các lát ảnh theo chiều ngang và decode thành text.

```text
Text crop
  ↓
N horizontal slices
  ↓
MobileNetV3 slice encoder
  ↓
Projection thành embedding 256 chiều
  ↓
ViT-style Transformer
  ↓
CTC head
  ↓
CTC decode
  ↓
Final text
```

### Tensor flow

```text
Input:
[B, N, 3, H, W]

MobileNetV3 feature encoder:
[B, N, 3, H, W] → [B, N, 576]

Projection:
[B, N, 576] → [B, N, 256]

Transformer:
[B, N, 256] → [B, N, 256]

CTC head:
[B, N, 256] → [B, N, vocab_size]

CTC decode:
[B, N, vocab_size] → text
```

### Thiết kế model dự kiến

```text
MobileNetV3 backbone: frozen hoặc fine-tune một phần
Projection: trainable
Transformer blocks: trainable
RMSNorm: trainable
Multi-Latent Attention: trainable
DeepSeek-style MoE: trainable
CTC head: trainable
```

Thông số model cụ thể sẽ được cập nhật sau khi finalize architecture.

---

## 10. CTC Decoding

Recognizer dùng CTC vì alignment giữa visual slice và ký tự output là không biết trước.

Ví dụ output theo timestep:

```text
blank 5 5 blank 9 blank A A 1 blank 2 3 4 5
```

CTC decode:

```text
59A12345
```

### Vocabulary

Vocab cơ bản:

```text
blank
0-9
A-Z
```

Có thể mở rộng:

```text
-
.
space
```

tùy dataset thực tế.

### Training loss

```text
CTCLoss
```

Input cho CTC:

```text
log_probs:      [T, B, C]
targets:        [sum_target_lengths]
input_lengths:  [B]
target_lengths: [B]
```

---

## 11. Confidence Routing

Backend route kết quả dựa vào confidence.

### Thành phần confidence

```text
detector_confidence
recognizer_confidence
regex_validity
final_confidence
```

Công thức V1 dự kiến:

```text
final_confidence =
    detector_confidence
  × recognizer_confidence
  × regex_confidence
```

Công thức này có thể thay đổi sau khi đo calibration thực tế.

### Routing policy

```text
if final_confidence >= threshold:
    lưu vào accepted
else:
    gửi vào review queue
```

Trong V1, threshold sẽ được cấu hình bằng config, chưa hard-code.

---

## 12. Go Backend

Go backend chịu trách nhiệm nhận kết quả OCR và route dữ liệu.

### Nhiệm vụ

```text
Nhận OCR result từ Jetson
Lưu metadata
Upload ảnh / kết quả vào MinIO
Gửi low-confidence samples vào Kafka
Theo dõi model version
Theo dõi device status
Expose API cơ bản
Expose metrics cho Prometheus
```

### API dự kiến

```text
POST /api/v1/ocr/result
GET  /api/v1/devices
GET  /api/v1/models
GET  /api/v1/samples/{id}
GET  /api/v1/metrics
```

### OCR result payload dự kiến

```json
{
  "sample_id": "uuid",
  "device_id": "jetson-001",
  "timestamp": "2026-04-28T10:00:00Z",
  "image_uri": "minio://accepted/yyyy/mm/dd/sample.jpg",
  "prediction": "TEXT_RESULT",
  "detector_confidence": 0.0,
  "recognizer_confidence": 0.0,
  "final_confidence": 0.0,
  "model_versions": {
    "detector": "yolo11n-text-vX.Y.Z",
    "recognizer": "vit-ctc-sliced-vX.Y.Z"
  },
  "bbox": [0, 0, 0, 0]
}
```

Các giá trị trong payload trên chỉ là schema mẫu.

---

## 13. Storage với MinIO

### Buckets dự kiến

```text
raw/
accepted/
review/
teacher_labeled/
verified/
retrain/
models/
```

### Ý nghĩa

```text
raw:
  ảnh gốc từ camera

accepted:
  kết quả confidence cao

review:
  sample confidence thấp hoặc invalid

teacher_labeled:
  nhãn được teacher model gợi ý

verified:
  nhãn đã được validate

retrain:
  dataset dùng cho retraining

models:
  model artifacts, metrics, vocab, config, model card
```

### Model artifact layout

```text
models/
  recognizer/
    vit-ctc-sliced-vX.Y.Z/
      model.pt
      model.onnx
      vocab.json
      preprocess_config.json
      metrics.json
      mlflow_run.json
      model_card.md
      checksum.sha256

  detector/
    yolo11n-text-vX.Y.Z/
      model.pt
      model.onnx
      metrics.json
      mlflow_run.json
      model_card.md
      checksum.sha256
```

---

## 14. Kafka Topics

Kafka dùng cho workflow bất đồng bộ.

### Topics dự kiến

```text
ocr.low_confidence
ocr.teacher_labeled
ocr.verified
ocr.retrain.ready
model.candidate.ready
device.status
```

### Low-confidence event schema

```json
{
  "event_id": "uuid",
  "event_type": "LOW_CONFIDENCE_SAMPLE",
  "sample_id": "uuid",
  "device_id": "jetson-001",
  "image_uri": "minio://review/sample.jpg",
  "prediction": "TEXT_RESULT",
  "final_confidence": 0.0,
  "reason": "LOW_RECOGNIZER_CONFIDENCE",
  "model_versions": {
    "detector": "yolo11n-text-vX.Y.Z",
    "recognizer": "vit-ctc-sliced-vX.Y.Z"
  }
}
```

---

## 15. Teacher Labeling Service

Teacher model chỉ dùng cho offline labeling.

Không dùng LLM/VLM làm runtime fallback.

### Quy tắc

```text
Runtime inference luôn dùng YOLO + ViT-CTC trên Jetson.
Teacher model chỉ dùng offline để gợi ý nhãn cho sample sai hoặc confidence thấp.
```

### Flow

```text
Low-confidence sample
  ↓
Teacher model gợi ý label
  ↓
Rule validator kiểm tra format
  ↓
Optional manual verification
  ↓
Verified sample
  ↓
Retrain dataset
```

### Label states

```text
PENDING_REVIEW
TEACHER_SUGGESTED
AUTO_VALIDATED
HUMAN_VERIFIED
REJECTED
RETRAIN_READY
```

---

## 16. MLflow và MLOps trong V1

V1 sử dụng **MLflow** để tracking experiment và quản lý thông tin training run.

MLflow trong V1 không cần triển khai quá phức tạp. Mục tiêu là ghi lại đầy đủ quá trình thử nghiệm để có thể so sánh và tái lập kết quả.

### MLflow tracking

Mỗi training run cần log:

```text
run_id
git_commit
model_name
model_version_candidate
dataset_version
dataset_hash
training_config
preprocess_config
hyperparameters
loss curves
evaluation metrics
model artifact path
onnx artifact path
notes
```

### Parameters cần log

```text
input_size
num_slices
batch_size
learning_rate
optimizer
scheduler
num_epochs
d_model
num_layers
backbone_type
freeze_backbone
vocab_version
augmentation_config
```

### Metrics cần log

Detector:

```text
mAP50
mAP50-95
precision
recall
validation_loss
```

Recognizer:

```text
train_loss
validation_ctc_loss
CER
sequence_accuracy
normalized_edit_distance
false_accept_rate
false_reject_rate
```

Optimization:

```text
model_size_mb
sparsity_ratio
onnx_validation_diff
tensorrt_build_status
latency_p50
latency_p95
```

Các latency chỉ được log sau khi đã benchmark thật.

### Artifacts cần log

```text
model.pt
model.onnx
vocab.json
preprocess_config.json
metrics.json
confusion_cases.json
failure_samples.csv
model_card.md
```

### MLOps boundary của V1

V1 làm:

```text
experiment tracking
artifact versioning
dataset version reference
model candidate metadata
basic model card
manual promotion
```

V1 chưa làm:

```text
automatic production promotion
canary rollout
automatic rollback
full model registry service
feature store
large-scale workflow orchestration
```

---

## 17. Training Pipeline

Training chạy trên GPU training server.

### Detector training

```text
Dataset: ảnh + bbox vùng chữ
Model: YOLO11n
Output: detector .pt và .onnx
Tracking: MLflow
```

### Recognizer training

```text
Dataset: crop vùng chữ + text label
Preprocess: crop → slice thành N cột
Model: ViT-CTC sliced recognizer
Loss: CTCLoss
Output: recognizer .pt và .onnx
Tracking: MLflow
```

### Training flow

```text
1. Load dataset.
2. Load training config.
3. Train model.
4. Log parameters vào MLflow.
5. Evaluate model.
6. Log metrics vào MLflow.
7. Export artifacts.
8. Export ONNX.
9. Validate ONNX.
10. Push artifact vào MinIO.
11. Tạo model card bản nháp.
```

---

## 18. Evaluation Metrics

### Detector metrics

```text
mAP50
mAP50-95
Precision
Recall
IoU
Crop recall
```

### Recognizer metrics

```text
CER - Character Error Rate
Sequence Accuracy
Normalized Edit Distance
False Accept Rate
False Reject Rate
CTC Loss
```

### System metrics

```text
end-to-end latency
camera FPS
detector latency
crop/slice latency
recognizer latency
CTC decode latency
backend ingest latency
accepted rate
review rate
```

Tất cả số liệu cụ thể sẽ cập nhật sau khi có benchmark thật.

---

## 19. Model Optimization

V1 có các thử nghiệm tối ưu model.

Mục tiêu là học trade-off giữa:

```text
accuracy
latency
model size
memory usage
deployment complexity
```

### Optimization pipeline

```text
PyTorch training
  ↓
Evaluation
  ↓
Pruning experiment
  ↓
Fine-tuning
  ↓
ONNX export
  ↓
ONNX validation
  ↓
FP16 TensorRT build trên Jetson
  ↓
Benchmark
  ↓
Log kết quả vào MLflow
```

---

## 20. Pruning

Pruning dùng để thử giảm model size và kiểm tra ảnh hưởng đến accuracy/latency.

### Unstructured pruning

```text
small weights → zero
```

Mức thử nghiệm dự kiến:

```text
10%
30%
50%
```

### Structured pruning

Có thể thử:

```text
d_model 256 → 192
d_model 256 → 128
layers 2 → 1
giảm MoE hidden size
giảm số lượng experts
```

### Fine-tuning sau pruning

```text
train full model
  ↓
prune
  ↓
fine-tune
  ↓
evaluate
  ↓
log vào MLflow
```

README sẽ cập nhật kết quả pruning sau khi có số liệu thật.

---

## 21. ONNX Export

ONNX là format trung gian để deploy model.

### Export flow

```text
PyTorch .pt
  ↓
ONNX .onnx
  ↓
ONNX Runtime validation
  ↓
MinIO model artifact storage
```

### Validation

So sánh output PyTorch và ONNX:

```text
output shape
max absolute difference
mean absolute difference
dynamic/static input compatibility
```

Ngưỡng validation sẽ được cấu hình trong file config.

Nếu ONNX validation fail, artifact không được xem là deployable.

---

## 22. TensorRT Deployment

TensorRT engine được build trên Jetson.

### Nguyên tắc V1

```text
Training server export ONNX.
Jetson build TensorRT engine local.
```

Lý do: TensorRT engine phụ thuộc vào hardware và runtime environment.

Có thể phụ thuộc vào:

```text
GPU architecture
CUDA version
TensorRT version
cuDNN version
plugins
precision mode
input shape
optimization profile
```

### Jetson TensorRT flow

```text
1. Pull ONNX từ MinIO.
2. Verify checksum.
3. Build FP16 TensorRT engine local.
4. Save engine file.
5. Run warmup inference.
6. Benchmark latency.
7. Dùng engine cho runtime inference.
8. Log benchmark vào MLflow hoặc metrics JSON.
```

### Artifacts trên Jetson

```text
/opt/overwatch/models/
  recognizer/
    vit-ctc-sliced-vX.Y.Z/
      model.onnx
      model_fp16.engine
      manifest.json
      benchmark.json
      build_log.txt

  detector/
    yolo11n-text-vX.Y.Z/
      model.onnx
      model_fp16.engine
      manifest.json
      benchmark.json
      build_log.txt
```

---

## 23. Quantization

V1 tập trung vào FP16 TensorRT.

### FP16

```text
FP32 → FP16 TensorRT
```

Mục tiêu:

```text
giảm latency
giảm memory
ít ảnh hưởng accuracy
phù hợp Jetson
```

### INT8 experiment

INT8 để dạng thử nghiệm.

Cần calibration dataset:

```text
text crops đại diện
ảnh ngày / đêm
ảnh mờ
ảnh nghiêng
nhiều điều kiện ánh sáng
```

INT8 chỉ được giữ lại nếu accuracy drop nằm trong ngưỡng cho phép.

Kết quả INT8 sẽ cập nhật sau khi thử nghiệm thật.

---

## 24. Monitoring và Observability

V1 cần monitoring cơ bản để hiểu hệ thống chạy ra sao.

### Edge metrics

```text
camera_fps
capture_latency_ms
yolo_latency_ms
opencv_crop_latency_ms
opencv_slice_latency_ms
recognizer_latency_ms
ctc_decode_latency_ms
total_inference_latency_ms
jetson_cpu_percent
jetson_memory_mb
jetson_temperature_celsius
```

### Backend metrics

```text
ingest_requests_total
ingest_latency_ms
accepted_samples_total
review_samples_total
low_confidence_rate
minio_upload_latency_ms
kafka_publish_latency_ms
```

### Training/MLOps metrics

```text
training_loss
validation_cer
sequence_accuracy
mAP50
model_size_mb
onnx_export_success
onnx_validation_diff
tensorrt_build_success
```

### Observability stack V1

```text
Prometheus
Grafana
Loki hoặc structured logs cơ bản
```

### V2 Observability

```text
OpenTelemetry
Tempo
Alertmanager
distributed tracing
advanced alerting
```

---

## 25. CI/CD cho V1

CI/CD phù hợp với project này, nhưng V1 chỉ nên làm ở mức vừa phải.

Không nên cố làm full production CI/CD ngay từ đầu.

### Mục tiêu CI/CD V1

```text
đảm bảo code không lỗi cơ bản
đảm bảo test chạy được
đảm bảo Docker image build được
đảm bảo model export script không vỡ
đảm bảo ONNX validation chạy được trên sample nhỏ
```

### Pipeline đề xuất

```text
lint
  ↓
unit test
  ↓
build Docker images
  ↓
run small integration test
  ↓
optional ONNX export smoke test
  ↓
push image/artifact nếu pass
```

### CI cho backend

```text
go fmt
go vet
go test
docker build backend
```

### CI cho AI code

```text
ruff / formatter
pytest
type check nếu có
small forward-pass test
small CTC decode test
ONNX export smoke test
```

### CI cho edge

```text
docker build edge-agent
check config schema
run mock inference test
```

### Không nên làm trong V1

```text
auto deploy production
auto promote model
auto rollout Jetson
auto rollback
complex Kubernetes pipeline
```

### GitLab CI/CD hoặc GitHub Actions

Có thể dùng một trong hai:

```text
GitLab CI/CD nếu repo nằm trên GitLab
GitHub Actions nếu repo nằm trên GitHub
```

V1 ưu tiên pipeline đơn giản, dễ debug.

---

## 26. Benchmark Plan

Các bảng dưới đây là **template**.  
Số liệu sẽ được cập nhật sau khi benchmark thật.

### Recognizer benchmark

| Variant | Runtime | Precision | Input Size | Slices | CER | Seq Acc | p50 Latency | p95 Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Base | PyTorch | FP32 | TBD | TBD | TBD | TBD | TBD | TBD |
| Small Input | PyTorch | FP32 | TBD | TBD | TBD | TBD | TBD | TBD |
| ONNX | ONNX Runtime | FP32 | TBD | TBD | TBD | TBD | TBD | TBD |
| TensorRT | TensorRT | FP16 | TBD | TBD | TBD | TBD | TBD | TBD |
| Pruned | TensorRT | FP16 | TBD | TBD | TBD | TBD | TBD | TBD |
| INT8 Experiment | TensorRT | INT8 | TBD | TBD | TBD | TBD | TBD | TBD |

### End-to-end benchmark

| Stage | Runtime | Precision | p50 Latency | p95 Latency |
|---|---|---|---:|---:|
| Capture | Camera | - | TBD | TBD |
| YOLO detector | TensorRT | FP16 | TBD | TBD |
| OpenCV crop/slice | CPU | FP32 | TBD | TBD |
| ViT-CTC recognizer | TensorRT | FP16 | TBD | TBD |
| CTC decode | CPU | FP32 | TBD | TBD |
| Backend upload | HTTP/gRPC | - | TBD | TBD |
| End-to-end | Mixed | Mixed | TBD | TBD |

---

## 27. Failure Cases

Project cần ghi nhận các case OCR thường lỗi:

```text
motion blur
low light
overexposure
text region quá nhỏ
crop mất ký tự
vùng chữ bị nghiêng
biển số bẩn
ký tự dễ nhầm: 0/O, 1/I, 5/S, 8/B, 2/Z
format không hợp lệ
detector confidence thấp
recognizer confidence thấp
```

Các failure cases sẽ được lưu lại để teacher labeling và retraining.

---

## 28. Dataset Format

### Detector dataset

```text
image_path
bbox
class_id
```

Ví dụ:

```json
{
  "image": "images/sample_001.jpg",
  "annotations": [
    {
      "class": "text_region",
      "bbox": [120, 80, 420, 160]
    }
  ]
}
```

### Recognizer dataset

```text
crop_image
text_label
```

Ví dụ:

```json
{
  "image": "crops/sample_001.jpg",
  "label": "59A12345"
}
```

Recognizer preprocessing pipeline sẽ tự chuyển crop thành slices trong lúc training.

---

## 29. Cấu trúc repo đề xuất

```text
overwatch-edge-ai/
  edge/
    capture/
    detector/
    slicer/
    recognizer/
    tensorrt_runtime/
    model_manager/
    metrics/

  ai_forge/
    models/
      detector/
      recognizer/
    training/
    pruning/
    quantization/
    export_onnx/
    evaluation/
    benchmarks/
    mlflow/

  backend/
    cmd/api/
    internal/ingest/
    internal/routing/
    internal/storage/
    internal/device/
    internal/model/
    internal/events/
    internal/metrics/

  labeling/
    teacher_service/
    validators/
    exporters/

  infra/
    docker/
    minio/
    kafka/
    postgres/
    prometheus/
    grafana/
    mlflow/

  ci/
    scripts/
    smoke_tests/

  docs/
    architecture.md
    model-design.md
    training.md
    optimization.md
    deployment.md
    active-learning.md
    mlops.md
    monitoring.md
    ci-cd.md
    failure-analysis.md
    model-card-template.md
```

---

## 30. Roadmap V1

### Phase 1 — OCR Baseline

```text
[ ] Chạy YOLO text-region detection
[ ] Crop vùng detect bằng OpenCV
[ ] Slice crop thành N cột
[ ] Chạy forward pass ViT-CTC
[ ] Thêm CTC head
[ ] Thêm greedy decoder
[ ] Output predicted text
```

### Phase 2 — Training + MLflow

```text
[ ] Chuẩn bị detector dataset
[ ] Chuẩn bị recognizer dataset
[ ] Train YOLO11n detector
[ ] Train ViT-CTC recognizer
[ ] Log experiment vào MLflow
[ ] Evaluate CER và sequence accuracy
[ ] Save model artifacts
```

### Phase 3 — Backend và Storage

```text
[ ] Build Go ingest API
[ ] Upload result images vào MinIO
[ ] Store metadata
[ ] Publish low-confidence samples vào Kafka
[ ] Thêm confidence routing cơ bản
```

### Phase 4 — ONNX và TensorRT

```text
[ ] Export detector sang ONNX
[ ] Export recognizer sang ONNX
[ ] Validate ONNX outputs
[ ] Build FP16 TensorRT engine trên Jetson
[ ] Run TensorRT inference
[ ] Benchmark latency
[ ] Log benchmark artifact
```

### Phase 5 — Optimization

```text
[ ] Thử input size nhỏ hơn
[ ] Thử số slice khác nhau
[ ] Thử pruning
[ ] Fine-tune pruned model
[ ] Thử FP16 TensorRT
[ ] Optional INT8 calibration experiment
[ ] Log kết quả vào MLflow
```

### Phase 6 — Teacher Labeling và Retrain

```text
[ ] Collect low-confidence samples
[ ] Dùng teacher model gợi ý label
[ ] Validate label bằng rule
[ ] Export verified retrain dataset
[ ] Retrain recognizer
[ ] So sánh old model vs new model bằng MLflow
```

### Phase 7 — Monitoring

```text
[ ] Thêm edge metrics
[ ] Thêm backend metrics
[ ] Thêm Prometheus
[ ] Thêm Grafana dashboard
[ ] Theo dõi latency và review rate
```

### Phase 8 — CI/CD cơ bản

```text
[ ] Backend lint/test/build
[ ] AI code lint/test
[ ] Forward-pass smoke test
[ ] ONNX export smoke test
[ ] Docker build check
[ ] Config validation
```

---

## 31. Definition of Done cho V1

V1 hoàn thành khi:

```text
[ ] Camera image xử lý được end-to-end.
[ ] YOLO detect được text region.
[ ] OpenCV crop và slice được vùng chữ.
[ ] ViT-CTC predict được text.
[ ] CTC greedy decode hoạt động.
[ ] Training run được log bằng MLflow.
[ ] Jetson chạy inference local.
[ ] Go backend nhận OCR result.
[ ] MinIO lưu accepted và review samples.
[ ] Kafka nhận low-confidence events.
[ ] Recognizer export được sang ONNX.
[ ] ONNX output được validate với PyTorch.
[ ] Jetson build và chạy được FP16 TensorRT engine.
[ ] Có pruning experiment cơ bản.
[ ] Có quantization experiment cơ bản.
[ ] Có metrics cơ bản trên Grafana.
[ ] Có CI/CD cơ bản cho lint, test, build, smoke test.
[ ] README có architecture, benchmark template và failure cases.
```

---

## 32. Ý tưởng cho V2

V2 tập trung vào rollout an toàn và reliability.

```text
[ ] Zero-downtime model update
[ ] GPU temporary inference trong lúc Jetson build TensorRT
[ ] Jetson gateway maintenance mode
[ ] Safe rollout state machine
[ ] Shadow inference
[ ] Canary rollout
[ ] Automatic rollback
[ ] Multi-device model rollout
[ ] Full model registry service
[ ] Human review UI
[ ] OpenTelemetry tracing
[ ] Advanced alerting
[ ] Kubernetes deployment
```

---

## 33. Tóm tắt

Overwatch Edge AI không chỉ là một model OCR.

Đây là một project học toàn bộ vòng đời Edge AI:

```text
detect
crop
slice
recognize
decode
route
store
track experiment
label
retrain
optimize
export
deploy
monitor
test
```

Mục tiêu của V1 là xây dựng một pipeline end-to-end thực tế, có thể chạy trên Jetson Orin Nano, có backend, có data loop, có MLflow/MLOps cơ bản, có monitoring và có CI/CD vừa đủ.

Các số liệu chi tiết sẽ được cập nhật sau khi có implementation và benchmark thật.
