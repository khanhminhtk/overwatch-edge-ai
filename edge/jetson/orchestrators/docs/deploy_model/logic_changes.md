# Deploy model production safeguards

## Tóm tắt thay đổi logic

- Mục tiêu thay đổi: làm deployment Triton có kiểm soát đồng thời, integrity check và rollback runtime.
- Phạm vi thay đổi: `internal/modules/deploy_model/application/use_cases/runner`.
- Ngoài phạm vi: TLS/auth và persistent deployment job store.

## Luồng cũ vs luồng mới

- Hành vi cũ: copy trực tiếp, gọi Triton một lần, chỉ khôi phục config khi lỗi.
- Hành vi mới: lock từng model, copy qua file staging rồi atomic rename, SHA-256 xác minh artifact tồn tại, poll readiness/health, reload runtime sau rollback và xóa backup chỉ sau health pass.
- Lý do cần thay đổi: tránh partial deploy, overwrite artifact khác nội dung và race giữa các deploy.

## Tác động kỹ thuật

- Adapter/Port/Domain nào bị tác động: filesystem + Triton HTTP outbound adapter nằm trong runner; domain thêm `ModelFormat` và `DeploymentPolicy`; port `DeploymentRunner` định nghĩa inbound contract.
- Tương thích ngược: `Runner.Run` nay nhận một `dto.DeployRequest` thay vì các tham số rời rạc.
- Rủi ro và cách giảm thiểu: health global vẫn phụ thuộc toàn bộ Triton; polling có timeout 30 giây và rollback runtime được thực hiện khi failure.
