# Deploy model session context

## Phân tích vấn đề hiện tại

- Vấn đề hoặc khoảng trống: direct copy/config mutation không có lock, integrity, retry hoặc runtime rollback.
- Nguyên nhân gốc: runner là orchestration prototype nhưng được dùng để deploy Triton thật.
- Cách tiếp cận xử lý: transaction theo model repository với lock, staging, poll và rollback.

## Mục tiêu session

- Nâng cấp các safeguard production từ staging đến cleanup backup.

## Nhật ký thay đổi

- Người/agent thực hiện: Codex.
- Thời gian: 2026-07-22.
- Module tác động: deploy model runner.
- Tóm tắt hành động: thêm lock, checksum, atomic promotion, timeout/polling, reload khi rollback, domain deployment policy, DTO và port.

## Bối cảnh cho session tiếp theo

- Giả định đã chốt: Triton repository là local shared filesystem.
- Vấn đề còn tồn đọng: authentication/TLS và persistent deployment records.
- Bước tiếp theo ưu tiên: tách filesystem/Triton HTTP thành ports/adapters nếu module được mở rộng.
