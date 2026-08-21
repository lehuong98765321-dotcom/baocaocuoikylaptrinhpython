# DT25 Stage 7 EDA Execution Package

## Mục đích
Package chạy EDA cho bảy câu hỏi đã khóa, từ các CSV được nghiệm thu ở Giai đoạn 4 và 6. Package không chạy K-Means, không chọn k, không tạo PCA và không đặt tên cụm.

## File cần upload
Upload đúng năm file khi Cell 4 yêu cầu:

1. `online_retail_clean.csv`
2. `online_retail_rfm_eligible.csv`
3. `online_retail_cancellations_returns.csv`
4. `rfm_customers.csv`
5. `rfm_with_scores.csv`

Notebook kiểm tra tên, schema và baseline trước EDA. Không tự sửa file input.

## Cách chạy
1. Mở `DT25_Stage7_EDA_Execution.ipynb` bằng Google Colab.
2. Chọn **Runtime > Run all**.
3. Upload năm file thật ở Cell 4.
4. Không chỉnh số liệu, baseline, interpretation records hoặc Acceptance Summary.
5. Nếu toàn bộ test đạt, Cell 25 tải `DT25_Stage7_EDA_Evidence.zip`.

## Điều chỉnh tối thiểu Q07
Câu Q07 gốc cần cluster assignments, nhưng Giai đoạn 7 cấm chạy K-Means và chưa có cluster file. Package giữ mục tiêu profiling bằng baseline tổ hợp điểm RFM, ghi thay đổi đầy đủ vào `question_change_log.csv`, không coi điểm RFM là cụm cuối và hoãn chiến lược theo cụm tới giai đoạn profiling sau clustering.

## Peer review
Notebook không giả mạo xác nhận thành viên. Mọi câu giữ `TECHNICALLY PASS, PEER REVIEW PENDING` cho tới khi người kiểm tra chéo ký/xác nhận ngoài pipeline.

## Evidence gửi lại
Ưu tiên `DT25_Stage7_EDA_Evidence.zip`. Nếu giao diện không nhận ZIP, xuất TXT chứa manifest, SHA-256, critical evidence, source code, registry, bảng kết quả, metadata hình và acceptance outputs.

## Khi test FAIL
Không xóa assertion, không tạo dữ liệu mẫu và không sửa Acceptance Summary bằng tay. Giữ traceback, execution log và chỉ thực hiện patch kỹ thuật tối thiểu, sau đó Restart and run all từ đầu.
