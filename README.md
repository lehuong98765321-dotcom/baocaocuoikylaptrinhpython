# Nhóm 5 - Đề tài DT25

## Phân khúc khách hàng bán lẻ trực tuyến bằng mô hình RFM và thuật toán K-Means

### Thông tin học phần

- Học phần: Lập trình Python cho Phân tích Dữ liệu
- Lớp: 25CNTT3
- Giảng viên phụ trách: Nguyễn Hoàng Hải
- Nhóm thực hiện: Nhóm 5

### Thành viên

1. Nguyễn Hữu Lộc - MSSV: 7257984 - Nhóm trưởng
2. Phan Thị Diệu Hoa - MSSV: 312054787
3. Lê Thị Thu Phương - MSSV: 43574984
4. Trần Văn Thanh - MSSV: 23859742

### Phân công

- Nguyễn Hữu Lộc: tích hợp, K-Means và đánh giá mô hình.
- Phan Thị Diệu Hoa: đọc dữ liệu, audit và tiền xử lý.
- Lê Thị Thu Phương: EDA, trực quan hóa và diễn giải.
- Trần Văn Thanh: RFM, SQLite, kiểm thử và README.

Tỷ lệ đóng góp đề xuất: 25% mỗi thành viên, tổng 100%.

### Dữ liệu

Dự án sử dụng bộ dữ liệu UCI Online Retail.

Các trường chính:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

### Quy trình

1. Kiểm tra dữ liệu đầu vào và schema.
2. Audit chất lượng dữ liệu.
3. Xử lý trùng lặp, đơn hủy, số lượng âm và dữ liệu không hợp lệ.
4. Tạo tập giao dịch sạch.
5. Tính Recency, Frequency và Monetary.
6. Thực hiện phân tích khám phá dữ liệu.
7. Chuẩn bị dữ liệu cho K-Means.
8. Đánh giá mô hình bằng Elbow, Silhouette và các metric liên quan.

### Kết quả đã nghiệm thu

- Dữ liệu thô: 541.909 dòng.
- Dữ liệu sạch: 524.878 dòng.
- Dữ liệu đủ điều kiện RFM: 392.692 dòng.
- Khách hàng RFM: 4.338.
- CustomerID trùng trong bảng RFM: 0.
- Missing trong bảng RFM: 0.
- Manual RFM validation: 5/5 PASS.
- EDA: 7/7 câu hỏi technically passed.
- Output read-back: PASS.
- Chart validation: PASS.
- Pytest: PASS.

### Báo cáo

- /Nhom05_DT25_BaoCao.pdf
- /Nhom05_DT25_BaoCao.docx

### Công cụ hỗ trợ

Nhóm sử dụng Microsoft Copilot và Claude để hỗ trợ lập kế hoạch, rà soát cấu trúc, xây dựng mã và kiểm tra tính nhất quán. Các thành viên chịu trách nhiệm chạy lại, kiểm tra và giải thích toàn bộ sản phẩm khi vấn đáp.

### Trạng thái kỹ thuật

Các kết quả Cleaning, RFM và EDA đã có evidence thực thi. Kết quả K-Means cuối và SQLite cần được chạy và kiểm chứng đầy đủ trước khi xem dự án là hoàn thiện toàn bộ DT25.
