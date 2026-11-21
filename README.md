# Nhận diện độ chín của chuối

Đồ án môn **CS231 – Nhập môn Thị giác máy tính**, Trường Đại học Công nghệ Thông tin (UIT).

Dự án phân loại độ chín của chuối từ hình ảnh và so sánh ba hướng tiếp cận: **SVM**, **CNN** và **MobileNetV2**. Mục tiêu của đồ án là xây dựng các mô hình thị giác máy tính, đánh giá chúng trên dữ liệu ảnh chuối và tạo một web demo để người dùng thử dự đoán.

## Chạy web demo trên Windows

Các bước dưới đây dành cho người mới, chạy ứng dụng trên máy cá nhân. Bạn không cần tải dataset để dùng demo; các checkpoint mô hình đã nằm trong thư mục `models/` của repository.

### 1. Cài Python

Cài **Python 3.11 (64-bit)** từ [python.org](https://www.python.org/downloads/). Trong bộ cài Windows, chọn **Add Python to PATH**. Mở Command Prompt mới và kiểm tra:

```bat
py -3.11 --version
```

Bạn cũng có thể dùng Anaconda Prompt nếu đã cài Anaconda.

### 2. Tải mã nguồn

Mở Command Prompt tại thư mục muốn lưu dự án, rồi chạy:

```bat
git clone https://github.com/Thinhng04/CS231.git
cd CS231
```

Nếu máy chưa có Git, cài [Git for Windows](https://git-scm.com/download/win), sau đó mở lại Command Prompt.

### 3. Tạo môi trường Python và cài thư viện

Tạo một môi trường riêng cho demo rồi kích hoạt nó:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Cài các thư viện cần thiết cho cả SVM, CNN và MobileNetV2:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements-models.txt
```

Quá trình cài đặt có thể mất vài phút vì TensorFlow và PyTorch khá lớn. Hãy giữ môi trường `.venv` được kích hoạt khi chạy demo.

### 4. Khởi động web

```bat
python server.py
```

Khi terminal báo server đang chạy, mở trình duyệt tại:

**http://127.0.0.1:8000**

Để dừng server, quay lại cửa sổ terminal và nhấn `Ctrl+C`.

### 5. Thử phân loại ảnh

1. Tải ảnh chuối lên hoặc kéo ảnh vào vùng tải ảnh.
2. Chọn họ mô hình: SVM, CNN hoặc MobileNetV2.
3. Chọn phiên bản model nếu có nhiều phiên bản.
4. Nhấn **Phân tích ảnh** để xem nhãn dự đoán và xác suất của bốn lớp: Green, Semi-ripe, Ripe, Overripe.

Ảnh được xử lý trên máy local. Demo nhận ảnh JPG, PNG, WEBP và BMP; giới hạn kích thước file là 15 MB.

### Nếu gặp lỗi thường thấy

- **`py` hoặc `python` không được nhận diện:** cài lại Python và chọn **Add Python to PATH**, rồi mở terminal mới.
- **Thiếu thư viện:** xác nhận `.venv` đã được kích hoạt (đầu dòng lệnh thường có `(.venv)`), sau đó chạy lại `python -m pip install -r requirements-models.txt`.
- **Cổng 8000 đang được sử dụng:** chạy `python server.py --port 8001`, rồi mở `http://127.0.0.1:8001`.
- **Model không khả dụng:** kiểm tra checkpoint tương ứng có trong `models/` và cài dependencies bằng `requirements-models.txt`.

## Cấu trúc chính

- `server.py`: HTTP server local và API nhận ảnh, trả kết quả.
- `static/`: giao diện HTML, JavaScript và CSS.
- `inference.py`: tiền xử lý ảnh, nạp checkpoint và chạy dự đoán.
- `models.json`: danh sách model và đường dẫn checkpoint.
- `models/`: trọng số đã huấn luyện cho SVM, CNN và MobileNetV2.
- `notebook/`: các notebook và gói kết quả thí nghiệm.
- `requirements.txt`, `requirements-models.txt`: danh sách thư viện Python.

## Các phương pháp và nhánh phát triển

- **SVM:** phân loại bằng đặc trưng màu HSV, kết cấu LBP hoặc kết hợp HSV + LBP.
- **CNN:** ba cấu hình CNN, gồm baseline, CNN dùng global average pooling và CNN có batch normalization/dropout.
- **MobileNetV2:** các phiên bản đóng băng backbone, fine-tuning và fine-tuning có tăng cường dữ liệu.

Các nhánh `SVM`, `CNN` và `MobileNet` lưu quá trình phát triển từng hướng; nhánh `main` tổng hợp web demo và các tài liệu dùng chung.

## Lưu ý về kết quả

Các phần trăm trên giao diện là xác suất do mô hình ước lượng cho từng lớp, không phải cam kết rằng dự đoán chắc chắn đúng. Khi đánh giá hoặc so sánh mô hình, hãy dùng cùng tập test và các chỉ số như accuracy, macro-F1, cùng confusion matrix.
