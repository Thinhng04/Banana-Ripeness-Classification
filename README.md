# Banana Ripeness Classification — local web demo

Demo gồm giao diện tĩnh trong `static/`, một HTTP server local và các checkpoint
trong `models/`. Server cung cấp các API `/api/models`, `/api/preview` và
`/api/predict` mà giao diện gọi khi tải ảnh và phân tích.

## Chạy bằng Anaconda

Mở **Anaconda Prompt**, kích hoạt môi trường đã cài các thư viện (ví dụ `CV`),
rồi chạy:

```bat
conda activate CV
cd /d "C:\My Work\UIT\HK5\CV\Đồ án CK\Demo\web"
python server.py
```

Mở **http://127.0.0.1:8000**. Dừng server bằng `Ctrl+C`. Có thể đổi cổng bằng
`python server.py --port 8001`.

Nếu chưa cài dependencies cho model, khi đang ở đúng Conda environment chạy:

```bat
python -m pip install -r requirements-models.txt
```

`start.bat` cũng chạy `python server.py`; hãy mở nó từ Anaconda Prompt đã
activate môi trường `CV` để chắc chắn dùng đúng Python.

## Ảnh và model

Ảnh tải lên được xử lý trên máy local, không cần có trong dataset. Demo nhận
ảnh JPG, PNG, WEBP hoặc BMP; ảnh được xoay theo EXIF và tạo preview độ phân giải
cao, giữ nguyên kích thước ảnh nhỏ và chỉ thu nhỏ ảnh có cạnh dài
hơn 1600 px. Model chỉ phân loại bốn lớp: Green, Semi-ripe, Ripe,
Overripe.

Danh sách model và checkpoint tương ứng được cấu hình trong `models.json`.
Checkpoint bị loại khỏi Git vì dung lượng lớn; cần có các file trong `models/`
để chạy inference. SVM chỉ trả về nhãn dự đoán, giao diện biểu diễn kết quả
đó thành điểm 100% cho nhãn được chọn. CNN và MobileNetV2 trả về softmax.

## Dependencies

`requirements.txt` cài Pillow và NumPy. `requirements-models.txt` bổ sung
OpenCV, scikit-image, scikit-learn, TensorFlow, PyTorch và torchvision để chạy
đủ ba họ model.
