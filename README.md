# Nhận diện độ chín của chuối

Đồ án môn **CS231 – Nhập môn Thị giác máy tính**, Trường Đại học Công nghệ Thông tin (UIT).

Dự án phân loại độ chín của chuối từ hình ảnh, hướng đến xây dựng và so sánh ba phương pháp: **SVM**, **CNN** và **MobileNet**.

## Mục tiêu

- Xây dựng các mô hình phân loại độ chín của chuối.
- Đánh giá, so sánh các mô hình trên cùng cách chia dữ liệu train/validation/test.

## Các nhánh

- `main`: Tổng hợp code và tài liệu chung của dự án.
- `SVM`: Phát triển phương pháp sử dụng SVM.
- `CNN`: Phát triển mô hình CNN.
- `MobileNet`: Phát triển mô hình sử dụng MobileNet.

## Các notebook hiện có

- `CNN_Improved_Banana.ipynb`

Các notebook phục vụ khảo sát dữ liệu và thử nghiệm mô hình. Dự án đang trong quá trình phát triển.

## Cách sử dụng

1. Clone repository:

   ```bash
   git clone https://github.com/Thinhng04/CS231.git
   cd CS231
   ```

2. Chuyển sang nhánh cần làm việc, ví dụ:

   ```bash
   git switch CNN
   ```

3. Mở notebook tương ứng bằng Jupyter Notebook, Google Colab hoặc Kaggle.
4. Chuẩn bị dataset, chỉnh đường dẫn dữ liệu và cài các thư viện theo notebook.
5. Chạy các ô theo thứ tự để tiền xử lý, huấn luyện và đánh giá.

Dataset và trọng số mô hình cần được chuẩn bị riêng nếu không có trong repository. Khi so sánh các phương pháp, sử dụng cùng cách chia dữ liệu; dùng chung `manifest_used.csv` nếu notebook có hỗ trợ.

## Cập nhật code

Sau khi chuyển sang đúng nhánh và chỉnh sửa code:

```bash
git add .
git commit -m "Update model"
git push
```

Nếu nhánh chưa có upstream, chạy `git push -u origin <ten-nhanh>` ở lần push đầu tiên, ví dụ `git push -u origin CNN`.
