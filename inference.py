"""Inference adapters that follow the preprocessing in Demo's three notebooks."""

import json
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LABELS = ["Green", "Semi-ripe", "Ripe", "Overripe"]
LOCK = RLock()
CACHE = {}


def config():
    return json.loads((ROOT / "models.json").read_text(encoding="utf-8"))


def model_path(item):
    return ROOT / item["path"]


def catalog():
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "family": item["family"],
            "detail": item["detail"],
            "available": model_path(item).is_file(),
        }
        for item in config()
    ]


def _open_rgb(source, *, transpose_exif):
    if isinstance(source, Image.Image):
        opened = source.copy()
    elif isinstance(source, (str, Path)):
        try:
            opened = Image.open(source)
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Không đọc được ảnh đầu vào.") from exc
    else:
        raise ValueError("Hãy tải lên một ảnh hợp lệ.")

    try:
        if opened.width * opened.height > 25_000_000:
            raise ValueError("Ảnh quá lớn. Hãy chọn ảnh dưới 25 megapixel.")
        normalized = ImageOps.exif_transpose(opened) if transpose_exif else opened
        try:
            return normalized.convert("RGB")
        finally:
            if normalized is not opened:
                normalized.close()
    finally:
        opened.close()


def _svm_input_bgr(source):
    import cv2

    rgb = np.asarray(_open_rgb(source, transpose_exif=False))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_with_pad_bgr(image, size=(128, 128)):
    """Copy of svm_kaggle.resize_with_pad_bgr, including its valid-pixel mask."""
    import cv2

    target_h, target_w = size
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Ảnh có kích thước không hợp lệ.")
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    right = target_w - resized_w - left
    bottom = target_h - resized_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    valid_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    valid_mask[top : top + resized_h, left : left + resized_w] = 255
    return padded, valid_mask


def _svm_features(image, feature_name):
    import cv2
    from skimage.feature import local_binary_pattern

    image, valid_mask = _resize_with_pad_bgr(image)

    def extract_hsv():
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], valid_mask, [16, 8], [0, 180, 0, 256]
        )
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    def extract_lbp():
        ys, xs = np.where(valid_mask > 0)
        if not len(xs) or not len(ys):
            raise ValueError("Không tìm thấy vùng ảnh hợp lệ để tính LBP.")
        roi = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        bins = 10  # P + 2, exactly as in svm_kaggle.ipynb
        hist, _ = np.histogram(
            lbp.ravel(), bins=bins, range=(0, bins), density=True
        )
        return hist.astype(np.float32)

    if feature_name == "hsv":
        values = extract_hsv()
    elif feature_name == "lbp":
        values = extract_lbp()
    elif feature_name == "hsv_lbp":
        values = np.hstack([extract_hsv(), extract_lbp()])
    else:
        raise ValueError("Bộ đặc trưng SVM không hợp lệ.")
    return values.reshape(1, -1)


def _resize_pad_mobile(image, size, fill=(128, 128, 128)):
    """Copy of mobilenet_kaggle.ResizePad for validation/inference images."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Ảnh có kích thước không hợp lệ.")
    scale = size / max(width, height)
    new_w = min(size, max(1, int(round(width * scale))))
    new_h = min(size, max(1, int(round(height * scale))))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def load_model(item):
    path = model_path(item)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có checkpoint {item['path']}. Xem README.md trong Demo/web."
        )
    key = (str(path), path.stat().st_mtime_ns)
    if key in CACHE:
        return CACHE[key]

    if item["family"] == "SVM":
        import joblib

        loaded = joblib.load(path)
    elif item["family"] == "CNN":
        import tensorflow as tf

        loaded = tf.keras.models.load_model(path, compile=False)
    elif item["family"] == "MobileNetV2":
        import torch
        from torchvision.models import mobilenet_v2

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = mobilenet_v2(weights=None)
        network.classifier[1] = torch.nn.Linear(network.classifier[1].in_features, 4)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        loaded = (network, checkpoint)
    else:
        raise ValueError(f"Họ mô hình không được hỗ trợ: {item['family']}")

    CACHE[key] = loaded
    return loaded


def _validated_scores(values, classes):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(classes) != 4 or set(classes) != set(LABELS) or values.shape != (4,):
        raise ValueError("Model phải có đúng bốn lớp theo thứ tự nhãn trong notebook.")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isclose(values.sum(), 1, atol=1e-3)
    ):
        raise ValueError("Output của model không phải phân phối softmax hợp lệ.")
    return {label: float(values[index]) for index, label in enumerate(classes)}


def predict(image_source, model_id):
    item = next((model for model in config() if model["id"] == model_id), None)
    if item is None:
        raise ValueError("Hãy chọn một model hợp lệ.")

    with LOCK:
        model = load_model(item)
        if item["family"] == "SVM":
            if not hasattr(model, "predict_proba"):
                raise ValueError("Checkpoint SVM chưa hỗ trợ predict_proba.")
            bgr = _svm_input_bgr(image_source)
            features = _svm_features(bgr, item["feature"])
            values = model.predict_proba(features)[0]
            model_classes = getattr(model, "classes_", None)
            if model_classes is None:
                raise ValueError("Checkpoint SVM không có danh sách lớp classes_.")
            try:
                classes = [item["class_names"][int(class_id)] for class_id in model_classes]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError("Checkpoint SVM có mã lớp không hợp lệ.") from exc
            scores = _validated_scores(values, classes)
            label = max(scores, key=scores.get)
            return {
                "label": label,
                "scores": scores,
                "note": "SVM · xác suất được ước lượng bằng predict_proba.",
            }

        if item["family"] == "CNN":
            import tensorflow as tf

            image = _open_rgb(image_source, transpose_exif=True)
            shape = model.input_shape
            batch = tf.image.resize_with_pad(
                tf.cast(np.asarray(image), tf.float32),
                shape[1],
                shape[2],
                method="bilinear",
                antialias=True,
            )[None, ...]
            # The CNN notebook puts Rescaling(1/255) inside each saved model.
            values = model(batch, training=False).numpy()[0]
            classes = item["class_names"]
            note = "CNN · ảnh RGB 128×128, bilinear resize-with-pad; model tự chuẩn hóa pixel 0–255."
        else:
            import torch
            from torchvision import transforms

            network, checkpoint = model
            image = _open_rgb(image_source, transpose_exif=False)
            size = int(checkpoint["image_size"])
            padded = _resize_pad_mobile(image, size)
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
            with torch.inference_mode():
                logits = network(transform(padded).unsqueeze(0))
                values = logits.softmax(dim=1)[0].cpu().numpy()
            classes = checkpoint.get("classes", item["class_names"])
            note = (
                f"MobileNetV2 · ảnh RGB {size}×{size}, bilinear letterbox nền xám; "
                "chuẩn hóa ImageNet."
            )

        scores = _validated_scores(values, classes)
        label = max(scores, key=scores.get)
        return {"label": label, "scores": scores, "note": note}
