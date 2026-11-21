"""Local HTTP server for the static Banana Ripeness demo."""

import argparse
import base64
import io
import json
import mimetypes
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from inference import LABELS, catalog, predict


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_PREVIEW_SIDE = 1600


def read_rgb(source):
    """Decode bytes or a path and apply EXIF orientation without resizing."""
    if isinstance(source, (bytes, bytearray)):
        stream = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        stream = source
    else:
        raise ValueError("Hãy tải lên một ảnh hợp lệ.")

    try:
        opened = Image.open(stream)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Không đọc được ảnh đầu vào.") from exc

    try:
        if opened.width * opened.height > MAX_PIXELS:
            raise ValueError("Ảnh quá lớn. Hãy chọn ảnh dưới 25 megapixel.")
        oriented = ImageOps.exif_transpose(opened)
        try:
            rgb = oriented.convert("RGB")
        finally:
            if oriented is not opened:
                oriented.close()
    finally:
        opened.close()

    return rgb


def read_image(source):
    """Create a detailed preview, shrinking large images but never enlarging."""
    rgb = read_rgb(source)
    longest_side = max(rgb.size)
    if longest_side > MAX_PREVIEW_SIDE:
        scale = MAX_PREVIEW_SIDE / longest_side
        size = (round(rgb.width * scale), round(rgb.height * scale))
        preview = rgb.resize(size, Image.Resampling.LANCZOS)
    else:
        preview = rgb.copy()
    rgb.close()
    return preview


def ordered_probabilities(scores):
    """Return scores in the UI's fixed four-class order."""
    values = [float(scores.get(label, 0.0)) for label in LABELS]
    if not set(scores).issubset(LABELS):
        raise ValueError("Model trả về nhãn không thuộc bốn lớp của demo.")
    if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-3:
        raise ValueError("Output của model không phải phân phối xác suất hợp lệ.")
    return [
        {"label": label, "probability": values[index]}
        for index, label in enumerate(LABELS)
    ]


def _preview_data_url(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class Handler(BaseHTTPRequestHandler):
    server_version = "BananaDemo/1.0"

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Kích thước request không hợp lệ.") from exc
        if length <= 0:
            raise ValueError("Request không có dữ liệu ảnh.")
        if length > MAX_UPLOAD_BYTES:
            raise OverflowError("Vui lòng chọn ảnh nhỏ hơn 15 MB.")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Dữ liệu ảnh tải lên chưa đầy đủ.")
        return body

    def do_GET(self):
        request = urlsplit(self.path)
        if request.path == "/api/models":
            return self._send_json(200, {"models": catalog()})

        relative = "index.html" if request.path == "/" else unquote(request.path).lstrip("/")
        candidate = (STATIC / relative).resolve()
        if STATIC.resolve() not in candidate.parents or not candidate.is_file():
            return self._send_json(404, {"error": "Không tìm thấy tài nguyên."})

        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",):
            content_type += "; charset=utf-8"
        self._send(200, candidate.read_bytes(), content_type)

    def do_POST(self):
        request = urlsplit(self.path)
        try:
            raw = self._read_body()
            if request.path == "/api/preview":
                image = read_image(raw)
                try:
                    return self._send_json(
                        200,
                        {
                            "preview": _preview_data_url(image),
                            "width": image.width,
                            "height": image.height,
                        },
                    )
                finally:
                    image.close()

            if request.path == "/api/predict":
                model_id = parse_qs(request.query).get("model", [""])[0]
                model = next((item for item in catalog() if item["id"] == model_id), None)
                if model is None:
                    return self._send_json(400, {"error": "Hãy chọn một model hợp lệ."})
                if not model["available"]:
                    return self._send_json(409, {"error": "Chưa có checkpoint cho model này."})

                image = read_rgb(raw)
                started = time.perf_counter()
                try:
                    result = predict(image, model_id)
                finally:
                    image.close()
                scores = result["scores"] or {result["label"]: 1.0}
                return self._send_json(
                    200,
                    {
                        "probabilities": ordered_probabilities(scores),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        "model": model["name"],
                        "has_probabilities": result["scores"] is not None,
                        "note": result.get("note", ""),
                    },
                )

            return self._send_json(404, {"error": "Không tìm thấy API."})
        except OverflowError as exc:
            return self._send_json(413, {"error": str(exc)})
        except ImportError as exc:
            return self._send_json(
                503,
                {"error": f"Thiếu thư viện {exc.name}. Cài requirements-models.txt rồi khởi động lại."},
            )
        except (OSError, ValueError) as exc:
            return self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self._send_json(
                500, {"error": f"Không chạy được model: {type(exc).__name__}: {exc}"}
            )

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Banana Ripeness Classification")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Banana demo running at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Banana demo...", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
