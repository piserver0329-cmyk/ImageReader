import io
import os
import logging
from flask import Flask, request, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError
import zxingcpp

# ---- config via env vars (set in the Render dashboard) ----
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", "2500"))  # px, longest side
API_KEY       = os.environ.get("API_KEY", "")                 # shared-secret auth

Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("barcode")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def decode_barcodes(raw: bytes, max_dim: int) -> list[dict]:
    """Decode every barcode/QR found in one image."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        im = ImageOps.exif_transpose(im)

        # Downscale huge images (never upscale); grayscale helps the detector + RAM
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        if im.mode != "L":
            im = im.convert("L")

        results = zxingcpp.read_barcodes(im)

    out = []
    for r in results:
        pos = r.position
        out.append({
            "text": r.text,
            "format": str(r.format),
            "content_type": str(r.content_type),
            "orientation_deg": r.orientation,
            "position": {
                "top_left":     [pos.top_left.x, pos.top_left.y],
                "top_right":    [pos.top_right.x, pos.top_right.y],
                "bottom_right": [pos.bottom_right.x, pos.bottom_right.y],
                "bottom_left":  [pos.bottom_left.x, pos.bottom_left.y],
            },
        })
    return out


@app.before_request
def auth():
    if request.path == "/health":
        return
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decode")
def decode():
    if "image" in request.files:
        raw = request.files["image"].read()
    else:
        raw = request.get_data()

    if not raw:
        return jsonify(error="no image data"), 400

    max_dim = int(request.args.get("max_dim", MAX_DIMENSION))

    try:
        barcodes = decode_barcodes(raw, max_dim)
    except UnidentifiedImageError:
        return jsonify(error="not a valid image"), 400
    except Image.DecompressionBombError:
        return jsonify(error="image too large (pixel bomb guard)"), 413
    except Exception as e:  # noqa: BLE001
        log.exception("decode failed")
        return jsonify(error=f"decode failed: {e}"), 500

    log.info("decoded %d byte image -> %d barcode(s)", len(raw), len(barcodes))
    return jsonify(count=len(barcodes), barcodes=barcodes)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))