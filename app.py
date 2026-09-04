import io
import os
import logging

from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
import zxingcpp

# ---- config via env vars (set in the Render dashboard) ----
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "16"))
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", "3500"))   # px, longest side
MIN_DIMENSION = int(os.environ.get("MIN_DIMENSION", "1400"))   # upscale smaller images
API_KEY       = os.environ.get("API_KEY", "")                  # shared-secret auth

Image.MAX_IMAGE_PIXELS = 60_000_000  # decompression-bomb guard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("barcode")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# 1-D formats common on courier / retail labels, plus QR
FORMATS = (
    zxingcpp.BarcodeFormat.Code39
    | zxingcpp.BarcodeFormat.Code128
    | zxingcpp.BarcodeFormat.Code93
    | zxingcpp.BarcodeFormat.ITF
    | zxingcpp.BarcodeFormat.Codabar
    | zxingcpp.BarcodeFormat.EAN13
    | zxingcpp.BarcodeFormat.EAN8
    | zxingcpp.BarcodeFormat.UPCA
    | zxingcpp.BarcodeFormat.UPCE
    | zxingcpp.BarcodeFormat.QRCode
    | zxingcpp.BarcodeFormat.DataMatrix
    | zxingcpp.BarcodeFormat.PDF417
    | zxingcpp.BarcodeFormat.Aztec
)


def _prep(raw: bytes) -> Image.Image:
    """Decode bytes -> normalized grayscale PIL image sized for barcode reading."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        im = ImageOps.exif_transpose(im)          # honor camera orientation
        im = im.convert("L")                      # grayscale

        w, h = im.size
        longest = max(w, h)

        # shrink very large photos to protect the 512 MB dyno
        if longest > MAX_DIMENSION:
            scale = MAX_DIMENSION / longest
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        # gently upscale tiny images so thin bars have enough pixels
        elif longest < MIN_DIMENSION:
            scale = MIN_DIMENSION / longest
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        return im


def _read(im: Image.Image):
    """Older zxing-cpp API: try_rotate / try_downscale default to True."""
    try:
        return zxingcpp.read_barcodes(im, formats=FORMATS)
    except TypeError:
        # very old builds don't accept the `formats` kwarg
        return zxingcpp.read_barcodes(im)


def decode_barcodes(raw: bytes) -> list[dict]:
    """Try a few cheap enhancement passes; return the first non-empty result."""
    base = _prep(raw)

    candidates = [
        base,
        ImageOps.autocontrast(base, cutoff=1),
        base.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)),
        ImageOps.invert(base),
    ]

    results = []
    for img in candidates:
        results = _read(img)
        if results:
            break

    out = []
    for r in results:
        p = r.position
        out.append({
            "text": r.text,
            "bytes_hex": r.bytes.hex() if getattr(r, "bytes", None) else None,
            "format": str(r.format),
            "orientation_deg": getattr(r, "orientation", None),
            "position": {
                "top_left":     [p.top_left.x, p.top_left.y],
                "top_right":    [p.top_right.x, p.top_right.y],
                "bottom_right": [p.bottom_right.x, p.bottom_right.y],
                "bottom_left":  [p.bottom_left.x, p.bottom_left.y],
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

    try:
        barcodes = decode_barcodes(raw)
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