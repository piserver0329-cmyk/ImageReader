import io
import os
import re
import logging

import requests
from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
import zxingcpp

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "16"))
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", "3500"))
MIN_DIMENSION = int(os.environ.get("MIN_DIMENSION", "1600"))
API_KEY       = os.environ.get("API_KEY", "")
OCR_API_KEY   = os.environ.get("OCR_API_KEY", "")           # from ocr.space
OCR_REGEX     = os.environ.get("OCR_REGEX", r"\b\d{8,12}\b")  # tracking-no shape

Image.MAX_IMAGE_PIXELS = 60_000_000
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("barcode")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

FORMATS = (
    zxingcpp.BarcodeFormat.Code39 | zxingcpp.BarcodeFormat.Code128
    | zxingcpp.BarcodeFormat.Code93 | zxingcpp.BarcodeFormat.ITF
    | zxingcpp.BarcodeFormat.Codabar | zxingcpp.BarcodeFormat.EAN13
    | zxingcpp.BarcodeFormat.EAN8 | zxingcpp.BarcodeFormat.UPCA
    | zxingcpp.BarcodeFormat.UPCE | zxingcpp.BarcodeFormat.QRCode
    | zxingcpp.BarcodeFormat.DataMatrix | zxingcpp.BarcodeFormat.PDF417
    | zxingcpp.BarcodeFormat.Aztec
)


def _prep(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        im = ImageOps.exif_transpose(im).convert("L")
        w, h = im.size
        longest = max(w, h)
        if longest > MAX_DIMENSION:
            s = MAX_DIMENSION / longest
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        elif longest < MIN_DIMENSION:
            s = MIN_DIMENSION / longest
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        return im


def _read(im):
    try:
        return zxingcpp.read_barcodes(im, formats=FORMATS)
    except TypeError:
        return zxingcpp.read_barcodes(im)


def try_barcode(raw: bytes):
    base = _prep(raw)
    variants = [
        base,
        ImageOps.autocontrast(base, cutoff=1),
        base.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)),
    ]
    for img in variants:
        res = _read(img)
        if res:
            return res
        for angle in (90, 270, 5, -5, 10, -10, 15, -15, 45, -45):
            res = _read(img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC))
            if res:
                return res
    return []


def try_ocr(raw: bytes):
    if not OCR_API_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": ("image.jpg", raw)},
            data={"apikey": OCR_API_KEY, "OCREngine": "2", "scale": "true"},
            timeout=30,
        )
        data = resp.json()
        parsed = data.get("ParsedResults") or []
        text = "\n".join(p.get("ParsedText", "") for p in parsed)
        matches = re.findall(OCR_REGEX, text)
        return {"raw_text": text.strip(), "matches": matches}
    except Exception as e:  # noqa: BLE001
        log.exception("ocr failed")
        return {"error": str(e)}


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
    raw = request.files["image"].read() if "image" in request.files else request.get_data()
    if not raw:
        return jsonify(error="no image data"), 400

    try:
        results = try_barcode(raw)
    except UnidentifiedImageError:
        return jsonify(error="not a valid image"), 400
    except Image.DecompressionBombError:
        return jsonify(error="image too large"), 413
    except Exception as e:  # noqa: BLE001
        log.exception("decode failed")
        return jsonify(error=f"decode failed: {e}"), 500

    if results:
        out = []
        for r in results:
            p = r.position
            out.append({
                "text": r.text,
                "format": str(r.format),
                "orientation_deg": getattr(r, "orientation", None),
                "position": {
                    "top_left": [p.top_left.x, p.top_left.y],
                    "top_right": [p.top_right.x, p.top_right.y],
                    "bottom_right": [p.bottom_right.x, p.bottom_right.y],
                    "bottom_left": [p.bottom_left.x, p.bottom_left.y],
                },
            })
        return jsonify(source="barcode", count=len(out), barcodes=out)

    ocr = try_ocr(raw)
    if ocr and ocr.get("matches"):
        return jsonify(source="ocr", count=len(ocr["matches"]),
                       candidates=ocr["matches"], ocr_text=ocr["raw_text"])

    return jsonify(source="none", count=0, barcodes=[],
                   ocr=ocr, note="no barcode decoded and no number matched"), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))