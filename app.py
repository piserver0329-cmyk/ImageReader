import io
import os
import re
import logging

import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
import zxingcpp

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "16"))
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", "3500"))
MIN_DIMENSION = int(os.environ.get("MIN_DIMENSION", "1600"))
DETECT_WIDTH  = int(os.environ.get("DETECT_WIDTH", "1000"))   # downscale for detection
API_KEY       = os.environ.get("API_KEY", "")
OCR_API_KEY   = os.environ.get("OCR_API_KEY", "")
OCR_REGEX     = os.environ.get("OCR_REGEX", r"\b\d{8,12}\b")

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


# ---------- image prep ----------
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


# ---------- barcode region detection ----------
def _run_bounds(score: np.ndarray, frac: float = 0.35):
    """Longest contiguous run where score exceeds frac * max."""
    if score.max() <= 0:
        return 0, len(score)
    good = score > score.max() * frac
    best = (0, 0)
    cur = None
    for i, v in enumerate(good):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    if cur is not None and len(good) - cur > best[1] - best[0]:
        best = (cur, len(good))
    return best


def locate_barcode(im: Image.Image):
    """Return a padded (left, top, right, bottom) box around the striped region,
    in full-image coordinates, or None."""
    W, H = im.size
    scale = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
    small = im.resize((round(W * scale), round(H * scale)), Image.BILINEAR)
    a = np.asarray(small, dtype=np.int16)

    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    gy = np.abs(np.diff(a, axis=0, append=a[-1:, :]))
    grad = np.abs(gx.astype(np.int32) - gy.astype(np.int32)).astype(np.float32)

    # smooth so bar clusters merge into a blob
    gimg = Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8))
    gimg = gimg.filter(ImageFilter.BoxBlur(max(4, small.width // 40)))
    g = np.asarray(gimg, dtype=np.float32)

    mask = g > (g.mean() + g.std())
    if mask.sum() < 50:
        return None

    row_b = _run_bounds(mask.sum(axis=1))
    col_b = _run_bounds(mask.sum(axis=0))

    pad_x = int((col_b[1] - col_b[0]) * 0.25) + 20
    pad_y = int((row_b[1] - row_b[0]) * 0.35) + 20

    left   = max(0, int((col_b[0] - pad_x) / scale))
    right  = min(W, int((col_b[1] + pad_x) / scale))
    top    = max(0, int((row_b[0] - pad_y) / scale))
    bottom = min(H, int((row_b[1] + pad_y) / scale))

    if right - left < 40 or bottom - top < 20:
        return None
    return (left, top, right, bottom)


# ---------- decoding ----------
def _read(im):
    try:
        return zxingcpp.read_barcodes(im, formats=FORMATS)
    except TypeError:
        return zxingcpp.read_barcodes(im)


def _decode_variants(img: Image.Image):
    variants = [
        img,
        ImageOps.autocontrast(img, cutoff=1),
        img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3)),
    ]
    for v in variants:
        for scale in (1, 2, 3):
            big = v if scale == 1 else v.resize(
                (v.width * scale, v.height * scale), Image.LANCZOS)
            res = _read(big)
            if res:
                return res
            for angle in (90, 270, 5, -5, 10, -10, 15, -15, 45, -45):
                res = _read(big.rotate(angle, expand=True, fillcolor=255,
                                       resample=Image.BICUBIC))
                if res:
                    return res
    return []


def try_barcode(raw: bytes):
    base = _prep(raw)

    box = locate_barcode(base)
    if box:
        res = _decode_variants(base.crop(box))
        if res:
            return res, box

    # fall back to the whole image
    return _decode_variants(base), None


# ---------- OCR fallback ----------
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
        parsed = resp.json().get("ParsedResults") or []
        text = "\n".join(p.get("ParsedText", "") for p in parsed)
        return {"raw_text": text.strip(), "matches": re.findall(OCR_REGEX, text)}
    except Exception as e:  # noqa: BLE001
        log.exception("ocr failed")
        return {"error": str(e)}


# ---------- routes ----------
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
        results, box = try_barcode(raw)
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
        return jsonify(source="barcode", count=len(out),
                       auto_crop=box, barcodes=out)

    ocr = try_ocr(raw)
    if ocr and ocr.get("matches"):
        return jsonify(source="ocr", count=len(ocr["matches"]),
                       candidates=ocr["matches"], ocr_text=ocr["raw_text"])

    return jsonify(source="none", count=0, barcodes=[], auto_crop=box,
                   ocr=ocr, note="no barcode decoded and no number matched"), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))