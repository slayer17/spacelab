import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# UTILS
# =====================================================

def compute_signature(img):
    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))

    b = float(np.mean(small[:, :, 0]))
    g = float(np.mean(small[:, :, 1]))
    r = float(np.mean(small[:, :, 2]))

    return {
        "mean": mean,
        "std": std,
        "color": [b, g, r]
    }


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_quad(img, pts):
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    if maxW < 10 or maxH < 10:
        return None

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxW, maxH))

    return warp


# =====================================================
# STEP 1.1 MAIN CARD DETECTION
# =====================================================

def detect_main_card(img):
    max_dim = 1400

    h, w = img.shape[:2]
    scale = 1.0

    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 200)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    mask = cv2.bitwise_or(edges, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < image_area * 0.15:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(c)
            quad = cv2.boxPoints(rect).astype("float32")

        warp = warp_quad(img, quad)
        if warp is None:
            continue

        wh, ww = warp.shape[:2]
        if ww == 0 or wh == 0:
            continue

        ratio = wh / float(ww)
        if ratio < 1.3 or ratio > 1.65:
            continue

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        rect_area = bw * bh
        fill_ratio = area / float(rect_area)

        if fill_ratio < 0.75:
            continue

        candidates.append({
            "area": area,
            "rect_area": rect_area,
            "ratio": ratio,
            "fill": fill_ratio,
            "quad": quad,
            "bbox": [x, y, bw, bh]
        })

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["rect_area"], reverse=True)
    best = candidates[0]

    quad = best["quad"]

    if scale != 1.0:
        quad = quad / scale

    quad = order_points(quad)

    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))

    return {
        "x": int(x),
        "y": int(y),
        "w": int(bw),
        "h": int(bh),
        "type": "MAIN_CARD",
        "quad": quad.astype(int).tolist()
    }


# =====================================================
# ROUTES
# =====================================================
@app.route("/build_signatures")
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"})

    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"ok": False, "error": "Invalid image"})

    rect = detect_main_card(img)

    if rect is None:
        return jsonify({
            "ok": True,
            "rects": [],
            "warp": False,
            "signature": None
        })

    quad = np.array(rect["quad"], dtype="float32")
    warp = warp_quad(img, quad)

    sig = None

    if warp is not None:
        cv2.imwrite(
            "warp.jpg",
            warp,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        sig = compute_signature(warp)

    return jsonify({
        "ok": True,
        "rects": [rect],
        "warp": warp is not None,
        "signature": sig
    })


@app.route("/warp")
def warp_image():
    return send_from_directory(".", "warp.jpg")
#======================================================
# signature
#======================================================
import glob
import json


@app.route("/build_signatures")
def build_signatures():

    files = glob.glob("cards/*.jpg")
    files += glob.glob("cards/*.jpeg")
    files += glob.glob("cards/*.png")

    cards = []

    for f in files:

        img = cv2.imread(f)

        if img is None:
            continue

        h, w = img.shape[:2]

        quad = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")

        warp = warp_quad(img, quad)

        if warp is None:
            continue

        sig = compute_signature(warp)

        name = os.path.basename(f)

        cards.append({
            "id": name,
            "signature": {
                "scan": {
                    "globalSignature": sig
                }
            }
        })

    with open("cards.js", "w", encoding="utf-8") as f:

        f.write("window.CARDS = ")
        json.dump(cards, f, indent=2)

    return "OK signatures built"

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
