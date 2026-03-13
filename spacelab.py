import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# UTILS
# =====================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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

    if maxW < 20 or maxH < 20:
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


def resize_for_detection(img, max_dim=1400):
    h, w = img.shape[:2]
    scale = 1.0

    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img, scale


def contour_to_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")

    # fallback plus robuste : minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype="float32")


# =====================================================
# STEP 1.1 - DETECT MAIN CARD
# =====================================================

def detect_main_card(img):
    original = img.copy()
    img_small, scale = resize_for_detection(img, max_dim=1400)

    h, w = img_small.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Double approche pour être plus stable photo + scan
    edges = cv2.Canny(blur, 60, 180)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )

    combined = cv2.bitwise_or(edges, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        # Ignore petits contours
        if area < image_area * 0.08:
            continue

        quad = contour_to_quad(c)
        if quad is None or len(quad) != 4:
            continue

        warp = warp_quad(img_small, quad)
        if warp is None:
            continue

        wh, ww = warp.shape[:2]
        if ww == 0 or wh == 0:
            continue

        ratio = wh / float(ww)
        rect_area = ww * wh

        # ratio carte portrait : assez large pour accepter photo/scans
        if ratio < 1.20 or ratio > 1.75:
            continue

        # score : on préfère le plus grand rectangle plausible
        box = cv2.boundingRect(quad.astype(np.int32))
        x, y, bw, bh = box

        fill_ratio = area / float(max(bw * bh, 1))  # rectangulaire ?
        if fill_ratio < 0.65:
            continue

        candidates.append({
            "contour_area": area,
            "rect_area": rect_area,
            "fill_ratio": fill_ratio,
            "ratio": ratio,
            "quad": quad,
            "bbox": [x, y, bw, bh]
        })

    if not candidates:
        return None

    # plus grand candidat plausible
    candidates.sort(key=lambda c: c["rect_area"], reverse=True)
    best = candidates[0]

    quad = best["quad"]

    # remettre à l’échelle d’origine
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
        "quad": quad.astype(int).tolist(),
        "debug": {
            "ratio": float(best["ratio"]),
            "fill_ratio": float(best["fill_ratio"]),
            "rect_area": int(best["rect_area"])
        }
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():
    try:
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
                "mode": "single",
                "rects": [],
                "message": "No main card found"
            })

        return jsonify({
            "ok": True,
            "mode": "single",
            "rects": [rect]
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"ok": False, "error": str(e)})


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
