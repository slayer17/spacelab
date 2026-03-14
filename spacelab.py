import cv2
import numpy as np
import os
import json

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, "cards")
WARP_PATH = os.path.join(BASE_DIR, "warp.jpg")
CARDS_JS_PATH = os.path.join(BASE_DIR, "cards.js")


# =====================================================
# UTILS
# =====================================================

def compute_signature(img):

    img = cv2.resize(img, (200, 300))

    h, w = img.shape[:2]

    # ======================
    # ROI interne (enlever bord)
    # ======================

    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    y1 = int(h * 0.05)
    y2 = int(h * 0.95)

    img = img[y1:y2, x1:x2]

    h, w = img.shape[:2]

    # ======================
    # COLOR (haut gauche)
    # ======================

    x1 = int(w * 0.00)
    x2 = int(w * 0.55)

    y1 = int(h * 0.00)
    y2 = int(h * 0.30)

    zone = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    color_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "color": zone.mean(axis=(0, 1)).tolist()
    }

    # ======================
    # SYMBOL
    # ======================

    x1 = int(w * 0.00)
    x2 = int(w * 0.35)

    y1 = int(h * 0.30)
    y2 = int(h * 0.55)

    zone = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    symbol_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }

    # ======================
    # BOTTOM
    # ======================

    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    y1 = int(h * 0.70)
    y2 = int(h * 1.00)

    zone = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    bottom_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }

    # ======================
    # GLOBAL
    # ======================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }

    return {
        "color": color_sig,
        "symbol": symbol_sig,
        "bottom": bottom_sig,
        "global": global_sig
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

    warp = cv2.warpPerspective(
        img,
        M,
        (maxW, maxH)
    )

    return warp


def crop_percent(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    xa = max(0, min(w, int(w * x1)))
    xb = max(0, min(w, int(w * x2)))
    ya = max(0, min(h, int(h * y1)))
    yb = max(0, min(h, int(h * y2)))

    if xb <= xa or yb <= ya:
        return None

    return img[ya:yb, xa:xb]


def compute_signature_safe(img):
    if img is None or img.size == 0:
        return None
    return compute_signature(img)


def load_cards_js():
    with open(CARDS_JS_PATH, "r", encoding="utf-8") as f:
        txt = f.read()

    txt = txt.replace("window.CARDS =", "", 1).strip()

    if txt.endswith(";"):
        txt = txt[:-1]

    return json.loads(txt)


def save_cards_js(cards):
    with open(CARDS_JS_PATH, "w", encoding="utf-8") as f:
        f.write("window.CARDS = ")
        json.dump(cards, f, indent=2, ensure_ascii=False)


def find_card_image(card_id):
    base = card_id.lower()
    for ext in [".jpeg", ".jpg", ".png"]:
        path = os.path.join(CARDS_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


# =====================================================
# DETECT MAIN CARD
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

    edges = cv2.Canny(blur, 60, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < image_area * 0.15:
            continue

        if area > image_area * 0.98:
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

        if ratio < 1.2 or ratio > 1.8:
            continue

        candidates.append((area, quad))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    quad = candidates[0][1]

    if scale != 1.0:
        quad = quad / scale

    quad = order_points(quad)

    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))

    return {
        "x": int(x),
        "y": int(y),
        "w": int(bw),
        "h": int(bh),
        "quad": quad.astype(int).tolist()
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/test")
def test():
    return "OK TEST"


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)


# =====================================================
# UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return jsonify({
                "rects": [],
                "signature": None,
                "error": "no image"
            })

        file = request.files["image"]

        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "rects": [],
                "signature": None,
                "error": "invalid image"
            })

        mode = request.form.get("mode", "BOARD")

        # Pour l’instant on utilise la même détection pour les 2 modes.
        # En photo carte seule, il faut détecter la carte.
        # En scan propre, la carte prendra naturellement presque toute l’image.
        rect = detect_main_card(img)

        if rect is None:
            return jsonify({
                "rects": [],
                "signature": None,
                "mode": mode
            })

        quad = np.array(rect["quad"], dtype="float32")
        warp = warp_quad(img, quad)

      sig = None
      rois = []

      if warp is not None:

          cv2.imwrite(WARP_PATH, warp)

          sig, rois = compute_signature(warp)

        return jsonify({
            "rects": [rect],
            "signature": sig,
            "mode": mode
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({
            "rects": [],
            "signature": None,
            "error": str(e)
        }), 500


@app.route("/warp")
def warp():
    if not os.path.exists(WARP_PATH):
        return "warp not found", 404
    return send_from_directory(BASE_DIR, "warp.jpg")


# =====================================================
# BUILD SIGNATURES
# =====================================================

@app.route("/build_signatures")
def build_signatures():

    with open("cards.js", "r", encoding="utf-8") as f:
        txt = f.read()

    txt = txt.replace("window.CARDS =", "").strip()

    if txt.endswith(";"):
        txt = txt[:-1]

    cards = json.loads(txt)

    for c in cards:

        name = c["id"].lower() + ".jpeg"

        path = os.path.join("cards", name)

        if not os.path.exists(path):
            continue

        img = cv2.imread(path)

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

        c["signature"] = {
            "scan": sig
               
            }
        

    with open("cards.js", "w", encoding="utf-8") as f:

        f.write("window.CARDS = ")
        json.dump(cards, f, indent=2)

    return "OK"


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
