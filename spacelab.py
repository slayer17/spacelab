import os
import json
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, "cards")
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols")
CARDS_JS_PATH = os.path.join(BASE_DIR, "cards.js")
WARP_PATH = os.path.join(BASE_DIR, "warp.jpg")


# =====================================================
# SYMBOL DETECTION
# =====================================================

def detect_symbol(zone):
    templates = {
        "SCIENTIFIQUE": cv2.imread(os.path.join(SYMBOLS_DIR, "scientifique.png"), 0),
        "ASTRONAUTE": cv2.imread(os.path.join(SYMBOLS_DIR, "astronaute.png"), 0),
        "MECANICIEN": cv2.imread(os.path.join(SYMBOLS_DIR, "mecanicien.png"), 0),
        "MEDECIN": cv2.imread(os.path.join(SYMBOLS_DIR, "medecin.png"), 0),
    }

    if zone is None or zone.size == 0:
        return None, 0.0

    h, w = zone.shape[:2]

    # Crop encore un peu plus centré
    x1 = int(w * 0.18)
    x2 = int(w * 0.82)
    y1 = int(h * 0.18)
    y2 = int(h * 0.82)

    zone = zone[y1:y2, x1:x2]

    if zone is None or zone.size == 0:
        return None, 0.0

    # Préparation image scan
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)

    # Binarisation pour isoler la forme
    _, scan_bin = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Petit nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    scan_bin = cv2.morphologyEx(scan_bin, cv2.MORPH_OPEN, kernel)
    scan_bin = cv2.morphologyEx(scan_bin, cv2.MORPH_CLOSE, kernel)

    best_name = None
    best_score = -1.0

    for name, tpl in templates.items():
        if tpl is None or tpl.size == 0:
            continue

        th, tw = tpl.shape[:2]

        tx1 = int(tw * 0.18)
        tx2 = int(tw * 0.82)
        ty1 = int(th * 0.18)
        ty2 = int(th * 0.82)

        tpl = tpl[ty1:ty2, tx1:tx2]

        if tpl is None or tpl.size == 0:
            continue

        tpl = cv2.GaussianBlur(tpl, (3, 3), 0)
        tpl = cv2.equalizeHist(tpl)
        tpl = cv2.resize(tpl, (64, 64), interpolation=cv2.INTER_CUBIC)

        _, tpl_bin = cv2.threshold(
            tpl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        tpl_bin = cv2.morphologyEx(tpl_bin, cv2.MORPH_OPEN, kernel)
        tpl_bin = cv2.morphologyEx(tpl_bin, cv2.MORPH_CLOSE, kernel)

        # Score gris
        res_gray = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        score_gray = float(res_gray.max())

        # Score masque binaire
        res_bin = cv2.matchTemplate(scan_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
        score_bin = float(res_bin.max())

        # Score combiné : on favorise la forme
        score = (score_gray * 0.35) + (score_bin * 0.65)

        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        return None, 0.0

    return best_name, best_score


# =====================================================
# COLOR DETECTION
# =====================================================

def detect_card_color(zone):
    """
    Détection robuste de couleur à partir des pixels saturés.
    Retourne :
      - couleur détectée
      - debug counts
      - moyenne BGR brute
    """
    if zone is None or zone.size == 0:
        return "ROUGE", {"reason": "empty"}, [0.0, 0.0, 0.0]

    mean_bgr = zone.mean(axis=(0, 1)).tolist()

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Garde les pixels assez colorés et assez visibles
    mask = (s > 60) & (v > 50)

    hues = h[mask]

    if len(hues) == 0:
        return "ROUGE", {"reason": "no_saturated_pixels"}, mean_bgr

    counts = {
        "ROUGE": int(np.sum((hues <= 10) | (hues >= 170))),
        "JAUNE": int(np.sum((hues >= 15) & (hues <= 35))),
        "VERT": int(np.sum((hues >= 40) & (hues <= 85))),
        "BLEU": int(np.sum((hues >= 90) & (hues <= 130))),
    }

    detected = max(counts, key=counts.get)

    return detected, counts, mean_bgr


# =====================================================
# SMALL PATCH SIGNATURE
# =====================================================

def compute_patch_signature(zone, size=(16, 16)):
    if zone is None or zone.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "vector": []
        }

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "vector": small.flatten().astype(float).tolist()
    }


# =====================================================
# CORE SIGNATURE
# =====================================================

def compute_signature(img):
    rois = []

    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]

    # ======================
    # COLOR
    # ======================
    x1 = int(w * 0.00)
    x2 = int(w * 0.38)
    y1 = int(h * 0.00)
    y2 = int(h * 0.18)

    zone = img[y1:y2, x1:x2]

    rois.append({
        "type": "COLOR",
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1
    })

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    detected_color, color_debug, mean_bgr = detect_card_color(zone)

    color_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "color": mean_bgr,
        "detected": detected_color,
        "debug": color_debug
    }

    # ======================
    # SYMBOL
    # ======================
    x1 = int(w * 0.05)
    x2 = int(w * 0.20)
    y1 = int(h * 0.20)
    y2 = int(h * 0.31)

    zone = img[y1:y2, x1:x2]

    rois.append({
        "type": "SYMBOL",
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1
    })

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    symbol_name, symbol_score = detect_symbol(zone)

    symbol_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "name": symbol_name,
        "score": float(symbol_score)
    }

    # ======================
    # BOTTOM
    # ======================
    x1 = int(w * 0.00)
    x2 = int(w * 0.55)
    y1 = int(h * 0.82)
    y2 = int(h * 1.00)

    zone = img[y1:y2, x1:x2]

    rois.append({
        "type": "BOTTOM",
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1
    })

    bottom_sig = compute_patch_signature(zone, size=(16, 16))

    # ======================
    # GLOBAL
    # ======================
    rois.append({
        "type": "GLOBAL",
        "x": 0,
        "y": 0,
        "w": w,
        "h": h
    })

    global_sig = compute_patch_signature(img, size=(16, 16))

    return (
        {
            "color": color_sig,
            "symbol": symbol_sig,
            "bottom": bottom_sig,
            "global": global_sig
        },
        rois
    )


def compute_signature_safe(img):
    if img is None or img.size == 0:
        return None, []
    return compute_signature(img)


# =====================================================
# GEOMETRY
# =====================================================

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


def crop_percent(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    xa = max(0, min(w, int(w * x1)))
    xb = max(0, min(w, int(w * x2)))
    ya = max(0, min(h, int(h * y1)))
    yb = max(0, min(h, int(h * y2)))

    if xb <= xa or yb <= ya:
        return None

    return img[ya:yb, xa:xb]


# =====================================================
# CARDS.JS HELPERS
# =====================================================

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
    if img is None or img.size == 0:
        return None

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
            return jsonify({"rects": [], "signature": None, "rois": []})

        file = request.files["image"]

        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"rects": [], "signature": None, "rois": []})

        rect = detect_main_card(img)

        if rect is None:
            return jsonify({"rects": [], "signature": None, "rois": []})

        quad = np.array(rect["quad"], dtype="float32")
        warped = warp_quad(img, quad)

        sig = None
        rois = []

        if warped is not None:
            cv2.imwrite(WARP_PATH, warped)
            sig, rois = compute_signature(warped)

        return jsonify({
            "rects": [rect],
            "signature": sig,
            "rois": rois
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({
            "rects": [],
            "signature": None,
            "rois": []
        })


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
    try:
        cards = load_cards_js()

        for c in cards:
            card_id = c.get("id")
            if not card_id:
                continue

            path = find_card_image(card_id)
            if path is None:
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

            warped = warp_quad(img, quad)
            if warped is None:
                continue

            sig, _ = compute_signature(warped)

            c["signature"] = {
                "scan": sig
            }

        save_cards_js(cards)
        return "OK"

    except Exception as e:
        print("BUILD SIGNATURES ERROR:", e)
        return "ERROR", 500


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
