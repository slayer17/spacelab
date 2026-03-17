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


def detect_symbol(zone):
    base = os.path.dirname(__file__)

    templates = {
        "SCIENTIFIQUE": cv2.imread(os.path.join(base, "symbols", "scientifique.png"), 0),
        "ASTRONAUTE": cv2.imread(os.path.join(base, "symbols", "astronaute.png"), 0),
        "MECANICIEN": cv2.imread(os.path.join(base, "symbols", "mecanicien.png"), 0),
        "MEDECIN": cv2.imread(os.path.join(base, "symbols", "medecin.png"), 0),
    }

    if zone is None or zone.size == 0:
        return None, 0.0

    # =========================
    # 1) Crop centre utile
    # =========================
    h, w = zone.shape[:2]

    x1 = int(w * 0.12)
    x2 = int(w * 0.88)
    y1 = int(h * 0.12)
    y2 = int(h * 0.88)

    zone = zone[y1:y2, x1:x2]

    if zone.size == 0:
        return None, 0.0

    # =========================
    # 2) Préparation scan
    # =========================
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)

    best_name = None
    best_score = -1.0

    # =========================
    # 3) Compare templates propres
    # =========================
    for name, tpl in templates.items():

        if tpl is None:
            continue

        th, tw = tpl.shape[:2]

        tx1 = int(tw * 0.12)
        tx2 = int(tw * 0.88)
        ty1 = int(th * 0.12)
        ty2 = int(th * 0.88)

        tpl = tpl[ty1:ty2, tx1:tx2]

        tpl = cv2.GaussianBlur(tpl, (3, 3), 0)
        tpl = cv2.equalizeHist(tpl)
        tpl = cv2.resize(tpl, (64, 64), interpolation=cv2.INTER_CUBIC)

        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())

        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score
# =====================================================
# UTILS
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

    color_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "color": zone.mean(axis=(0, 1)).tolist()
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

    name, score = detect_symbol(zone)

    symbol_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "name": name,
        "score":score
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

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    bottom_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }

    # ======================
    # GLOBAL
    # ======================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rois.append({
        "type": "GLOBAL",
        "x": 0,
        "y": 0,
        "w": w,
        "h": h
    })

    global_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }

    return (
        {
            "color": color_sig,
            "symbol": symbol_sig,
            "bottom": bottom_sig,
            "global": global_sig
        },
        rois
    )
    def detect_card_color(zone):
    if zone is None or zone.size == 0:
        return "ROUGE", {"reason": "empty"}

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # On garde surtout les pixels colorés
    mask = (s > 60) & (v > 50)

    hues = h[mask]

    if len(hues) == 0:
        return "ROUGE", {"reason": "no_saturated_pixels"}

    counts = {
        "ROUGE": int(np.sum((hues <= 10) | (hues >= 170))),
        "JAUNE": int(np.sum((hues >= 15) & (hues <= 35))),
        "VERT":  int(np.sum((hues >= 40) & (hues <= 85))),
        "BLEU":  int(np.sum((hues >= 90) & (hues <= 130))),
    }

    detected = max(counts, key=counts.get)

    return detected, counts
    
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
        return None, []

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
            return jsonify({"rects": [], "signature": None})

        file = request.files["image"]

        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        rect = detect_main_card(img)

        if rect is None:
            return jsonify({"rects": [], "signature": None})

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
            "rois": rois
        })

    except Exception as e:

        print(e)

        return jsonify({
            "rects": [],
            "signature": None
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

    with open("cards.js", "r", encoding="utf-8") as f:
        txt = f.read()

    start = txt.find("[")
    end = txt.rfind("]") + 1

    json_txt = txt[start:end]

    cards = json.loads(json_txt)

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

        sig, _ = compute_signature(warp)

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
