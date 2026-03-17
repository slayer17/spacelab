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

def _keep_largest_component(bin_img):
    """
    Garde uniquement le plus gros composant blanc.
    Ça évite que du bruit parasite casse la reconnaissance.
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)

    if num_labels <= 1:
        return bin_img

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    out = np.zeros_like(bin_img)
    out[labels == largest_label] = 255
    return out


def _normalize_symbol_mask(img_or_mask, is_template=False):
    """
    Transforme une image de symbole en masque binaire propre,
    centré et redimensionné de manière stable.
    """
    if img_or_mask is None or img_or_mask.size == 0:
        return None

    # Si l'image est en couleur, on passe en gris
    if len(img_or_mask.shape) == 3:
        gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_mask.copy()

    # Lissage léger
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    # Binarisation
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nettoyage
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # On garde seulement le plus gros objet
    bin_img = _keep_largest_component(bin_img)

    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = bin_img[y1:y2 + 1, x1:x2 + 1]
    if crop is None or crop.size == 0:
        return None

    # On place la forme dans un canvas carré
    target = 96
    canvas = np.zeros((target, target), dtype=np.uint8)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min((target - 16) / w, (target - 16) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def _symbol_iou(a, b):
    """
    Compare deux masques binaires par recouvrement.
    """
    if a is None or b is None:
        return 0.0

    a_bool = a > 0
    b_bool = b > 0

    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


def _symbol_xor_score(a, b):
    """
    Mesure simple de différence de forme.
    Plus c'est proche de 1, mieux c'est.
    """
    if a is None or b is None:
        return 0.0

    diff = cv2.bitwise_xor(a, b)
    ratio = np.count_nonzero(diff) / float(diff.size)

    return float(max(0.0, 1.0 - ratio))


def _hu_score(a, b):
    """
    Compare les moments de Hu.
    Sert de sécurité supplémentaire sur la forme.
    """
    if a is None or b is None:
        return 0.0

    ma = cv2.moments(a)
    mb = cv2.moments(b)

    hua = cv2.HuMoments(ma).flatten()
    hub = cv2.HuMoments(mb).flatten()

    # Passage log pour stabiliser
    eps = 1e-10
    hua = -np.sign(hua) * np.log10(np.abs(hua) + eps)
    hub = -np.sign(hub) * np.log10(np.abs(hub) + eps)

    dist = np.mean(np.abs(hua - hub))
    score = 1.0 / (1.0 + dist)

    return float(score)


def detect_symbol(zone):
    """
    Détection symbole plus robuste :
    - crop un peu plus large
    - nettoyage fort
    - recentrage du symbole
    - comparaison de forme
    - retour du meilleur score + écart avec le 2e
    """
    templates = {
        "SCIENTIFIQUE": cv2.imread(os.path.join(SYMBOLS_DIR, "scientifique.png"), 0),
        "ASTRONAUTE": cv2.imread(os.path.join(SYMBOLS_DIR, "astronaute.png"), 0),
        "MECANICIEN": cv2.imread(os.path.join(SYMBOLS_DIR, "mecanicien.png"), 0),
        "MEDECIN": cv2.imread(os.path.join(SYMBOLS_DIR, "medecin.png"), 0),
    }

    if zone is None or zone.size == 0:
        return None, 0.0, 0.0

    h, w = zone.shape[:2]

    # Crop un peu plus large que la version précédente
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    y1 = int(h * 0.10)
    y2 = int(h * 0.90)

    zone = zone[y1:y2, x1:x2]

    if zone is None or zone.size == 0:
        return None, 0.0, 0.0

    scan_mask = _normalize_symbol_mask(zone, is_template=False)

    if scan_mask is None:
        return None, 0.0, 0.0

    scores = []

    for name, tpl in templates.items():
        if tpl is None or tpl.size == 0:
            continue

        tpl_mask = _normalize_symbol_mask(tpl, is_template=True)
        if tpl_mask is None:
            continue

        # 3 scores complémentaires
        iou = _symbol_iou(scan_mask, tpl_mask)
        xor_score = _symbol_xor_score(scan_mask, tpl_mask)
        hu = _hu_score(scan_mask, tpl_mask)

        # Score combiné : priorité à la forme réelle
        score = (iou * 0.50) + (xor_score * 0.30) + (hu * 0.20)

        scores.append((name, float(score)))

    if not scores:
        return None, 0.0, 0.0

    scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scores[0]

    if len(scores) >= 2:
        second_score = scores[1][1]
        gap = float(best_score - second_score)
    else:
        gap = float(best_score)

    return best_name, float(best_score), gap


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

    # COLOR
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

    # SYMBOL
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
symbol_name, symbol_score, symbol_gap = detect_symbol(zone)

# Règle de confiance :
# - si le score brut est trop faible, on annule
# - si le meilleur symbole est trop proche du 2e, on annule
# Ça évite les faux positifs.
if symbol_score < 0.50 or symbol_gap < 0.06:
    symbol_name = None

symbol_sig = {
    "mean": float(np.mean(gray)),
    "std": float(np.std(gray)),
    "name": symbol_name,
    "score": float(symbol_score),
    "gap": float(symbol_gap)
}

    # BOTTOM
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

    # GLOBAL
    rois.append({
        "type": "GLOBAL",
        "x": 0,
        "y": 0,
        "w": w,
        "h": h
    })

    global_sig = compute_patch_signature(img, size=(16, 16))

    return {
        "color": color_sig,
        "symbol": symbol_sig,
        "bottom": bottom_sig,
        "global": global_sig
    }, rois


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
