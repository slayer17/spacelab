import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, "cards")
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols")
DIGITS_DIR = os.path.join(BASE_DIR, "digits")
CARDS_JS_PATH = os.path.join(BASE_DIR, "cards.js")
WARP_PATH = os.path.join(BASE_DIR, "warp.jpg")

# =====================================================
# UTILS GÉOMÉTRIE & IMAGE
# =====================================================

def order_points(pts):
    """ Ordonne les points : haut-gauche, haut-droit, bas-droit, bas-gauche. """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_quad(img, pts):
    """ Redresse une zone quadrilatère en un rectangle parfait. """
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
    return cv2.warpPerspective(img, M, (maxW, maxH))

def _clip_box(x, y, w, h, max_w, max_h):
    """ Empêche une bounding box de sortir de l'image. """
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h

# =====================================================
# DÉTECTION DES SYMBOLES (Métier)
# =====================================================

def _keep_largest_component(bin_img):
    if bin_img is None or bin_img.size == 0:
        return bin_img
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if num_labels <= 1:
        return bin_img
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    out = np.zeros_like(bin_img)
    out[labels == largest_label] = 255
    return out

def _normalize_symbol_mask(img_or_mask):
    if img_or_mask is None or img_or_mask.size == 0:
        return None
    gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY) if len(img_or_mask.shape) == 3 else img_or_mask.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = _keep_largest_component(bin_img)
    
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0: return None
    crop = bin_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    
    target = 96
    canvas = np.zeros((target, target), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = min((target - 16) / w, (target - 16) / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas[(target-nh)//2:(target-nh)//2+nh, (target-nw)//2:(target-nw)//2+nw] = resized
    return canvas

def detect_symbol(zone):
    templates = {
        "SCIENTIFIQUE": cv2.imread(os.path.join(SYMBOLS_DIR, "scientifique.png"), 0),
        "ASTRONAUTE": cv2.imread(os.path.join(SYMBOLS_DIR, "astronaute.png"), 0),
        "MECANICIEN": cv2.imread(os.path.join(SYMBOLS_DIR, "mecanicien.png"), 0),
        "MEDECIN": cv2.imread(os.path.join(SYMBOLS_DIR, "medecin.png"), 0),
    }
    if zone is None or zone.size == 0: return None, 0.0, 0.0
    scan_mask = _normalize_symbol_mask(zone)
    if scan_mask is None: return None, 0.0, 0.0
    
    scores = []
    for name, tpl in templates.items():
        if tpl is None: continue
        tpl_mask = _normalize_symbol_mask(tpl)
        # Comparaison par IoU
        inter = np.logical_and(scan_mask > 0, tpl_mask > 0).sum()
        union = np.logical_or(scan_mask > 0, tpl_mask > 0).sum()
        score = inter / union if union > 0 else 0
        scores.append((name, float(score)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    gap = scores[0][1] - scores[1][1] if len(scores) > 1 else scores[0][1]
    return scores[0][0], scores[0][1], gap

# =====================================================
# DÉTECTION DU CHIFFRE (Points)
# =====================================================

def _normalize_digit_mask(img_or_mask):
    if img_or_mask is None or img_or_mask.size == 0: return None
    gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY) if len(img_or_mask.shape) == 3 else img_or_mask.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    crop = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    target = 96
    canvas = np.zeros((target, target), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = min((target - 12) / w, (target - 12) / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas[(target-nh)//2:(target-nh)//2+nh, (target-nw)//2:(target-nw)//2+nw] = resized
    return canvas

def _digit_score(scan_badge, tpl_badge):
    """ Template Matching pour tolérer le décalage. """
    if scan_badge is None or tpl_badge is None: return 0.0
    m = 10 
    tpl_crop = tpl_badge[m:-m, m:-m]
    res = cv2.matchTemplate(scan_badge, tpl_crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max(0.0, max_val))

def detect_digit(zone):
    if zone is None or zone.size == 0: return None, 0.0, 0.0
    scan_badge = _normalize_digit_mask(zone)
    if scan_badge is None: return None, 0.0, 0.0
    scores = []
    for n in range(1, 11):
        tpl = cv2.imread(os.path.join(DIGITS_DIR, f"{n}.png"))
        if tpl is None: continue
        tpl_badge = _normalize_digit_mask(tpl)
        scores.append((n, _digit_score(scan_badge, tpl_badge)))
    scores.sort(key=lambda x: x[1], reverse=True)
    gap = scores[0][1] - scores[1][1] if len(scores) > 1 else scores[0][1]
    return int(scores[0][0]), scores[0][1], gap

# =====================================================
# LOCALISATION DU BADGE DE POINTS
# =====================================================

def find_points_badge(bottom_zone):
    if bottom_zone is None or bottom_zone.size == 0: return None, None
    h, w = bottom_zone.shape[:2]
    search_zone = bottom_zone[:, :int(w * 0.58)]
    gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        ratio = bw / float(bh) if bh > 0 else 0
        area_ratio = cv2.contourArea(c) / float(search_zone.size)
        if 0.01 < area_ratio < 0.30 and 0.5 < ratio < 1.5:
            score = (1.0 - x/w) * 2 + (1.0 - abs(ratio-1.0))
            candidates.append((score, x, y, bw, bh))
    
    if not candidates: return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return search_zone[y:y+bh, x:x+bw], (x, y, bw, bh)

# =====================================================
# SIGNATURE & COULEUR
# =====================================================

def detect_card_color(zone):
    if zone is None or zone.size == 0: return "ROUGE", {}, [0,0,0]
    mean_bgr = zone.mean(axis=(0,1)).tolist()
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
    hues = hsv[:,:,0][mask]
    if len(hues) == 0: return "ROUGE", {"reason": "no_sat"}, mean_bgr
    counts = {
        "ROUGE": int(np.sum((hues <= 10) | (hues >= 170))),
        "JAUNE": int(np.sum((hues >= 15) & (hues <= 35))),
        "VERT": int(np.sum((hues >= 40) & (hues <= 85))),
        "BLEU": int(np.sum((hues >= 90) & (hues <= 130))),
    }
    return max(counts, key=counts.get), counts, mean_bgr

def compute_patch_signature(zone, size=(16, 16)):
    if zone is None or zone.size == 0: return {"mean":0, "std":0, "vector":[]}
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return {"mean": float(np.mean(gray)), "std": float(np.std(gray)), "vector": small.flatten().tolist()}

def compute_signature(img):
    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]
    rois = []

    # Couleur
    czone = img[0:int(h*0.18), 0:int(w*0.38)]
    det_c, _, m_bgr = detect_card_color(czone)
    color_sig = {"detected": det_c, "color": m_bgr}
    rois.append({"type":"COLOR", "x":0, "y":0, "w":int(w*0.38), "h":int(h*0.18)})

    # Symbole
    szone = img[int(h*0.20):int(h*0.31), int(w*0.05):int(w*0.20)]
    s_name, s_score, s_gap = detect_symbol(szone)
    symbol_sig = {"name": s_name if s_score > 0.45 else None, "score": s_score}
    rois.append({"type":"SYMBOL", "x":int(w*0.05), "y":int(h*0.20), "w":int(w*0.15), "h":int(h*0.11)})

    # Points
    bzone = img[int(h*0.82):h, 0:int(w*0.55)]
    badge_crop, badge_box = find_points_badge(bzone)
    points_sig = {"digit": None, "score": 0.0}
    if badge_crop is not None:
        bx, by, bw2, bh2 = badge_box
        d_val, d_score, d_gap = detect_digit(badge_crop)
        if d_score > 0.40: points_sig = {"digit": d_val, "score": d_score}
        rois.append({"type":"POINTS_BADGE", "x":bx, "y":int(h*0.82)+by, "w":bw2, "h":bh2})

    return {
        "color": color_sig,
        "symbol": symbol_sig,
        "points": points_sig,
        "global": compute_patch_signature(img)
    }, rois

# =====================================================
# DÉTECTION CARTE (MAIN)
# =====================================================

def detect_main_card(img):
    if img is None: return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return {"quad": approx.reshape(4, 2).astype(int).tolist()}
    return None

# =====================================================
# ROUTES FLASK
# =====================================================

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file: return jsonify({"rects": []})
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    rect = detect_main_card(img)
    if not rect: return jsonify({"rects": []})
    
    warped = warp_quad(img, np.array(rect["quad"], dtype="float32"))
    if warped is not None:
        cv2.imwrite(WARP_PATH, warped)
        sig, rois = compute_signature(warped)
        return jsonify({"rects": [rect], "signature": sig, "rois": rois})
    return jsonify({"rects": [rect]})

@app.route("/build_signatures")
def build_signatures():
    try:
        with open(CARDS_JS_PATH, "r", encoding="utf-8") as f:
            data = f.read().replace("window.CARDS =", "").strip().rstrip(";")
            cards = json.loads(data)
        for c in cards:
            for ext in [".jpg", ".png", ".jpeg"]:
                p = os.path.join(CARDS_DIR, c["id"] + ext)
                if os.path.exists(p):
                    img = cv2.imread(p)
                    sig, _ = compute_signature(img)
                    c["signature"] = {"scan": sig}
                    break
        with open(CARDS_JS_PATH, "w", encoding="utf-8") as f:
            f.write("window.CARDS = " + json.dumps(cards, indent=2))
        return "OK"
    except Exception as e:
        return str(e), 500

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
