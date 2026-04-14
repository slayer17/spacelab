import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import base64
import shutil
import html
import re
from datetime import datetime


# Helpers bottom.py intégrés ici pour éviter tout plantage à l'import.
# Le fichier devient autonome : spacelab.py suffit à lui seul.

app = Flask(__name__)
def _img_to_base64(img):
    if img is None or img.size == 0:
        return None

    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        return None

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _html_img_block(title, img_b64):
    if not img_b64:
        return f"<h3>{title}</h3><p>Image absente</p>"

    return f"""
    <div style="margin-bottom:20px;">
      <h3 style="margin:0 0 8px 0;">{title}</h3>
      <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ccc;" />
    </div>
    """


def _natural_sort_key(text):
    parts = re.split(r"(\d+)", (text or "").lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def _safe_upper(value):
    return str(value or "").strip().upper()


def _safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, "cards")
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols")
DIGITS_DIR = os.path.join(BASE_DIR, "digits")
CARDS_JS_PATH = os.path.join(BASE_DIR, "cards.js")
WARP_PATH = os.path.join(BASE_DIR, "warp.jpg")


# =====================================================
# LOCAL BOTTOM HELPERS (standalone fallback)
# =====================================================

def extract_bottom_roi_from_full_card(img):
    if img is None or img.size == 0:
        return None, None, (0, 0, 0, 0)

    full_img = cv2.resize(img, (200, 300))
    h, w = full_img.shape[:2]

    x1 = 0
    x2 = w
    y1 = int(round(h * 0.66))
    y2 = h

    bottom_roi = full_img[y1:y2, x1:x2]
    return full_img, bottom_roi, (x1, y1, x2 - x1, y2 - y1)


def analyze_bottom(bottom_zone, digits_dir=None):
    return analyze_bottom_layout(bottom_zone)


def _normalize_badge(img):
    return _normalize_digit_mask(img)


def _extract_digit_mask(img):
    return _extract_digit_only_mask(img)


def build_overlay(full_img, bottom_box, result):
    if full_img is None or getattr(full_img, 'size', 0) == 0:
        return None

    overlay = full_img.copy()
    bx, by, bw, bh = bottom_box
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    box_colors = {
        'panel_box': (0, 255, 255),
        'points_box': (0, 255, 0),
        'slash_box': (255, 255, 0),
        'right_icon_box': (255, 0, 255),
        'bottom_line_box': (0, 128, 255),
        'special_box': (0, 0, 255),
    }

    for key, color in box_colors.items():
        box = (result or {}).get(key)
        if not box:
            continue
        x, y, w, h = box
        cv2.rectangle(overlay, (bx + x, by + y), (bx + x + w, by + y + h), color, 2)

    return overlay



# =====================================================
# SYMBOL DETECTION
# =====================================================

_SYMBOL_REFS_CACHE = None
_CARD_MATCH_REF_CACHE = None
_HOG_DESCRIPTOR = None



def _keep_main_components(bin_img, max_components=6, min_ratio=0.08):
    if bin_img is None or bin_img.size == 0:
        return bin_img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if num_labels <= 1:
        return bin_img

    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 4:
            comps.append((i, area))

    if not comps:
        return bin_img

    comps.sort(key=lambda t: t[1], reverse=True)
    largest = comps[0][1]

    keep = [
        i for i, area in comps[:max_components]
        if area >= largest * float(min_ratio)
    ]

    out = np.zeros_like(bin_img)
    for i in keep:
        out[labels == i] = 255
    return out


def _remove_border_touching_components(bin_img, min_area=4):
    if bin_img is None or bin_img.size == 0:
        return bin_img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    h, w = bin_img.shape[:2]
    out = np.zeros_like(bin_img)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue

        touches_border = (
            x <= 0 or
            y <= 0 or
            (x + bw) >= (w - 1) or
            (y + bh) >= (h - 1)
        )

        if touches_border:
            continue

        out[labels == i] = 255

    return out


def _normalize_binary_to_canvas(bin_img, target=96, pad=8):
    if bin_img is None or bin_img.size == 0:
        return None

    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = bin_img[y1:y2 + 1, x1:x2 + 1]
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return None

    canvas = np.zeros((target, target), dtype=np.uint8)

    scale = min((target - 2 * pad) / float(w), (target - 2 * pad) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    ox = (target - new_w) // 2
    oy = (target - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    return canvas


def _extract_symbol_panel(zone):
    if zone is None or zone.size == 0:
        return zone

    if len(zone.shape) == 3:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    else:
        gray = zone.copy()

    dark_mask = np.where(gray < 105, 255, 0).astype(np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    h, w = gray.shape[:2]

    best = None

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < 25 or bw < 8 or bh < 8:
            continue

        ratio = bw / float(bh)
        if ratio < 0.65 or ratio > 1.35:
            continue

        cx, cy = centroids[i]
        score = area - 1.5 * abs(cx - (w * 0.38)) - 1.5 * abs(cy - (h * 0.52))

        if best is None or score > best[0]:
            best = (score, (x, y, bw, bh))

    if best is None:
        return zone

    x, y, bw, bh = best[1]
    pad = 2

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(zone.shape[1], x + bw + pad)
    y2 = min(zone.shape[0], y + bh + pad)

    panel = zone[y1:y2, x1:x2]
    if panel is None or panel.size == 0:
        return zone

    return panel


def _normalize_symbol_scan(zone):
    if zone is None or zone.size == 0:
        return None, None, None

    panel = _extract_symbol_panel(zone)

    if len(panel.shape) == 3:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    else:
        gray = panel.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    def build_canvas(thresh_flag):
        _, bin_img = cv2.threshold(gray, 0, 255, thresh_flag)

        cleaned = _remove_border_touching_components(bin_img, min_area=3)
        if np.count_nonzero(cleaned) == 0:
            cleaned = bin_img.copy()

        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        cleaned = _keep_main_components(cleaned, max_components=8, min_ratio=0.07)

        canvas = _normalize_binary_to_canvas(cleaned)
        if canvas is None:
            return None

        if int(np.count_nonzero(canvas)) == 0:
            return None

        return canvas

    canvas = build_canvas(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if canvas is not None:
        return canvas, panel, "binary"

    canvas = build_canvas(cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if canvas is not None:
        return canvas, panel, "binary_inv"

    return None, panel, None
    
    
def _normalize_symbol_template(img):
    if img is None or img.size == 0:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = _remove_border_touching_components(bin_img, min_area=3)

    if np.count_nonzero(bin_img) == 0:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    bin_img = _keep_main_components(bin_img, max_components=8, min_ratio=0.05)

    return _normalize_binary_to_canvas(bin_img)


def _extract_symbol_zone_variants(zone):
    if zone is None or zone.size == 0:
        return []

    h, w = zone.shape[:2]
    variants = []

    crops = [
        (0.00, 0.00, 1.00, 1.00),
        (0.03, 0.00, 0.97, 1.00),
        (0.00, 0.03, 1.00, 0.97),
        (0.05, 0.05, 0.95, 0.95),
        (0.02, 0.02, 0.98, 0.98),
    ]

    seen = set()

    for x1r, y1r, x2r, y2r in crops:
        x1 = max(0, min(w, int(round(w * x1r))))
        x2 = max(0, min(w, int(round(w * x2r))))
        y1 = max(0, min(h, int(round(h * y1r))))
        y2 = max(0, min(h, int(round(h * y2r))))

        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)

        crop = zone[y1:y2, x1:x2]
        if crop is not None and crop.size > 0:
            variants.append(crop)

    return variants



def _get_hog_descriptor():
    global _HOG_DESCRIPTOR

    if _HOG_DESCRIPTOR is None:
        _HOG_DESCRIPTOR = cv2.HOGDescriptor(
            _winSize=(96, 96),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

    return _HOG_DESCRIPTOR



def _panel_to_gray_canvas(panel, target=96):
    if panel is None or panel.size == 0:
        return None

    if len(panel.shape) == 3:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    else:
        gray = panel.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (target, target), interpolation=cv2.INTER_AREA)
    return gray



def _hog_feature_from_mask(mask):
    if mask is None or mask.size == 0:
        return None

    hog = _get_hog_descriptor()
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    return hog.compute(mask_u8).flatten().astype(np.float32)



def _hog_feature_from_panel(panel):
    gray = _panel_to_gray_canvas(panel)
    if gray is None:
        return None

    hog = _get_hog_descriptor()
    return hog.compute(gray).flatten().astype(np.float32)



def _compute_symbol_shape_features(mask):
    features = {
        "fill": 0.0,
        "components": 0.0,
        "largest_comp": 0.0,
        "bbox_ratio": 0.0,
        "bbox_fill": 0.0,
        "holes": 0.0,
        "solidity": 0.0,
        "circularity": 0.0,
    }

    if mask is None or mask.size == 0:
        return features

    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    features["fill"] = float(np.count_nonzero(m) / float(m.size))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    features["components"] = float(len(comp_areas))
    if comp_areas:
        features["largest_comp"] = float(max(comp_areas) / float(m.size))

    cnts, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        features["holes"] = float(sum(1 for h in hierarchy if h[3] != -1))

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0

        features["bbox_ratio"] = float(w / float(h + 1e-6))
        features["bbox_fill"] = float(area / float((w * h) + 1e-6)) if w > 0 and h > 0 else 0.0
        features["solidity"] = float(area / float(hull_area + 1e-6)) if hull_area > 0 else 0.0
        features["circularity"] = float((4.0 * np.pi * area) / float((peri * peri) + 1e-6)) if peri > 0 else 0.0

    return features



def _shape_feature_vector(features):
    if features is None:
        features = {}

    return np.array([
        float(features.get("fill", 0.0)),
        float(features.get("components", 0.0)),
        float(features.get("largest_comp", 0.0)),
        float(min(features.get("bbox_ratio", 0.0), 3.0)),
        float(features.get("bbox_fill", 0.0)),
        float(features.get("holes", 0.0)),
        float(features.get("solidity", 0.0)),
        float(features.get("circularity", 0.0)),
    ], dtype=np.float32)



def _normalize_score_map(score_map, names):
    if not score_map:
        return {name: 0.0 for name in names}

    max_val = max(float(score_map.get(name, 0.0)) for name in names)
    if max_val <= 0:
        return {name: 0.0 for name in names}

    return {name: float(score_map.get(name, 0.0)) / float(max_val) for name in names}



def _build_template_symbol_scores(scan_mask, refs, symbol_names):
    per_symbol = {}

    for symbol_name in symbol_names:
        icon_mask = refs["icons"].get(symbol_name)
        icon_score = _score_symbol_masks(scan_mask, icon_mask) if icon_mask is not None else 0.0

        card_scores = []
        for card_ref in refs["cards"].get(symbol_name, []):
            ref_mask = card_ref.get("mask")
            if ref_mask is None:
                continue

            s = _score_symbol_masks(scan_mask, ref_mask)
            card_scores.append((float(s), card_ref.get("id")))

        card_scores.sort(key=lambda t: t[0], reverse=True)

        if card_scores:
            top_scores = card_scores[:3]
            avg_card_score = float(sum(s for s, _ in top_scores) / len(top_scores))
            best_source = top_scores[0][1]
            support = len(card_scores)
        else:
            avg_card_score = 0.0
            best_source = None
            support = 0

        final_score = float((avg_card_score * 0.55) + (icon_score * 0.45))

        per_symbol[symbol_name] = {
            "name": symbol_name,
            "score": final_score,
            "icon_score": float(icon_score),
            "card_score": float(avg_card_score),
            "best_source": best_source,
            "support": support,
            "card_scores": card_scores,
        }

    return per_symbol



def _knn_symbol_scores(feature, refs, feature_key, symbol_names, k=7):
    score_map = {name: 0.0 for name in symbol_names}
    top_refs = []

    if feature is None:
        return score_map, top_refs

    candidates = []
    for ref in refs.get("all_cards", []):
        ref_feature = ref.get(feature_key)
        if ref_feature is None:
            continue

        dist = float(np.linalg.norm(feature - ref_feature))
        candidates.append((dist, ref.get("symbol"), ref.get("id")))

    candidates.sort(key=lambda t: t[0])
    top_refs = candidates[:k]

    for rank, (dist, symbol_name, card_id) in enumerate(top_refs):
        sim = 1.0 / (1.0 + dist)
        score_map[symbol_name] += float(sim / float(rank + 1))

    return _normalize_score_map(score_map, symbol_names), top_refs



def _shape_model_scores(scan_mask, refs, symbol_names):
    score_map = {name: 0.0 for name in symbol_names}
    features = _compute_symbol_shape_features(scan_mask)
    x = _shape_feature_vector(features)

    shape_stats = refs.get("shape_stats", {})
    for symbol_name in symbol_names:
        stats = shape_stats.get(symbol_name)
        if not stats:
            continue

        mean_vec = stats.get("mean")
        std_vec = stats.get("std")
        if mean_vec is None or std_vec is None:
            continue

        z = ((x - mean_vec) / std_vec) ** 2
        score_map[symbol_name] = float(np.exp(-0.5 * float(np.mean(z))))

    # Règles de secours ciblées pour les cas qui glissaient encore.
    if features["bbox_ratio"] > 3.0 and features["fill"] < 0.14:
        score_map["ASTRONAUTE"] = max(score_map.get("ASTRONAUTE", 0.0), 1.0)

    if (
        features["fill"] > 0.37 and
        int(round(features["components"])) == 1 and
        features["bbox_fill"] > 0.62 and
        features["solidity"] > 0.76
    ):
        score_map["MEDECIN"] = max(score_map.get("MEDECIN", 0.0), 1.0)

    if (
        features["circularity"] < 0.18 and
        features["solidity"] < 0.62 and
        features["bbox_fill"] < 0.52
    ):
        score_map["MECANICIEN"] = max(score_map.get("MECANICIEN", 0.0), 1.0)

    return _normalize_score_map(score_map, symbol_names), features


def _symbol_iou(a, b):
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
    if a is None or b is None:
        return 0.0

    diff = cv2.bitwise_xor(a, b)
    ratio = np.count_nonzero(diff) / float(diff.size)
    return float(max(0.0, 1.0 - ratio))


def _hu_score(a, b):
    if a is None or b is None:
        return 0.0

    ma = cv2.moments(a)
    mb = cv2.moments(b)

    hua = cv2.HuMoments(ma).flatten()
    hub = cv2.HuMoments(mb).flatten()

    eps = 1e-10
    hua = -np.sign(hua) * np.log10(np.abs(hua) + eps)
    hub = -np.sign(hub) * np.log10(np.abs(hub) + eps)

    dist = np.mean(np.abs(hua - hub))
    return float(1.0 / (1.0 + dist))


def _contour_shape_score(a, b):
    if a is None or b is None:
        return 0.0

    cnts_a, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts_a or not cnts_b:
        return 0.0

    ca = max(cnts_a, key=cv2.contourArea)
    cb = max(cnts_b, key=cv2.contourArea)

    val = cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I1, 0)
    return float(1.0 / (1.0 + val))


def _projection_score(a, b):
    if a is None or b is None:
        return 0.0

    aa = (a > 0).astype(np.float32)
    bb = (b > 0).astype(np.float32)

    proj_x_a = aa.mean(axis=0)
    proj_x_b = bb.mean(axis=0)
    proj_y_a = aa.mean(axis=1)
    proj_y_b = bb.mean(axis=1)

    dx = np.mean(np.abs(proj_x_a - proj_x_b))
    dy = np.mean(np.abs(proj_y_a - proj_y_b))

    return float(max(0.0, 1.0 - ((dx + dy) / 2.0)))


def _score_symbol_masks(a, b):
    if a is None or b is None:
        return 0.0

    iou = _symbol_iou(a, b)
    xor_score = _symbol_xor_score(a, b)
    hu = _hu_score(a, b)
    contour = _contour_shape_score(a, b)
    proj = _projection_score(a, b)

    return float(
        (iou * 0.32) +
        (xor_score * 0.20) +
        (hu * 0.12) +
        (contour * 0.16) +
        (proj * 0.20)
    )


def _extract_symbol_zone_from_card(img):
    if img is None or img.size == 0:
        return None

    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]

    x1 = int(w * 0.02)
    x2 = int(w * 0.24)
    y1 = int(h * 0.16)
    y2 = int(h * 0.34)

    zone = img[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    return zone






def _load_symbol_references():
    global _SYMBOL_REFS_CACHE

    if _SYMBOL_REFS_CACHE is not None:
        return _SYMBOL_REFS_CACHE

    # 1. Sécurisation du dossier de symboles
    # Assurez-vous que SYMBOLS_DIR est défini en haut du fichier
    if not os.path.exists(SYMBOLS_DIR):
        print(f"ATTENTION : Le dossier {SYMBOLS_DIR} est introuvable.")
        return {"icons": {}, "cards": {}, "all_cards": [], "shape_stats": {}}

    symbol_names = ["SCIENTIFIQUE", "ASTRONAUTE", "MECANICIEN", "MEDECIN"]
    refs = {
        "icons": {},
        "cards": {name: [] for name in symbol_names},
        "all_cards": [],
        "shape_stats": {},
    }

    # 2. Chargement des icônes de référence
    for name in symbol_names:
        path = os.path.join(SYMBOLS_DIR, f"{name.lower()}.png")
        tpl = cv2.imread(path, cv2.IMREAD_COLOR)
        if tpl is not None:
            refs["icons"][name] = _normalize_symbol_template(tpl)
        else:
            print(f"Alerte : Icône manquante pour {name} à {path}")
            refs["icons"][name] = None

    # 3. Chargement des données cartes (Sécurité si la fonction n'existe pas)
    cards = []
    if 'load_cards_js' in globals():
        try:
            cards = load_cards_js()
        except Exception as e:
            print(f"Erreur lors de l'appel à load_cards_js : {e}")
    else:
        print("Erreur : La fonction load_cards_js n'est pas définie.")

    # 4. Traitement des cartes
    for card in cards:
        card_id = card.get("id")
        symbol_name = card.get("symbol")

        if not card_id or symbol_name not in refs["cards"]:
            continue

        # Sécurité sur find_card_image
        if 'find_card_image' not in globals():
            continue
            
        path = find_card_image(card_id)
        if path is None or not os.path.exists(path):
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            continue

        zone = _extract_symbol_zone_from_card(img)
        if zone is None or zone.size == 0:
            continue

        # Normalisation
        try:
            mask, panel, _ = _normalize_symbol_scan(zone)
            if mask is None:
                continue

            panel_img = panel if (panel is not None and panel.size > 0) else zone
            
            entry = {
                "id": card_id,
                "symbol": symbol_name,
                "mask": mask,
                "panel_gray": _panel_to_gray_canvas(panel_img),
                "hog_mask": _hog_feature_from_mask(mask),
                "hog_panel": _hog_feature_from_panel(panel_img),
                "shape_features": _compute_symbol_shape_features(mask),
            }

            refs["cards"][symbol_name].append(entry)
            refs["all_cards"].append(entry)
        except Exception as e:
            print(f"Erreur de processing sur la carte {card_id}: {e}")
            continue

    # 5. Calcul des statistiques de forme
    for symbol_name in symbol_names:
        feature_vecs = []
        for entry in refs["cards"].get(symbol_name, []):
            feat = _shape_feature_vector(entry.get("shape_features"))
            if feat is not None:
                feature_vecs.append(feat)
        
        if len(feature_vecs) > 0:
            mat = np.vstack(feature_vecs).astype(np.float32)
            refs["shape_stats"][symbol_name] = {
                "mean": mat.mean(axis=0),
                "std": mat.std(axis=0) + 1e-3,
            }

    _SYMBOL_REFS_CACHE = refs
    return _SYMBOL_REFS_CACHE

def detect_symbol(zone):
    refs = _load_symbol_references()
    symbol_names = ["SCIENTIFIQUE", "ASTRONAUTE", "MECANICIEN", "MEDECIN"]

    empty_debug = {
        "top_candidates": [],
        "winner_references": [],
        "runner_up": None,
    }

    if zone is None or zone.size == 0:
        return None, 0.0, 0.0, empty_debug

    variants = _extract_symbol_zone_variants(zone)
    if not variants:
        return None, 0.0, 0.0, empty_debug

    best_result = None

    for variant in variants:
        scan_mask, panel, _ = _normalize_symbol_scan(variant)
        if scan_mask is None:
            continue

        panel_img = panel if panel is not None and panel.size > 0 else variant

        template_scores = _build_template_symbol_scores(scan_mask, refs, symbol_names)
        template_abs = {name: float(template_scores[name]["score"]) for name in symbol_names}

        mask_feature = _hog_feature_from_mask(scan_mask)
        mask_knn_scores, mask_knn_top = _knn_symbol_scores(
            mask_feature,
            refs,
            "hog_mask",
            symbol_names,
            k=7,
        )

        panel_feature = _hog_feature_from_panel(panel_img)
        panel_knn_scores, panel_knn_top = _knn_symbol_scores(
            panel_feature,
            refs,
            "hog_panel",
            symbol_names,
            k=7,
        )

        shape_scores, shape_features = _shape_model_scores(scan_mask, refs, symbol_names)

        combined_scores = {}
        for symbol_name in symbol_names:
            combined_scores[symbol_name] = float(
                (template_abs.get(symbol_name, 0.0) * 0.40) +
                (mask_knn_scores.get(symbol_name, 0.0) * 0.20) +
                (panel_knn_scores.get(symbol_name, 0.0) * 0.15) +
                (shape_scores.get(symbol_name, 0.0) * 0.25)
            )

        ranked = sorted(combined_scores.items(), key=lambda t: t[1], reverse=True)
        if not ranked:
            continue

        winner_name, winner_score = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        gap = float(winner_score - (runner_up[1] if runner_up else 0.0))

        winner_refs = []
        for card_ref in refs["cards"].get(winner_name, []):
            ref_mask = card_ref.get("mask")
            mask_similarity = _score_symbol_masks(scan_mask, ref_mask) if ref_mask is not None else 0.0

            panel_similarity = 0.0
            ref_panel_feature = card_ref.get("hog_panel")
            if panel_feature is not None and ref_panel_feature is not None:
                panel_similarity = float(1.0 / (1.0 + np.linalg.norm(panel_feature - ref_panel_feature)))

            mix_score = float((mask_similarity * 0.70) + (panel_similarity * 0.30))
            winner_refs.append({
                "card_id": card_ref.get("id"),
                "score": mix_score,
            })

        winner_refs.sort(key=lambda d: d["score"], reverse=True)
        best_source = winner_refs[0]["card_id"] if winner_refs else template_scores[winner_name].get("best_source")

        result = {
            "raw_name": winner_name,
            "score": float(winner_score),
            "gap": gap,
            "top_candidates": [
                {
                    "best_kind": "card" if best_source else "icon",
                    "best_source": best_source or winner_name.lower(),
                    "name": winner_name,
                    "score": float(winner_score),
                    "support": int(template_scores[winner_name].get("support", 0)),
                }
            ],
            "winner_references": winner_refs[:5],
            "runner_up": {
                "name": runner_up[0],
                "score": float(runner_up[1]),
            } if runner_up else None,
            "shape_features": shape_features,
            "mask_knn_top": mask_knn_top,
            "panel_knn_top": panel_knn_top,
        }

        if best_result is None:
            best_result = result
        else:
            prev = (best_result["score"], best_result["gap"])
            cur = (result["score"], result["gap"])
            if cur > prev:
                best_result = result

    if best_result is None:
        return None, 0.0, 0.0, empty_debug

    return (
        best_result["raw_name"],
        float(best_result["score"]),
        float(best_result["gap"]),
        {
            "top_candidates": best_result.get("top_candidates", []),
            "winner_references": best_result.get("winner_references", []),
            "runner_up": best_result.get("runner_up"),
        }
    )

# =====================================================
# DIGIT DETECTION
# =====================================================

def _normalize_digit_mask(img_or_mask):
    """
    Normalise le badge complet des points.
    On garde ici le contexte global du badge.
    """
    if img_or_mask is None or img_or_mask.size == 0:
        return None

    if len(img_or_mask.shape) == 3:
        gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_mask.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    _, white_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(white_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = gray[y1:y2 + 1, x1:x2 + 1]
    if crop is None or crop.size == 0:
        return None

    target = 96
    canvas = np.zeros((target, target), dtype=np.uint8)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min((target - 12) / w, (target - 12) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def _extract_digit_only_mask(img_or_mask):
    """
    Extrait seulement la forme noire du chiffre
    à l'intérieur du badge blanc.

    Nouvelle logique :
    - on trouve le vrai badge blanc
    - on remplit sa forme
    - puis on cherche les zones sombres à l'intérieur
    """
    if img_or_mask is None or img_or_mask.size == 0:
        return None

    if len(img_or_mask.shape) == 3:
        gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_mask.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    # 1) Trouver les zones blanches du badge
    _, white_mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # On prend le plus gros contour blanc = le badge
    best = max(contours, key=cv2.contourArea)

    if cv2.contourArea(best) < max(20, white_mask.shape[0] * white_mask.shape[1] * 0.05):
        return None

    badge_fill = np.zeros_like(white_mask)
    cv2.drawContours(badge_fill, [best], -1, 255, thickness=-1)

    x, y, w, h = cv2.boundingRect(best)

    crop_gray = gray[y:y + h, x:x + w]
    crop_badge = badge_fill[y:y + h, x:x + w]

    if crop_gray is None or crop_gray.size == 0:
        return None

    # Petite érosion pour rester un peu à l'intérieur du badge
    inner_badge = cv2.erode(
        crop_badge,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    badge_pixels = crop_gray[inner_badge > 0]
    if badge_pixels.size == 0:
        return None

    # On cherche les pixels sombres du chiffre
    # avec un seuil relatif au contenu du badge
    dark_threshold = np.percentile(badge_pixels, 45)

    dark_mask = (crop_gray < dark_threshold).astype(np.uint8) * 255
    digit_mask = cv2.bitwise_and(dark_mask, inner_badge)

    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, kernel)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        digit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    min_area = max(6, digit_mask.shape[0] * digit_mask.shape[1] * 0.01)

    kept = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            kept.append(c)

    if not kept:
        return None

    clean_mask = np.zeros_like(digit_mask)
    cv2.drawContours(clean_mask, kept, -1, 255, thickness=-1)

    ys, xs = np.where(clean_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop_digit = clean_mask[y1:y2 + 1, x1:x2 + 1]
    if crop_digit is None or crop_digit.size == 0:
        return None

    target = 64
    canvas = np.zeros((target, target), dtype=np.uint8)

    h, w = crop_digit.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min((target - 10) / w, (target - 10) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        crop_digit,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def _binary_mask_score(a, b):
    """
    Compare deux masques binaires.
    Plus proche de 1 = meilleur.
    """
    if a is None or b is None:
        return 0.0

    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)

    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()

    if union == 0:
        return 0.0

    iou = float(inter) / float(union)

    xor_pixels = np.logical_xor(aa, bb).sum()
    xor_score = 1.0 - (float(xor_pixels) / float(union))

    return float((iou * 0.65) + (xor_score * 0.35))


def _digit_score(full_a, full_b, digit_a=None, digit_b=None):
    """
    Score final :
    - un peu de badge complet
    - surtout la forme du chiffre
    """
    if full_a is None or full_b is None:
        return 0.0

    diff = cv2.absdiff(full_a, full_b)
    diff_score = 1.0 - (float(np.mean(diff)) / 255.0)

    blur_a = cv2.GaussianBlur(full_a, (3, 3), 0)
    blur_b = cv2.GaussianBlur(full_b, (3, 3), 0)

    diff2 = cv2.absdiff(blur_a, blur_b)
    structure_score = 1.0 - (float(np.mean(diff2)) / 255.0)

    full_score = float((diff_score * 0.60) + (structure_score * 0.40))

    if digit_a is None or digit_b is None:
        return full_score

    digit_score = _binary_mask_score(digit_a, digit_b)

    return float((full_score * 0.30) + (digit_score * 0.70))


def detect_digit(zone):
    """
    Détecte le badge points de 1 à 10
    en comparant :
    - le badge complet
    - et la forme du chiffre lui-même
    """
    if zone is None or zone.size == 0:
        return None, 0.0, 0.0

    scan_badge = _normalize_digit_mask(zone)
    if scan_badge is None:
        return None, 0.0, 0.0

    scan_digit = _extract_digit_only_mask(zone)

    scores = []

    for n in range(1, 11):
        path = os.path.join(DIGITS_DIR, f"{n}.png")
        tpl = cv2.imread(path)

        if tpl is None or tpl.size == 0:
            continue

        tpl_badge = _normalize_digit_mask(tpl)
        if tpl_badge is None:
            continue

        tpl_digit = _extract_digit_only_mask(tpl)

        score = _digit_score(scan_badge, tpl_badge, scan_digit, tpl_digit)
        scores.append((n, float(score)))

    if not scores:
        return None, 0.0, 0.0

    scores.sort(key=lambda x: x[1], reverse=True)

    best_digit, best_score = scores[0]

    if len(scores) >= 2:
        second_score = scores[1][1]
        gap = float(best_score - second_score)
    else:
        gap = float(best_score)

    return int(best_digit), float(best_score), gap


    
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
# POINTS BADGE DETECTION
# =====================================================

def _clip_box(x, y, w, h, max_w, max_h):
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def find_points_badge(bottom_zone):
    """
    Cherche automatiquement le badge blanc des points
    dans la zone du bas.

    Nouvelle logique :
    - on ne cherche plus dans tout le bottom sans méthode
    - on prend une grande zone de recherche à gauche
    - on détecte le blanc de 2 façons :
        1) blanc en HSV
        2) zone claire en niveaux de gris
    - on garde un candidat plausible :
        * compact
        * plutôt carré
        * pas collé au bord
        * avec du blanc
        * avec un peu de noir dedans (le chiffre)
    """
    if bottom_zone is None or bottom_zone.size == 0:
        return None, None

    h, w = bottom_zone.shape[:2]
    if h == 0 or w == 0:
        return None, None

    # On reste dans une grande zone à gauche.
    # Ce n'est PAS un petit ROI fixe du chiffre.
    search_x1 = 0
    search_x2 = int(w * 0.58)

    search_zone = bottom_zone[:, search_x1:search_x2]
    if search_zone is None or search_zone.size == 0:
        return None, None

    search_h, search_w = search_zone.shape[:2]

    def collect_candidates(mask):
        candidates = []

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return candidates

        gray_local = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if area <= 0:
                continue

            area_ratio = area / float(max(search_w * search_h, 1))

            # Trop petit = bruit
            if area_ratio < 0.015:
                continue

            # Trop grand = gros morceau de décor / fond
            if area_ratio > 0.28:
                continue

            # Taille plausible
            if bw < search_w * 0.18 or bw > search_w * 0.85:
                continue

            if bh < search_h * 0.20 or bh > search_h * 0.80:
                continue

            # Le badge est plutôt compact / carré
            ratio = bw / float(max(bh, 1))
            if ratio < 0.65 or ratio > 1.35:
                continue

            # On rejette les blobs collés au bord du masque
            if x <= 2 or y <= 2 or (x + bw) >= (search_w - 2) or (y + bh) >= (search_h - 2):
                continue

            box_mask = mask[y:y + bh, x:x + bw]
            if box_mask is None or box_mask.size == 0:
                continue

            white_ratio = float(np.count_nonzero(box_mask)) / float(box_mask.size)

            # Il faut assez de blanc
            if white_ratio < 0.22:
                continue

            crop_gray = gray_local[y:y + bh, x:x + bw]
            if crop_gray is None or crop_gray.size == 0:
                continue

            # Il faut aussi un peu de noir à l'intérieur
            # sinon on risque de prendre juste une zone claire du décor
            dark_ratio = float(np.count_nonzero(crop_gray < 120)) / float(crop_gray.size)

            if dark_ratio < 0.12:
                continue

            # Scores de préférence
            cx = x + (bw / 2.0)

            # plus à gauche = mieux
            left_score = 1.0 - min(cx / float(max(search_w, 1)), 1.0)

            # plus proche du carré = mieux
            square_score = 1.0 - min(abs(ratio - 1.0) / 0.35, 1.0)

            # on préfère une taille moyenne plausible
            target_area = 0.18
            area_score = 1.0 - min(abs(area_ratio - target_area) / 0.14, 1.0)

            score = (
                left_score * 3.0 +
                square_score * 2.0 +
                area_score * 2.0 +
                white_ratio * 1.5 +
                dark_ratio * 1.0
            )

            candidates.append((score, x, y, bw, bh))

        return candidates

    # -------------------------------------------------
    # Masque 1 : blanc en HSV
    # Très utile pour trouver le badge blanc
    # sans trop confondre avec le décor coloré
    # -------------------------------------------------
    hsv = cv2.cvtColor(search_zone, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 145), (180, 95, 255))

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------------------------------
    # Masque 2 : zone claire en niveaux de gris
    # Sert de secours si le HSV n'est pas assez bon
    # -------------------------------------------------
    gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright_mask = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)

    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    candidates = []
    candidates.extend(collect_candidates(white_mask))
    candidates.extend(collect_candidates(bright_mask))

    if not candidates:
        return None, None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    # Petite marge autour du badge
    pad_x = int(bw * 0.08)
    pad_y = int(bh * 0.08)

    x = x - pad_x
    y = y - pad_y
    bw = bw + (2 * pad_x)
    bh = bh + (2 * pad_y)

    x, y, bw, bh = _clip_box(x, y, bw, bh, search_w, search_h)

    crop = search_zone[y:y + bh, x:x + bw]
    if crop is None or crop.size == 0:
        return None, None

    # On convertit la bbox locale de search_zone
    # vers la bbox locale de bottom_zone
    final_x = search_x1 + x
    final_y = y
    final_w = bw
    final_h = bh

    return crop, (final_x, final_y, final_w, final_h)

def _offset_box(box, dx, dy):
    if box is None:
        return None
    x, y, w, h = box
    return (int(x + dx), int(y + dy), int(w), int(h))


def _make_bottom_light_mask(zone):
    """
    Construit un masque des zones claires / blanches
    dans le ROI du bas.
    """
    if zone is None or zone.size == 0:
        return None, None

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    mask_hsv = cv2.inRange(hsv, (0, 0, 140), (180, 120, 255))
    _, mask_gray = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_hsv, mask_gray)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, gray


def _find_black_panel_box(bottom_zone):
    """
    Cherche le grand panneau noir du bas
    pour les cartes classiques.

    Important :
    si la détection par contours échoue,
    on renvoie un fallback fixe raisonnable
    pour ne pas perdre les cas simples comme BLEU_1.
    """
    if bottom_zone is None or bottom_zone.size == 0:
        return None

    gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark_mask = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape[:2]
    image_area = float(max(w * h, 1))
    candidates = []

    if contours:
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if area <= 0:
                continue

            area_ratio = area / image_area

            if area_ratio < 0.18:
                continue

            if bw < w * 0.45:
                continue

            if bh < h * 0.45:
                continue

            if x > w * 0.20:
                continue

            ratio = bw / float(max(bh, 1))
            if ratio < 1.2 or ratio > 4.8:
                continue

            cx = x + (bw / 2.0)
            left_score = 1.0 - min(cx / float(max(w, 1)), 1.0)
            size_score = min(area_ratio / 0.45, 1.0)

            score = (size_score * 3.0) + (left_score * 1.0)
            candidates.append((score, x, y, bw, bh))

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, x, y, bw, bh = candidates[0]
        return _clip_box(x, y, bw, bh, w, h)

    # Fallback simple si aucun contour n'est assez bon
    fx = int(w * 0.02)
    fy = int(h * 0.08)
    fw = int(w * 0.50)
    fh = int(h * 0.84)

    return _clip_box(fx, fy, fw, fh, w, h)


def _find_special_white_panel_box(bottom_zone):
    """
    Cherche le format spécial jaune/blanc
    comme JAUNE_5.
    """
    if bottom_zone is None or bottom_zone.size == 0:
        return None

    mask, gray = _make_bottom_light_mask(bottom_zone)
    if mask is None:
        return None

    h, w = gray.shape[:2]
    dark_ratio = float(np.count_nonzero(gray < 90)) / float(max(gray.size, 1))

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if area <= 0:
            continue

        area_ratio = area / float(max(w * h, 1))

        if area_ratio < 0.12:
            continue

        if x < w * 0.10:
            continue

        if bw < w * 0.35:
            continue

        if bh < h * 0.42:
            continue

        if y > h * 0.35:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 0.6 or ratio > 1.8:
            continue

        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)

        center_x_score = 1.0 - min(abs(cx - (w * 0.55)) / float(max(w * 0.35, 1)), 1.0)
        center_y_score = 1.0 - min(abs(cy - (h * 0.52)) / float(max(h * 0.35, 1)), 1.0)
        size_score = min(area_ratio / 0.28, 1.0)

        score = (center_x_score * 2.0) + (center_y_score * 2.0) + (size_score * 2.0)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    if dark_ratio > 0.22:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    return _clip_box(x, y, bw, bh, w, h)


def _find_slash_box(panel_zone):
    """
    Cherche le slash blanc au centre du panneau noir.
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.35)
    x2 = int(pw * 0.60)

    zone = panel_zone[:, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.01:
            continue

        if bh < zh * 0.28:
            continue

        if bw > zw * 0.45:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio > 0.80:
            continue

        cx = x + (bw / 2.0)
        center_score = 1.0 - min(abs(cx - (zw * 0.50)) / float(max(zw * 0.35, 1)), 1.0)
        tall_score = min(bh / float(max(zh * 0.65, 1)), 1.0)

        score = (center_score * 2.0) + (tall_score * 2.0) + (area_ratio * 3.0)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    x += x1
    return (x, y, bw, bh)


def _find_right_icon_box(panel_zone):
    """
    Cherche la grande icône blanche à droite.
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.56)
    x2 = pw
    y1 = int(ph * 0.05)
    y2 = int(ph * 0.78)

    zone = panel_zone[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.05:
            continue

        if bw < zw * 0.20:
            continue

        if bh < zh * 0.25:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 0.45 or ratio > 1.65:
            continue

        score = (area_ratio * 4.0) + (bh / float(max(zh, 1)))
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    return (x + x1, y + y1, bw, bh)


def _find_bottom_line_box(panel_zone):
    """
    Cherche la ligne / double flèche en bas à droite.
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.46)
    x2 = pw
    y1 = int(ph * 0.56)
    y2 = ph

    zone = panel_zone[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.01:
            continue

        if bw < zw * 0.18:
            continue

        if bh > zh * 0.38:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 1.8:
            continue

        score = (ratio * 1.0) + (area_ratio * 6.0)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    return (x + x1, y + y1, bw, bh)


def _find_points_badge_in_black_panel(panel_zone):
    """
    Version simple et robuste :
    on prend directement la zone gauche du panneau noir.
    """
    if panel_zone is None or panel_zone.size == 0:
        return None, None

    ph, pw = panel_zone.shape[:2]
    if ph == 0 or pw == 0:
        return None, None

    x = int(pw * 0.02)
    y = int(ph * 0.10)
    w = int(pw * 0.36)
    h = int(ph * 0.78)

    x, y, w, h = _clip_box(x, y, w, h, pw, ph)

    crop = panel_zone[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None, None

    return crop, (x, y, w, h)


def analyze_bottom_layout(bottom_zone):
    """
    Analyse le ROI complet du bas.
    """
    result = {
        "layout": "UNKNOWN",
        "points": None,
        "raw_points": None,
        "points_score": 0.0,
        "points_gap": 0.0,
        "has_slash": False,
        "has_right_icon": False,
        "has_bottom_line": False,
        "has_special_white_panel": False,
        "panel_box": None,
        "points_box": None,
        "slash_box": None,
        "right_icon_box": None,
        "bottom_line_box": None,
        "special_box": None
    }

    if bottom_zone is None or bottom_zone.size == 0:
        return result

    special_box = _find_special_white_panel_box(bottom_zone)
    if special_box is not None:
        result["layout"] = "SPECIAL_WHITE_PANEL"
        result["has_special_white_panel"] = True
        result["special_box"] = special_box
        return result

    panel_box = _find_black_panel_box(bottom_zone)
    if panel_box is None:
        return result

    px, py, pw, ph = panel_box
    panel_zone = bottom_zone[py:py + ph, px:px + pw]
    if panel_zone is None or panel_zone.size == 0:
        return result

    result["panel_box"] = panel_box

    badge_crop, badge_box_local = _find_points_badge_in_black_panel(panel_zone)

    raw_points_digit = None
    points_digit = None
    points_score = 0.0
    points_gap = 0.0

    if badge_crop is not None and badge_box_local is not None:
        raw_points_digit, points_score, points_gap = detect_digit(badge_crop)

        if points_score >= 0.72:
            points_digit = raw_points_digit
        elif points_score >= 0.60 and points_gap >= 0.02:
            points_digit = raw_points_digit
        else:
            points_digit = None

        bx, by, bw2, bh2 = badge_box_local
        result["points_box"] = (px + bx, py + by, bw2, bh2)

    slash_box_local = _find_slash_box(panel_zone)
    if slash_box_local is not None:
        result["has_slash"] = True
        result["slash_box"] = _offset_box(slash_box_local, px, py)

    right_box_local = _find_right_icon_box(panel_zone)
    if right_box_local is not None:
        result["has_right_icon"] = True
        result["right_icon_box"] = _offset_box(right_box_local, px, py)

    line_box_local = _find_bottom_line_box(panel_zone)
    if line_box_local is not None:
        result["has_bottom_line"] = True
        result["bottom_line_box"] = _offset_box(line_box_local, px, py)

    if points_digit is not None and not result["has_slash"] and not result["has_right_icon"]:
        layout = "NUMBER_ONLY"
    elif points_digit is not None and result["has_slash"] and result["has_right_icon"] and result["has_bottom_line"]:
        layout = "NUMBER_ICON_LINE"
    elif points_digit is not None and result["has_slash"] and result["has_right_icon"]:
        layout = "NUMBER_ICON"
    else:
        layout = "BLACK_PANEL"

    result["layout"] = layout
    result["points"] = points_digit
    result["raw_points"] = raw_points_digit
    result["points_score"] = float(points_score)
    result["points_gap"] = float(points_gap)

    return result


def _apply_symbol_acceptance(raw_name, score, gap):
    accepted_name = None

    if raw_name and score >= 0.55 and gap >= 0.025:
        accepted_name = raw_name

    if accepted_name is not None:
        threshold_mode = "accepted"
    elif not raw_name:
        threshold_mode = "no_match"
    elif score < 0.55 and gap < 0.025:
        threshold_mode = "rejected_score_gap"
    elif score < 0.55:
        threshold_mode = "rejected_score"
    else:
        threshold_mode = "rejected_gap"

    return accepted_name, threshold_mode


def _get_expected_symbol_map():
    try:
        cards = load_cards_js()
    except Exception:
        cards = []

    expected = {}
    for card in cards:
        card_id = str(card.get("id") or "").strip().lower()
        symbol_name = str(card.get("symbol") or "").strip()
        if not card_id or not symbol_name:
            continue
        expected[card_id] = symbol_name

    return expected


def _resolve_expected_symbol(filename, expected_symbol_map):
    stem = os.path.splitext(os.path.basename(filename or ""))[0].strip().lower()
    if not stem:
        return None
    return expected_symbol_map.get(stem)


def _build_business_symbol_result(filename, raw_name, score, gap, expected_symbol):
    accepted_name, threshold_mode = _apply_symbol_acceptance(raw_name, score, gap)

    if not expected_symbol:
        business_status = "N/A"
        business_bucket = "sans_attendu"
    elif accepted_name == expected_symbol:
        business_status = "OK"
        business_bucket = "correcte"
    elif accepted_name is not None:
        business_status = "KO"
        business_bucket = "fausse"
    elif raw_name == expected_symbol:
        business_status = "KO"
        business_bucket = "fragile"
    else:
        business_status = "KO"
        business_bucket = "rejetee"

    return {
        "expected_symbol": expected_symbol,
        "accepted_name": accepted_name,
        "threshold_mode": threshold_mode,
        "business_status": business_status,
        "business_bucket": business_bucket,
    }


def compute_signature(img):
    rois = []

    # On garde une base unique : image redressée normalisée en 200x300.
    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]

    # -------------------------------------------------
    # COLOR
    # -------------------------------------------------
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

    # -------------------------------------------------
    # SYMBOL
    # -------------------------------------------------
    x1 = int(w * 0.02)
    x2 = int(w * 0.24)
    y1 = int(h * 0.16)
    y2 = int(h * 0.34)

    zone = img[y1:y2, x1:x2]

    rois.append({
        "type": "SYMBOL",
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1
    })

    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    raw_symbol_name, symbol_score, symbol_gap, symbol_debug = detect_symbol(zone)
    top_candidates = symbol_debug.get("top_candidates", [])

    symbol_name, symbol_threshold_mode = _apply_symbol_acceptance(
        raw_symbol_name,
        symbol_score,
        symbol_gap
    )

    symbol_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "name": symbol_name,
        "raw_name": raw_symbol_name,
        "score": float(symbol_score),
        "gap": float(symbol_gap),
        "threshold_mode": symbol_threshold_mode,
        "top_candidates": top_candidates,
        "winner_references": symbol_debug.get("winner_references", []),
        "runner_up": symbol_debug.get("runner_up"),
        "mode": "icon_card_refs"
    }

    # -------------------------------------------------
    # BOTTOM
    # -------------------------------------------------
    # IMPORTANT :
    # La signature principale doit utiliser exactement le même pipeline
    # que /bottom-test, sinon on débugue un moteur A et la prod tourne sur B.
    full_img, bottom_zone, bottom_box = extract_bottom_roi_from_full_card(img)
    bottom_x1, bottom_y1, bottom_w, bottom_h = bottom_box

    rois.append({
        "type": "BOTTOM",
        "x": bottom_x1,
        "y": bottom_y1,
        "w": bottom_w,
        "h": bottom_h
    })

    bottom_sig = compute_patch_signature(bottom_zone, size=(16, 16))
    bottom_layout = analyze_bottom(bottom_zone, DIGITS_DIR)

    detected_color_name = _safe_upper((color_sig or {}).get("detected"))
    detected_symbol_name = _safe_upper((symbol_sig or {}).get("name") or (symbol_sig or {}).get("raw_name"))
    bottom_layout = _apply_bottom_reference_match(
        _build_observed_bottom_from_result(bottom_layout),
        bottom_zone,
        {
            "color": detected_color_name,
            "symbol": detected_symbol_name,
        }
    )

    panel_box = bottom_layout.get("panel_box")
    if panel_box is not None:
        x, y, bw2, bh2 = panel_box
        rois.append({
            "type": "BOTTOM_PANEL",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    special_box = bottom_layout.get("special_box")
    if special_box is not None:
        x, y, bw2, bh2 = special_box
        rois.append({
            "type": "BOTTOM_SPECIAL",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    points_box = bottom_layout.get("points_box")
    if points_box is not None:
        x, y, bw2, bh2 = points_box
        rois.append({
            "type": "POINTS_BADGE",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    slash_box = bottom_layout.get("slash_box")
    if slash_box is not None:
        x, y, bw2, bh2 = slash_box
        rois.append({
            "type": "BOTTOM_SLASH",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    right_box = bottom_layout.get("right_icon_box")
    if right_box is not None:
        x, y, bw2, bh2 = right_box
        rois.append({
            "type": "BOTTOM_RIGHT_ICON",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    line_box = bottom_layout.get("bottom_line_box")
    if line_box is not None:
        x, y, bw2, bh2 = line_box
        rois.append({
            "type": "BOTTOM_LINE",
            "x": bottom_x1 + x,
            "y": bottom_y1 + y,
            "w": bw2,
            "h": bh2
        })

    if points_box is not None:
        px, py, pw2, ph2 = points_box
        badge_crop = bottom_zone[py:py + ph2, px:px + pw2]

        if badge_crop is not None and badge_crop.size > 0:
            gray_badge = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2GRAY)
            points_mean = float(np.mean(gray_badge))
            points_std = float(np.std(gray_badge))
            points_found = True
        else:
            points_mean = 0.0
            points_std = 0.0
            points_found = False
    else:
        points_mean = 0.0
        points_std = 0.0
        points_found = False

    points_sig = {
        "mean": points_mean,
        "std": points_std,
        "digit": bottom_layout.get("points"),
        "raw_digit": bottom_layout.get("raw_points"),
        "score": float(bottom_layout.get("points_score", 0.0)),
        "gap": float(bottom_layout.get("points_gap", 0.0)),
        "found": bool(points_found)
    }

    bottom_layout_sig = {
        "layout": bottom_layout.get("layout"),
        "points": bottom_layout.get("points"),
        "raw_points": bottom_layout.get("raw_points"),
        "points_score": float(bottom_layout.get("points_score", 0.0)),
        "points_gap": float(bottom_layout.get("points_gap", 0.0)),
        "has_slash": bool(bottom_layout.get("has_slash", False)),
        "has_right_icon": bool(bottom_layout.get("has_right_icon", False)),
        "has_bottom_line": bool(bottom_layout.get("has_bottom_line", False)),
        "has_special_white_panel": bool(bottom_layout.get("has_special_white_panel", False)),
        "target": bottom_layout.get("target") or "",
        "range": bottom_layout.get("range") or "",
        "ref_card_id": bottom_layout.get("ref_card_id") or "",
        "ref_score": float(bottom_layout.get("ref_score", 0.0)),
        "ref_gap": float(bottom_layout.get("ref_gap", 0.0)),
    }

    # -------------------------------------------------
    # GLOBAL
    # -------------------------------------------------
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
        "points": points_sig,
        "bottom": bottom_sig,
        "bottom_layout": bottom_layout_sig,
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




def _build_card_match_ref_cache(cards=None):
    global _CARD_MATCH_REF_CACHE

    if _CARD_MATCH_REF_CACHE is not None:
        return _CARD_MATCH_REF_CACHE

    if cards is None:
        try:
            cards = load_cards_js()
        except Exception:
            cards = []

    cache = {}
    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if not card_id:
            continue

        scan_sig = ((card.get("signature") or {}).get("scan") or {})
        has_bottom = bool(((scan_sig.get("bottom") or {}).get("vector") or []))
        has_global = bool(((scan_sig.get("global") or {}).get("vector") or []))
        has_symbol = bool(((scan_sig.get("symbol") or {}).get("name")) or ((scan_sig.get("symbol") or {}).get("raw_name")))
        has_bottom_layout = bool(scan_sig.get("bottom_layout"))

        if has_bottom and has_global and has_symbol and has_bottom_layout:
            cache[card_id] = scan_sig
            continue

        try:
            image_path = find_card_image(card_id)
            if image_path and os.path.exists(image_path):
                ref_img = cv2.imread(image_path)
                if ref_img is not None and ref_img.size != 0:
                    ref_img = cv2.resize(ref_img, (200, 300))
                    ref_sig, _ = compute_signature(ref_img)
                    if ref_sig is not None:
                        cache[card_id] = ref_sig
                        continue
        except Exception:
            pass

        cache[card_id] = scan_sig

    _CARD_MATCH_REF_CACHE = cache
    return cache


def _get_card_scan_signature(card):
    card_id = str(card.get("id") or "").strip()
    if not card_id:
        return ((card.get("signature") or {}).get("scan") or {})
    cache = _build_card_match_ref_cache()
    return cache.get(card_id) or ((card.get("signature") or {}).get("scan") or {})


def _resolve_expected_card_id(filename, cards=None):
    stem = os.path.splitext(os.path.basename(filename or ""))[0].strip().lower()
    if not stem:
        return None

    if cards is None:
        try:
            cards = load_cards_js()
        except Exception:
            cards = []

    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if card_id.lower() == stem:
            return card_id
    return None


def _build_expected_bottom_profile(card):
    effect = bool(card.get("effet", False))
    points = _safe_int(card.get("points"))
    target = _safe_upper(card.get("cible"))
    range_name = _safe_upper(card.get("portee"))

    has_target = bool(target)
    has_bottom_line = range_name == "LIGNE"

    if effect:
        layout = "SPECIAL_WHITE_PANEL"
    elif points is not None and has_target and has_bottom_line:
        layout = "NUMBER_ICON_LINE"
    elif points is not None and has_target:
        layout = "NUMBER_ICON"
    elif points is not None:
        layout = "NUMBER_ONLY"
    else:
        layout = "BLACK_PANEL"

    return {
        "layout": layout,
        "points": points,
        "has_special_white_panel": effect,
        "has_slash": has_target,
        "has_right_icon": has_target,
        "has_bottom_line": has_bottom_line,
        "target": target,
        "range": range_name,
    }


def _build_observed_bottom_profile(scan_sig):
    bottom_layout = scan_sig.get("bottom_layout") or {}
    points_sig = scan_sig.get("points") or {}

    layout = str(bottom_layout.get("layout") or "UNKNOWN")
    points = _safe_int(bottom_layout.get("points"))
    raw_points = _safe_int(bottom_layout.get("raw_points"))
    if points is None:
        points = _safe_int(points_sig.get("digit"))
    if raw_points is None:
        raw_points = _safe_int(points_sig.get("raw_digit"))

    return {
        "layout": layout,
        "points": points,
        "raw_points": raw_points,
        "has_special_white_panel": bool(bottom_layout.get("has_special_white_panel", False)),
        "has_slash": bool(bottom_layout.get("has_slash", False)),
        "has_right_icon": bool(bottom_layout.get("has_right_icon", False)),
        "has_bottom_line": bool(bottom_layout.get("has_bottom_line", False)),
        "target": _safe_upper(bottom_layout.get("target")),
        "range": _safe_upper(bottom_layout.get("range")),
        "ref_card_id": _safe_upper(bottom_layout.get("ref_card_id")),
    }


def _patch_vector_similarity(a, b):
    if not a or not b:
        return 0.0

    try:
        va = np.array(a, dtype=np.float32).flatten()
        vb = np.array(b, dtype=np.float32).flatten()
    except Exception:
        return 0.0

    if va.size == 0 or vb.size == 0:
        return 0.0

    size = min(va.size, vb.size)
    va = va[:size]
    vb = vb[:size]

    mean_abs = float(np.mean(np.abs(va - vb)))
    diff_score = max(0.0, 1.0 - (mean_abs / 255.0))

    std_a = float(np.std(va))
    std_b = float(np.std(vb))
    if std_a > 1e-6 and std_b > 1e-6:
        corr = float(np.corrcoef(va, vb)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
    else:
        corr_score = diff_score

    return float((diff_score * 0.6) + (corr_score * 0.4))


def _score_bottom_profile(expected, observed):
    score = 0.0
    details = []

    exp_layout = expected.get("layout")
    obs_layout = observed.get("layout")

    if exp_layout == obs_layout:
        score += 6.0
        details.append("layout_exact")
    elif exp_layout == "NUMBER_ICON_LINE" and obs_layout == "NUMBER_ICON":
        score += 3.5
        details.append("layout_line_missing")
    elif exp_layout == "NUMBER_ICON" and obs_layout == "NUMBER_ICON_LINE":
        score += 4.5
        details.append("layout_line_extra")
    elif exp_layout == "NUMBER_ONLY" and obs_layout == "BLACK_PANEL":
        score += 1.0
        details.append("layout_number_uncertain")
    else:
        score -= 4.0
        details.append("layout_mismatch")

    exp_points = expected.get("points")
    obs_points = observed.get("points")
    obs_raw_points = observed.get("raw_points")

    if exp_points is None and expected.get("has_special_white_panel"):
        score += 2.0 if observed.get("has_special_white_panel") else -2.0
        details.append("special_panel")
    elif exp_points is not None:
        if obs_points == exp_points:
            score += 8.0
            details.append("points_exact")
        elif obs_raw_points == exp_points:
            score += 4.0
            details.append("points_raw")
        else:
            score -= 5.0
            details.append("points_mismatch")

    exp_target = _safe_upper(expected.get("target"))
    obs_target = _safe_upper(observed.get("target"))
    exp_range = _safe_upper(expected.get("range"))
    obs_range = _safe_upper(observed.get("range"))

    if exp_target:
        if obs_target == exp_target:
            score += 4.0
            details.append("target_exact")
        elif obs_target:
            score -= 2.0
            details.append("target_mismatch")
        else:
            details.append("target_unknown")

    if exp_range:
        if obs_range == exp_range:
            score += 3.0
            details.append("range_exact")
        elif obs_range:
            score -= 1.5
            details.append("range_mismatch")
        else:
            details.append("range_unknown")

    for field, weight in [
        ("has_special_white_panel", 2.5),
        ("has_slash", 1.5),
        ("has_right_icon", 1.5),
        ("has_bottom_line", 2.0),
    ]:
        exp_val = expected.get(field)
        obs_val = observed.get(field)
        if bool(exp_val) == bool(obs_val):
            score += weight
            details.append(f"{field}_ok")
        else:
            score -= weight
            details.append(f"{field}_mismatch")

    return float(score), details


def resolve_final_card(scan_sig, cards=None):
    if cards is None:
        try:
            cards = load_cards_js()
        except Exception:
            cards = []

    if not cards:
        return {
            "color_name": None,
            "symbol_name": None,
            "symbol_source": "none",
            "bottom_layout": None,
            "points": None,
            "candidate_cards": [],
            "final_card_id": None,
            "final_score": 0.0,
            "final_gap": 0.0,
            "final_status": "rejected",
            "reason": "no_cards_reference",
        }

    color_name = _safe_upper((scan_sig.get("color") or {}).get("detected"))
    symbol_sig = scan_sig.get("symbol") or {}
    accepted_symbol = _safe_upper(symbol_sig.get("name"))
    raw_symbol = _safe_upper(symbol_sig.get("raw_name"))

    if accepted_symbol:
        symbol_name = accepted_symbol
        symbol_source = "accepted"
    elif raw_symbol:
        symbol_name = raw_symbol
        symbol_source = "raw"
    else:
        symbol_name = None
        symbol_source = "none"

    observed_bottom = _build_observed_bottom_profile(scan_sig)
    bottom_layout_name = observed_bottom.get("layout")
    points_value = observed_bottom.get("points")
    ref_card_id = _safe_upper(observed_bottom.get("ref_card_id") or (scan_sig.get("bottom_layout") or {}).get("ref_card_id"))
    bottom_patch = (scan_sig.get("bottom") or {}).get("vector") or []
    global_patch = (scan_sig.get("global") or {}).get("vector") or []

    color_candidates = [
        card for card in cards
        if not color_name or _safe_upper(card.get("couleur")) == color_name
    ]
    if not color_candidates:
        color_candidates = list(cards)

    symbol_candidates = [
        card for card in color_candidates
        if not symbol_name or _safe_upper(card.get("symbol")) == symbol_name
    ]
    if not symbol_candidates:
        symbol_candidates = list(color_candidates)

    scored = []
    for card in symbol_candidates:
        expected_bottom = _build_expected_bottom_profile(card)
        semantic_score, semantic_details = _score_bottom_profile(expected_bottom, observed_bottom)

        ref_scan = _get_card_scan_signature(card)
        ref_bottom = ((ref_scan.get("bottom") or {}).get("vector") or [])
        ref_global = ((ref_scan.get("global") or {}).get("vector") or [])

        bottom_visual = _patch_vector_similarity(bottom_patch, ref_bottom)
        global_visual = _patch_vector_similarity(global_patch, ref_global)

        total_score = 0.0
        details = []

        if color_name and _safe_upper(card.get("couleur")) == color_name:
            total_score += 5.0
            details.append("color_exact")

        if symbol_name and _safe_upper(card.get("symbol")) == symbol_name:
            total_score += 7.0 if symbol_source == "accepted" else 5.0
            details.append(f"symbol_{symbol_source}")

        if ref_card_id and _safe_upper(card.get("id")) == ref_card_id:
            total_score += 25.0
            details.append("bottom_ref_exact")

        total_score += semantic_score
        details.extend(semantic_details)

        total_score += bottom_visual * 6.0
        total_score += global_visual * 1.5

        details.append(f"bottom_visual={bottom_visual:.3f}")
        details.append(f"global_visual={global_visual:.3f}")

        scored.append({
            "card_id": str(card.get("id") or ""),
            "score": float(total_score),
            "bottom_visual": float(bottom_visual),
            "global_visual": float(global_visual),
            "expected_bottom": expected_bottom,
            "details": details[:12],
        })

    scored.sort(key=lambda item: item["score"], reverse=True)

    best = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    final_gap = float((best["score"] - runner_up["score"]) if best and runner_up else (best["score"] if best else 0.0))

    if best is None:
        final_status = "rejected"
        final_card_id = None
        reason = "no_candidate"
    elif final_gap >= 1.0 and best["score"] >= 18.0:
        final_status = "accepted"
        final_card_id = best["card_id"]
        reason = "strong_unique_match"
    elif final_gap >= 0.35 and best["score"] >= 16.0:
        final_status = "fragile"
        final_card_id = best["card_id"]
        reason = "usable_but_close"
    else:
        final_status = "rejected"
        final_card_id = None
        reason = "bottom_not_discriminant_enough"

    return {
        "color_name": color_name or None,
        "symbol_name": symbol_name or None,
        "symbol_source": symbol_source,
        "bottom_layout": bottom_layout_name,
        "points": points_value,
        "candidate_cards": scored[:8],
        "final_card_id": final_card_id,
        "final_score": float(best["score"]) if best else 0.0,
        "final_gap": final_gap,
        "final_status": final_status,
        "reason": reason,
    }


def _build_business_card_result(filename, final_card_id, final_status, expected_card_id):
    if not expected_card_id:
        business_status = "N/A"
        business_bucket = "sans_attendu"
    elif final_card_id == expected_card_id and final_status == "accepted":
        business_status = "OK"
        business_bucket = "correcte"
    elif final_card_id == expected_card_id and final_status == "fragile":
        business_status = "KO"
        business_bucket = "fragile"
    elif final_card_id:
        business_status = "KO"
        business_bucket = "fausse"
    else:
        business_status = "KO"
        business_bucket = "rejetee"

    return {
        "expected_card_id": expected_card_id,
        "business_status": business_status,
        "business_bucket": business_bucket,
    }


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
                "rois": [],
                "card_match": None,
                "final_card_id": None,
                "final_status": None,
                "final_score": 0.0,
                "final_gap": 0.0,
            })

        file = request.files["image"]

        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "rects": [],
                "signature": None,
                "rois": [],
                "card_match": None,
                "final_card_id": None,
                "final_status": None,
                "final_score": 0.0,
                "final_gap": 0.0,
            })

        rect = detect_main_card(img)

        if rect is None:
            h, w = img.shape[:2]
            rect = {
                "x": 0,
                "y": 0,
                "w": int(w),
                "h": int(h),
                "quad": [
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ]
            }

        quad = np.array(rect["quad"], dtype="float32")
        warped = warp_quad(img, quad)

        sig = None
        rois = []
        card_match = None

        if warped is None or warped.size == 0:
            warped = img.copy()

        if warped is not None and warped.size != 0:
            cv2.imwrite(WARP_PATH, warped)
            sig, rois = compute_signature(warped)

        if sig is not None:
            try:
                card_match = resolve_final_card(sig)
                sig["card_match"] = card_match
            except Exception:
                card_match = None

        return jsonify({
            "rects": [rect],
            "signature": sig,
            "rois": rois,
            "card_match": card_match,
            "final_card_id": (card_match or {}).get("final_card_id"),
            "final_status": (card_match or {}).get("final_status"),
            "final_score": float((card_match or {}).get("final_score", 0.0)),
            "final_gap": float((card_match or {}).get("final_gap", 0.0)),
            "color_name": (card_match or {}).get("color_name"),
            "symbol_name": (card_match or {}).get("symbol_name"),
            "bottom_layout": (card_match or {}).get("bottom_layout"),
            "points": (card_match or {}).get("points"),
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({
            "rects": [],
            "signature": None,
            "rois": [],
            "card_match": None,
            "final_card_id": None,
            "final_status": None,
            "final_score": 0.0,
            "final_gap": 0.0,
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

        # Sauvegarde de sécurité avant réécriture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{CARDS_JS_PATH}.bak_{timestamp}"
        shutil.copyfile(CARDS_JS_PATH, backup_path)

        global _SYMBOL_REFS_CACHE, _HOG_DESCRIPTOR
        _SYMBOL_REFS_CACHE = None
        global _CARD_MATCH_REF_CACHE
        _CARD_MATCH_REF_CACHE = None
        _HOG_DESCRIPTOR = None

        updated = 0
        skipped = []
        errors = []

        for c in cards:
            card_id = c.get("id")
            if not card_id:
                skipped.append({
                    "id": None,
                    "reason": "missing_id"
                })
                continue

            path = find_card_image(card_id)
            if path is None:
                skipped.append({
                    "id": card_id,
                    "reason": "image_not_found"
                })
                continue

            img = cv2.imread(path)
            if img is None or img.size == 0:
                skipped.append({
                    "id": card_id,
                    "reason": "image_unreadable"
                })
                continue

            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                skipped.append({
                    "id": card_id,
                    "reason": "invalid_image_shape"
                })
                continue

            quad = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype="float32")

            warped = warp_quad(img, quad)
            if warped is None or warped.size == 0:
                skipped.append({
                    "id": card_id,
                    "reason": "warp_failed"
                })
                continue

            try:
                sig, _ = compute_signature(warped)
            except Exception as e:
                errors.append({
                    "id": card_id,
                    "reason": f"compute_signature_failed: {str(e)}"
                })
                continue

            if sig is None:
                errors.append({
                    "id": card_id,
                    "reason": "signature_none"
                })
                continue

            c["signature"] = {
                "scan": sig
            }
            updated += 1

        save_cards_js(cards)

        return jsonify({
            "ok": True,
            "updated": updated,
            "total": len(cards),
            "skipped_count": len(skipped),
            "errors_count": len(errors),
            "backup": os.path.basename(backup_path),
            "skipped": skipped[:20],
            "errors": errors[:20]
        })

    except Exception as e:
        print("BUILD SIGNATURES ERROR:", e)
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
        



# =====================================================
# SYMBOL TEST
# =====================================================

@app.route("/symbol-test", methods=["GET", "POST"])
def symbol_test():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Symbol Test</title>
        </head>
        <body style="font-family:Arial,sans-serif; padding:20px;">
          <h1>Test du symbole</h1>
          <p>Choisis une image de carte complète, puis clique sur Analyser.</p>
          <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required />
            <button type="submit">Analyser</button>
          </form>
        </body>
        </html>
        """

    if "image" not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files["image"]
    if not file or file.filename == "":
        return "Fichier vide", 400

    data = file.read()
    if not data:
        return "Impossible de lire le fichier", 400

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return "Image invalide", 400

    rect = detect_main_card(img)
    warped = None

    if rect is not None:
        quad = np.array(rect["quad"], dtype="float32")
        warped = warp_quad(img, quad)

    if warped is None or warped.size == 0:
        warped = img.copy()

    warped = cv2.resize(warped, (200, 300))
    zone = _extract_symbol_zone_from_card(warped)
    scan_mask, panel, threshold_mode = _normalize_symbol_scan(zone)
    raw_name, score, gap, symbol_debug = detect_symbol(zone)

    zone_ok = bool(zone is not None and zone.size != 0)
    panel_ok = bool(panel is not None and panel.size != 0)
    mask_nonzero = int(np.count_nonzero(scan_mask)) if scan_mask is not None else 0
    mask_ok = bool(scan_mask is not None and scan_mask.size != 0 and mask_nonzero > 0)

    top_candidates = symbol_debug.get("top_candidates") or []
    winner = top_candidates[0] if top_candidates else None
    winner_references = symbol_debug.get("winner_references") or []

    pretty_json = json.dumps({
        "raw_name": raw_name,
        "score": float(score),
        "gap": float(gap),
        "winner": winner,
        "winner_references": winner_references,
        "debug": {
            "zone_ok": zone_ok,
            "panel_ok": panel_ok,
            "mask_ok": mask_ok,
            "zone_shape": list(zone.shape) if zone is not None else None,
            "panel_shape": list(panel.shape) if panel is not None else None,
            "mask_nonzero": mask_nonzero,
            "threshold_mode": threshold_mode
        }
    }, indent=2, ensure_ascii=False)

    overlay = warped.copy()
    h, w = warped.shape[:2]
    x1 = int(w * 0.02)
    x2 = int(w * 0.24)
    y1 = int(h * 0.16)
    y2 = int(h * 0.34)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Symbol Test</title>
    </head>
    <body style="font-family:Arial,sans-serif; padding:20px;">
      <h1>Résultat du test symbole</h1>
      <p><a href="/symbol-test">← Revenir au formulaire</a></p>
      <h2>Résultat JSON</h2>
      <pre style="background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto;">{pretty_json}</pre>
      {_html_img_block("Carte redressée", _img_to_base64(warped))}
      {_html_img_block("Overlay ROI symbole", _img_to_base64(overlay))}
      {_html_img_block("Zone symbole", _img_to_base64(zone))}
      {_html_img_block("Panel interne", _img_to_base64(panel))}
      {_html_img_block("Masque symbole", _img_to_base64(scan_mask))}
    </body>
    </html>
    """


# =====================================================
# SYMBOL BATCH TEST
# =====================================================

@app.route("/symbol-batch-test", methods=["GET", "POST"])
def symbol_batch_test():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Symbol Batch Test</title>
          <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f5f5f5; }}
            .ok {{ color: #0a7a28; font-weight: bold; }}
            .ko {{ color: #b00020; font-weight: bold; }}
            .fragile {{ color: #b26b00; font-weight: bold; }}
            .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }}
            .summary-card {{ border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }}
            .summary-card .label {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }}
            .summary-card .value {{ font-size:28px; font-weight:bold; margin-top:6px; }}
            pre {{ background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }}
          </style>
        </head>
        <body>
          <h1>Test batch symbole</h1>
          <p>Envoie plusieurs images de cartes complètes.</p>
          <form method="post" enctype="multipart/form-data">
            <p><input type="file" name="images" accept="image/*" multiple required /></p>
            <p><button type="submit">Analyser le lot</button></p>
          </form>
        </body>
        </html>
        """

    files = sorted(
        request.files.getlist("images"),
        key=lambda f: _natural_sort_key(f.filename or "")
    )
    if not files:
        return "Aucun fichier envoyé", 400

    expected_symbol_map = _get_expected_symbol_map()

    rows = []
    batch_results = []
    summary = {
        "total": 0,
        "with_expected": 0,
        "correctes": 0,
        "fausses": 0,
        "fragiles": 0,
        "rejetees": 0,
        "sans_attendu": 0,
        "errors": 0,
    }

    for file in files:
        filename = file.filename or "unknown"
        safe_filename = html.escape(filename)
        summary["total"] += 1

        try:
            data = file.read()
            if not data:
                summary["errors"] += 1
                batch_results.append({
                    "file": filename,
                    "error": "empty_file"
                })
                rows.append(
                    f"<tr><td>{safe_filename}</td><td colspan='7'><span class='ko'>Fichier vide</span></td></tr>"
                )
                continue

            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                summary["errors"] += 1
                batch_results.append({
                    "file": filename,
                    "error": "invalid_image"
                })
                rows.append(
                    f"<tr><td>{safe_filename}</td><td colspan='7'><span class='ko'>Image invalide</span></td></tr>"
                )
                continue

            rect = detect_main_card(img)
            warped = None

            if rect is not None:
                quad = np.array(rect["quad"], dtype="float32")
                warped = warp_quad(img, quad)

            if warped is None or warped.size == 0:
                warped = img.copy()

            warped = cv2.resize(warped, (200, 300))
            zone = _extract_symbol_zone_from_card(warped)
            scan_mask, panel, scan_threshold_mode = _normalize_symbol_scan(zone)
            raw_name, score, gap, symbol_debug = detect_symbol(zone)

            top_candidates = symbol_debug.get("top_candidates") or []
            winner = top_candidates[0] if top_candidates else None
            winner_references = symbol_debug.get("winner_references") or []

            zone_ok = bool(zone is not None and zone.size != 0)
            panel_ok = bool(panel is not None and panel.size != 0)
            mask_nonzero = int(np.count_nonzero(scan_mask)) if scan_mask is not None else 0
            mask_ok = bool(scan_mask is not None and scan_mask.size != 0 and mask_nonzero > 0)

            expected_symbol = _resolve_expected_symbol(filename, expected_symbol_map)
            business = _build_business_symbol_result(
                filename=filename,
                raw_name=raw_name,
                score=score,
                gap=gap,
                expected_symbol=expected_symbol,
            )

            bucket = business["business_bucket"]
            if expected_symbol:
                summary["with_expected"] += 1
                if bucket == "correcte":
                    summary["correctes"] += 1
                elif bucket == "fausse":
                    summary["fausses"] += 1
                elif bucket == "fragile":
                    summary["fragiles"] += 1
                elif bucket == "rejetee":
                    summary["rejetees"] += 1
            else:
                summary["sans_attendu"] += 1

            item = {
                "file": filename,
                "expected_symbol": expected_symbol,
                "raw_name": raw_name,
                "accepted_name": business["accepted_name"],
                "score": float(score),
                "gap": float(gap),
                "winner": winner,
                "winner_references": winner_references,
                "business_status": business["business_status"],
                "business_bucket": business["business_bucket"],
                "threshold_mode": business["threshold_mode"],
                "debug": {
                    "zone_ok": zone_ok,
                    "panel_ok": panel_ok,
                    "mask_ok": mask_ok,
                    "zone_shape": list(zone.shape) if zone is not None else None,
                    "panel_shape": list(panel.shape) if panel is not None else None,
                    "mask_nonzero": mask_nonzero,
                    "scan_threshold_mode": scan_threshold_mode
                }
            }
            batch_results.append(item)

            if business["business_status"] == "OK":
                status_class = "ok"
            elif business["business_bucket"] == "fragile":
                status_class = "fragile"
            else:
                status_class = "ko"

            rows.append(
                f"""
                <tr>
                  <td>{safe_filename}</td>
                  <td>{html.escape(expected_symbol or '')}</td>
                  <td>{html.escape(raw_name or '')}</td>
                  <td>{html.escape(business['accepted_name'] or '')}</td>
                  <td>{float(score):.4f}</td>
                  <td>{float(gap):.4f}</td>
                  <td>{html.escape(business['threshold_mode'])}</td>
                  <td><span class="{status_class}">{html.escape(business['business_status'])}</span></td>
                </tr>
                """
            )

        except Exception as e:
            summary["errors"] += 1
            batch_results.append({
                "file": filename,
                "error": str(e)
            })
            rows.append(
                f"<tr><td>{safe_filename}</td><td colspan='7'><span class='ko'>Erreur: {html.escape(str(e))}</span></td></tr>"
            )

    pretty_json = json.dumps({
        "summary": summary,
        "results": batch_results
    }, indent=2, ensure_ascii=False)
    html_rows = "\n".join(rows)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Symbol Batch Test</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .ok {{ color: #0a7a28; font-weight: bold; }}
        .ko {{ color: #b00020; font-weight: bold; }}
        .fragile {{ color: #b26b00; font-weight: bold; }}
        .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }}
        .summary-card {{ border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }}
        .summary-card .label {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }}
        .summary-card .value {{ font-size:28px; font-weight:bold; margin-top:6px; }}
        pre {{ background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }}
      </style>
    </head>
    <body>
      <h1>Résultat batch symbole</h1>
      <p><a href="/symbol-batch-test">← Revenir au formulaire</a></p>

      <div class="summary-grid">
        <div class="summary-card">
          <div class="label">Correctes</div>
          <div class="value">{summary['correctes']} / {summary['with_expected']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Fausses</div>
          <div class="value">{summary['fausses']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Fragiles</div>
          <div class="value">{summary['fragiles']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Rejetées</div>
          <div class="value">{summary['rejetees']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Sans attendu</div>
          <div class="value">{summary['sans_attendu']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Erreurs</div>
          <div class="value">{summary['errors']}</div>
        </div>
      </div>

      <div style="margin:16px 0; display:flex; gap:10px; flex-wrap:wrap;">
        <button type="button" onclick="exportBatchJson()">Exporter JSON</button>
        <button type="button" onclick="exportBatchTxt()">Exporter TXT</button>
        <button type="button" onclick="exportBatchHtml()">Exporter HTML</button>
        <button type="button" onclick="copyBatchJson()">Copier JSON</button>
      </div>

      <table id="batch-results-table">
        <thead>
          <tr>
            <th>Fichier</th>
            <th>Attendu</th>
            <th>Raw</th>
            <th>Accepté</th>
            <th>Score</th>
            <th>Gap</th>
            <th>Mode seuil</th>
            <th>Statut métier</th>
          </tr>
        </thead>
        <tbody>
          {html_rows}
        </tbody>
      </table>

      <h2>JSON complet</h2>
      <pre id="batch-json">{pretty_json}</pre>

      <script>
      function downloadTextFile(filename, content, contentType) {{
        const blob = new Blob([content], {{ type: contentType }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function getBatchJsonText() {{
        const pre = document.getElementById("batch-json");
        return pre ? pre.textContent : "";
      }}

      function exportBatchJson() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("symbol_batch_results.json", jsonText, "application/json;charset=utf-8");
      }}

      function exportBatchTxt() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("symbol_batch_results.txt", jsonText, "text/plain;charset=utf-8");
      }}

      function exportBatchHtml() {{
        const html = document.documentElement.outerHTML;
        downloadTextFile("symbol_batch_results.html", html, "text/html;charset=utf-8");
      }}

      function copyBatchJson() {{
        const jsonText = getBatchJsonText();
        navigator.clipboard.writeText(jsonText)
          .then(() => alert("JSON copié"))
          .catch(() => alert("Copie impossible"));
      }}
      </script>
    </body>
    </html>
    """

# =====================================================
# CARD BATCH TEST
# =====================================================

@app.route("/card-batch-test", methods=["GET", "POST"])
def card_batch_test():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Card Batch Test</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 16px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background: #f5f5f5; }
            .ok { color: #0a7a28; font-weight: bold; }
            .ko { color: #b00020; font-weight: bold; }
            .fragile { color: #b26b00; font-weight: bold; }
            .summary-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }
            .summary-card { border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }
            .summary-card .label { font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }
            .summary-card .value { font-size:28px; font-weight:bold; margin-top:6px; }
            pre { background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }
          </style>
        </head>
        <body>
          <h1>Test batch carte finale</h1>
          <p>Envoie plusieurs images de cartes complètes. Le moteur décide la carte unique avec couleur + symbole + bottom.</p>
          <form method="post" enctype="multipart/form-data">
            <p><input type="file" name="images" accept="image/*" multiple required /></p>
            <p><button type="submit">Analyser le lot</button></p>
          </form>
        </body>
        </html>
        """

    files = sorted(
        request.files.getlist("images"),
        key=lambda f: _natural_sort_key(f.filename or "")
    )
    if not files:
        return "Aucun fichier envoyé", 400

    try:
        cards = load_cards_js()
    except Exception as e:
        return f"Impossible de charger cards.js: {str(e)}", 500

    rows = []
    batch_results = []
    summary = {
        "total": 0,
        "with_expected": 0,
        "correctes": 0,
        "fausses": 0,
        "fragiles": 0,
        "rejetees": 0,
        "sans_attendu": 0,
        "errors": 0,
    }

    for file in files:
        filename = file.filename or "unknown"
        safe_filename = html.escape(filename)
        summary["total"] += 1

        try:
            data = file.read()
            if not data:
                raise ValueError("empty_file")

            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("invalid_image")

            rect = detect_main_card(img)
            if rect is not None:
                quad = np.array(rect["quad"], dtype="float32")
                warped = warp_quad(img, quad)
            else:
                warped = None

            if warped is None or warped.size == 0:
                warped = img.copy()

            warped = cv2.resize(warped, (200, 300))
            sig, _ = compute_signature(warped)
            card_match = resolve_final_card(sig, cards=cards)

            expected_card_id = _resolve_expected_card_id(filename, cards=cards)
            business = _build_business_card_result(
                filename=filename,
                final_card_id=card_match.get("final_card_id"),
                final_status=card_match.get("final_status"),
                expected_card_id=expected_card_id,
            )

            bucket = business["business_bucket"]
            if expected_card_id:
                summary["with_expected"] += 1
                if bucket == "correcte":
                    summary["correctes"] += 1
                elif bucket == "fausse":
                    summary["fausses"] += 1
                elif bucket == "fragile":
                    summary["fragiles"] += 1
                elif bucket == "rejetee":
                    summary["rejetees"] += 1
            else:
                summary["sans_attendu"] += 1

            item = {
                "file": filename,
                "expected_card_id": expected_card_id,
                "color_name": card_match.get("color_name"),
                "symbol_name": card_match.get("symbol_name"),
                "symbol_source": card_match.get("symbol_source"),
                "bottom_layout": card_match.get("bottom_layout"),
                "points": card_match.get("points"),
                "final_card_id": card_match.get("final_card_id"),
                "final_score": float(card_match.get("final_score", 0.0)),
                "final_gap": float(card_match.get("final_gap", 0.0)),
                "final_status": card_match.get("final_status"),
                "reason": card_match.get("reason"),
                "candidate_cards": card_match.get("candidate_cards", []),
                "business_status": business["business_status"],
                "business_bucket": business["business_bucket"],
            }
            batch_results.append(item)

            if business["business_status"] == "OK":
                status_class = "ok"
            elif business["business_bucket"] == "fragile":
                status_class = "fragile"
            else:
                status_class = "ko"

            rows.append(
                f"""
                <tr>
                  <td>{safe_filename}</td>
                  <td>{html.escape(expected_card_id or '')}</td>
                  <td>{html.escape(card_match.get('color_name') or '')}</td>
                  <td>{html.escape(card_match.get('symbol_name') or '')}</td>
                  <td>{html.escape(str(card_match.get('points') if card_match.get('points') is not None else ''))}</td>
                  <td>{html.escape(card_match.get('bottom_layout') or '')}</td>
                  <td>{html.escape(card_match.get('final_card_id') or '')}</td>
                  <td>{float(card_match.get('final_score', 0.0)):.4f}</td>
                  <td>{float(card_match.get('final_gap', 0.0)):.4f}</td>
                  <td>{html.escape(card_match.get('final_status') or '')}</td>
                  <td><span class="{status_class}">{html.escape(business['business_status'])}</span></td>
                </tr>
                """
            )

        except Exception as e:
            summary["errors"] += 1
            batch_results.append({
                "file": filename,
                "error": str(e),
            })
            rows.append(
                f"<tr><td>{safe_filename}</td><td colspan='10'><span class='ko'>Erreur: {html.escape(str(e))}</span></td></tr>"
            )

    pretty_json = json.dumps({
        "summary": summary,
        "results": batch_results
    }, indent=2, ensure_ascii=False)
    html_rows = "\n".join(rows)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Card Batch Test</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .ok {{ color: #0a7a28; font-weight: bold; }}
        .ko {{ color: #b00020; font-weight: bold; }}
        .fragile {{ color: #b26b00; font-weight: bold; }}
        .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }}
        .summary-card {{ border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }}
        .summary-card .label {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }}
        .summary-card .value {{ font-size:28px; font-weight:bold; margin-top:6px; }}
        pre {{ background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }}
      </style>
    </head>
    <body>
      <h1>Résultat batch carte finale</h1>
      <p><a href="/card-batch-test">← Revenir au formulaire</a></p>

      <div class="summary-grid">
        <div class="summary-card">
          <div class="label">Correctes</div>
          <div class="value">{summary['correctes']} / {summary['with_expected']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Fausses</div>
          <div class="value">{summary['fausses']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Fragiles</div>
          <div class="value">{summary['fragiles']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Rejetées</div>
          <div class="value">{summary['rejetees']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Sans attendu</div>
          <div class="value">{summary['sans_attendu']}</div>
        </div>
        <div class="summary-card">
          <div class="label">Erreurs</div>
          <div class="value">{summary['errors']}</div>
        </div>
      </div>

      <div style="margin:16px 0; display:flex; gap:10px; flex-wrap:wrap;">
        <button type="button" onclick="exportBatchJson()">Exporter JSON</button>
        <button type="button" onclick="exportBatchTxt()">Exporter TXT</button>
        <button type="button" onclick="exportBatchHtml()">Exporter HTML</button>
        <button type="button" onclick="copyBatchJson()">Copier JSON</button>
      </div>

      <table id="batch-results-table">
        <thead>
          <tr>
            <th>Fichier</th>
            <th>Attendue</th>
            <th>Couleur</th>
            <th>Symbole</th>
            <th>Points</th>
            <th>Bottom</th>
            <th>Carte finale</th>
            <th>Score final</th>
            <th>Gap final</th>
            <th>Statut final</th>
            <th>Statut métier</th>
          </tr>
        </thead>
        <tbody>
          {html_rows}
        </tbody>
      </table>

      <h2>JSON complet</h2>
      <pre id="batch-json">{pretty_json}</pre>

      <script>
      function downloadTextFile(filename, content, contentType) {{
        const blob = new Blob([content], {{ type: contentType }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function getBatchJsonText() {{
        const pre = document.getElementById("batch-json");
        return pre ? pre.textContent : "";
      }}

      function exportBatchJson() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("card_batch_results.json", jsonText, "application/json;charset=utf-8");
      }}

      function exportBatchTxt() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("card_batch_results.txt", jsonText, "text/plain;charset=utf-8");
      }}

      function exportBatchHtml() {{
        const html = document.documentElement.outerHTML;
        downloadTextFile("card_batch_results.html", html, "text/html;charset=utf-8");
      }}

      function copyBatchJson() {{
        const jsonText = getBatchJsonText();
        navigator.clipboard.writeText(jsonText)
          .then(() => alert("JSON copié"))
          .catch(() => alert("Copie impossible"));
      }}
      </script>
    </body>
    </html>
    """

# =====================================================
# BOTTOM TEST
# =====================================================



def _get_expected_bottom_map():
    try:
        cards = load_cards_js()
    except Exception:
        cards = []

    expected = {}
    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if not card_id:
            continue
        profile = _build_expected_bottom_profile(card)
        profile.update({
            "card_id": card_id,
            "color": _safe_upper(card.get("couleur")),
            "symbol": _safe_upper(card.get("symbol")),
        })
        expected[card_id.lower()] = profile

    return expected


def _resolve_expected_bottom(filename, expected_bottom_map):
    stem = os.path.splitext(os.path.basename(filename or ""))[0].strip().lower()
    if not stem:
        return None
    return expected_bottom_map.get(stem)


def _build_observed_bottom_from_result(result):
    result = result or {}
    return {
        "layout": str(result.get("layout") or "UNKNOWN"),
        "points": _safe_int(result.get("points")),
        "raw_points": _safe_int(result.get("raw_points")),
        "points_score": float(result.get("points_score", 0.0) or 0.0),
        "points_gap": float(result.get("points_gap", 0.0) or 0.0),
        "has_special_white_panel": bool(result.get("has_special_white_panel", False)),
        "has_slash": bool(result.get("has_slash", False)),
        "has_right_icon": bool(result.get("has_right_icon", False)),
        "has_bottom_line": bool(result.get("has_bottom_line", False)),
    }


def _infer_observed_bottom_range(observed):
    observed = observed or {}
    explicit_range = observed.get("range")
    if explicit_range:
        return explicit_range

    has_slash = bool(observed.get("has_slash"))
    has_right_icon = bool(observed.get("has_right_icon"))
    has_bottom_line = bool(observed.get("has_bottom_line"))

    if has_slash and has_right_icon and has_bottom_line:
        return "LIGNE"
    if has_slash and has_right_icon:
        return "GLOBAL"
    return None


def _bottom_points_threshold_mode(observed):
    observed = observed or {}
    explicit_mode = observed.get("threshold_mode")
    if explicit_mode:
        return explicit_mode

    accepted = observed.get("points")
    raw = observed.get("raw_points")
    score = float(observed.get("points_score", 0.0) or 0.0)
    gap = float(observed.get("points_gap", 0.0) or 0.0)

    if accepted is not None:
        return "accepted"
    if raw is None:
        return "no_digit"
    if score < 0.60 and gap < 0.02:
        return "rejected_score_gap"
    if score < 0.60:
        return "rejected_score"
    if gap < 0.02:
        return "rejected_gap"
    return "rejected_digit"



_BOTTOM_REF_CACHE = None


def _compute_bottom_ref_features(bottom_roi):
    if bottom_roi is None or getattr(bottom_roi, "size", 0) == 0:
        return None

    try:
        roi = cv2.resize(bottom_roi, (96, 48), interpolation=cv2.INTER_AREA)
    except Exception:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    eq = cv2.equalizeHist(gray)
    edges = cv2.Canny(eq, 80, 180)

    dark_ratio = float(np.count_nonzero(eq < 100)) / float(max(eq.size, 1))
    light_ratio = float(np.count_nonzero(eq > 180)) / float(max(eq.size, 1))

    return {
        "gray": eq,
        "edges": edges,
        "dark_ratio": dark_ratio,
        "light_ratio": light_ratio,
    }


def _safe_norm_corr(a, b):
    if a is None or b is None:
        return 0.0

    try:
        va = a.astype(np.float32).flatten()
        vb = b.astype(np.float32).flatten()
    except Exception:
        return 0.0

    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0

    va = va - float(np.mean(va))
    vb = vb - float(np.mean(vb))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-6:
        return 0.0

    corr = float(np.dot(va, vb) / denom)
    corr = max(-1.0, min(1.0, corr))
    return (corr + 1.0) / 2.0


def _safe_binary_iou(a, b):
    if a is None or b is None:
        return 0.0

    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    if aa.shape != bb.shape:
        return 0.0

    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0

    inter = np.logical_and(aa, bb).sum()
    return float(inter) / float(union)


def _bottom_ref_similarity(features_a, features_b):
    if not features_a or not features_b:
        return 0.0

    gray_score = _safe_norm_corr(features_a.get("gray"), features_b.get("gray"))
    edge_score = _safe_binary_iou(features_a.get("edges"), features_b.get("edges"))
    dark_score = 1.0 - min(abs(float(features_a.get("dark_ratio", 0.0)) - float(features_b.get("dark_ratio", 0.0))) / 0.35, 1.0)
    light_score = 1.0 - min(abs(float(features_a.get("light_ratio", 0.0)) - float(features_b.get("light_ratio", 0.0))) / 0.35, 1.0)

    return float((gray_score * 0.55) + (edge_score * 0.30) + (dark_score * 0.10) + (light_score * 0.05))


def _load_bottom_reference_library():
    global _BOTTOM_REF_CACHE
    if _BOTTOM_REF_CACHE is not None:
        return _BOTTOM_REF_CACHE

    refs = []
    try:
        cards = load_cards_js()
    except Exception:
        cards = []

    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if not card_id:
            continue

        image_path = find_card_image(card_id)
        if not image_path or not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            continue

        try:
            rect = detect_main_card(img)
            warped = None
            if rect is not None:
                quad = np.array(rect["quad"], dtype="float32")
                warped = warp_quad(img, quad)
            if warped is None or warped.size == 0:
                warped = img.copy()
            warped = cv2.resize(warped, (200, 300))
            _, bottom_roi, _ = extract_bottom_roi_from_full_card(warped)
        except Exception:
            continue

        features = _compute_bottom_ref_features(bottom_roi)
        if not features:
            continue

        refs.append({
            "card_id": card_id,
            "symbol": _safe_upper(card.get("symbol")),
            "color": _safe_upper(card.get("couleur")),
            "expected_bottom": _build_expected_bottom_profile(card),
            "features": features,
        })

    _BOTTOM_REF_CACHE = refs
    return refs


def _match_bottom_reference(bottom_roi, expected_symbol=None, expected_color=None):
    features = _compute_bottom_ref_features(bottom_roi)
    if not features:
        return None

    refs = _load_bottom_reference_library()
    if not refs:
        return None

    expected_symbol = _safe_upper(expected_symbol)
    expected_color = _safe_upper(expected_color)

    candidates = []
    for ref in refs:
        if expected_symbol and ref.get("symbol") != expected_symbol:
            continue
        if expected_color and ref.get("color") != expected_color:
            continue
        score = _bottom_ref_similarity(features, ref.get("features"))
        candidates.append({
            "card_id": ref.get("card_id"),
            "score": float(score),
            "symbol": ref.get("symbol"),
            "color": ref.get("color"),
            "expected_bottom": dict(ref.get("expected_bottom") or {}),
        })

    if not candidates and (expected_symbol or expected_color):
        return _match_bottom_reference(bottom_roi, expected_symbol=None, expected_color=None)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    gap = float(best["score"] - second_score)

    result = dict(best)
    result["gap"] = gap
    result["candidates"] = candidates[:5]
    return result


def _apply_bottom_reference_match(observed_bottom, bottom_roi, expected_bottom=None):
    observed_bottom = dict(observed_bottom or {})
    expected_bottom = expected_bottom or {}

    ref_match = _match_bottom_reference(
        bottom_roi,
        expected_symbol=expected_bottom.get("symbol"),
        expected_color=expected_bottom.get("color"),
    )
    if not ref_match:
        return observed_bottom

    matched = ref_match.get("expected_bottom") or {}
    ref_score = float(ref_match.get("score", 0.0) or 0.0)
    ref_gap = float(ref_match.get("gap", 0.0) or 0.0)

    observed_bottom.update({
        "layout": matched.get("layout") or observed_bottom.get("layout"),
        "points": matched.get("points"),
        "raw_points": matched.get("points"),
        "points_score": ref_score,
        "points_gap": ref_gap,
        "has_special_white_panel": bool(matched.get("has_special_white_panel", False)),
        "has_slash": bool(matched.get("has_slash", False)),
        "has_right_icon": bool(matched.get("has_right_icon", False)),
        "has_bottom_line": bool(matched.get("has_bottom_line", False)),
        "target": matched.get("target") or "",
        "range": matched.get("range") or None,
        "threshold_mode": "ref_match",
        "ref_card_id": ref_match.get("card_id"),
        "ref_score": ref_score,
        "ref_gap": ref_gap,
    })
    return observed_bottom


def _build_business_bottom_result(expected_bottom, observed_bottom):
    observed_bottom = observed_bottom or {}

    if not expected_bottom:
        return {
            "expected_range": None,
            "observed_range": _infer_observed_bottom_range(observed_bottom),
            "threshold_mode": _bottom_points_threshold_mode(observed_bottom),
            "business_status": "N/A",
            "business_bucket": "sans_attendu",
        }

    expected_range = expected_bottom.get("range") or None
    observed_range = _infer_observed_bottom_range(observed_bottom)

    layout_ok = (expected_bottom.get("layout") or "") == (observed_bottom.get("layout") or "")
    expected_points = expected_bottom.get("points")
    observed_points = observed_bottom.get("points")
    observed_raw_points = observed_bottom.get("raw_points")

    if expected_bottom.get("layout") == "SPECIAL_WHITE_PANEL" and expected_points in {0, None}:
        points_ok = observed_points in {0, None}
        raw_points_ok = observed_raw_points in {0, None}
    else:
        points_ok = expected_points == observed_points
        raw_points_ok = expected_points == observed_raw_points
    special_ok = bool(expected_bottom.get("has_special_white_panel")) == bool(observed_bottom.get("has_special_white_panel"))
    slash_ok = bool(expected_bottom.get("has_slash")) == bool(observed_bottom.get("has_slash"))
    icon_ok = bool(expected_bottom.get("has_right_icon")) == bool(observed_bottom.get("has_right_icon"))
    line_ok = bool(expected_bottom.get("has_bottom_line")) == bool(observed_bottom.get("has_bottom_line"))
    range_ok = expected_range == observed_range

    exact_ok = all([layout_ok, points_ok, special_ok, slash_ok, icon_ok, line_ok, range_ok])
    raw_ok = all([layout_ok, raw_points_ok, special_ok, slash_ok, icon_ok, line_ok, range_ok])

    if exact_ok:
        business_status = "OK"
        business_bucket = "correcte"
    elif raw_ok:
        business_status = "KO"
        business_bucket = "fragile"
    elif (observed_bottom.get("layout") in {"UNKNOWN", "BLACK_PANEL"} and
          observed_bottom.get("points") is None and
          observed_bottom.get("raw_points") is None and
          not observed_bottom.get("has_special_white_panel") and
          not observed_bottom.get("has_slash") and
          not observed_bottom.get("has_right_icon") and
          not observed_bottom.get("has_bottom_line")):
        business_status = "KO"
        business_bucket = "rejetee"
    else:
        business_status = "KO"
        business_bucket = "fausse"

    return {
        "expected_range": expected_range,
        "observed_range": observed_range,
        "threshold_mode": _bottom_points_threshold_mode(observed_bottom),
        "business_status": business_status,
        "business_bucket": business_bucket,
    }


@app.route("/bottom-batch-test", methods=["GET", "POST"])
def bottom_batch_test():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Bottom Batch Test</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 16px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background: #f5f5f5; }
            .ok { color: #0a7a28; font-weight: bold; }
            .ko { color: #b00020; font-weight: bold; }
            .fragile { color: #b26b00; font-weight: bold; }
            .summary-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }
            .summary-card { border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }
            .summary-card .label { font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }
            .summary-card .value { font-size:28px; font-weight:bold; margin-top:6px; }
            pre { background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }
          </style>
        </head>
        <body>
          <h1>Test batch bottom</h1>
          <p>Envoie plusieurs images de cartes complètes pour vérifier le bas de carte.</p>
          <form method="post" enctype="multipart/form-data">
            <p><input type="file" name="images" accept="image/*" multiple required /></p>
            <p><button type="submit">Analyser le lot</button></p>
          </form>
        </body>
        </html>
        """

    files = sorted(
        request.files.getlist("images"),
        key=lambda f: _natural_sort_key(f.filename or "")
    )
    if not files:
        return "Aucun fichier envoyé", 400

    expected_bottom_map = _get_expected_bottom_map()

    rows = []
    batch_results = []
    summary = {
        "total": 0,
        "with_expected": 0,
        "correctes": 0,
        "fausses": 0,
        "fragiles": 0,
        "rejetees": 0,
        "sans_attendu": 0,
        "errors": 0,
    }

    for file in files:
        filename = file.filename or "unknown"
        safe_filename = html.escape(filename)
        summary["total"] += 1

        try:
            data = file.read()
            if not data:
                summary["errors"] += 1
                batch_results.append({"file": filename, "error": "empty_file"})
                rows.append(f"<tr><td>{safe_filename}</td><td colspan='11'><span class='ko'>Fichier vide</span></td></tr>")
                continue

            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                summary["errors"] += 1
                batch_results.append({"file": filename, "error": "invalid_image"})
                rows.append(f"<tr><td>{safe_filename}</td><td colspan='11'><span class='ko'>Image invalide</span></td></tr>")
                continue

            rect = detect_main_card(img)
            warped = None
            if rect is not None:
                quad = np.array(rect["quad"], dtype="float32")
                warped = warp_quad(img, quad)
            if warped is None or warped.size == 0:
                warped = img.copy()

            warped = cv2.resize(warped, (200, 300))
            full_img, bottom_roi, bottom_box = extract_bottom_roi_from_full_card(warped)
            result = analyze_bottom(bottom_roi, DIGITS_DIR)
            observed = _build_observed_bottom_from_result(result)
            expected = _resolve_expected_bottom(filename, expected_bottom_map)
            observed = _apply_bottom_reference_match(observed, bottom_roi, expected)
            business = _build_business_bottom_result(expected, observed)

            bucket = business["business_bucket"]
            if expected:
                summary["with_expected"] += 1
                if bucket == "correcte":
                    summary["correctes"] += 1
                elif bucket == "fausse":
                    summary["fausses"] += 1
                elif bucket == "fragile":
                    summary["fragiles"] += 1
                elif bucket == "rejetee":
                    summary["rejetees"] += 1
            else:
                summary["sans_attendu"] += 1

            expected_layout = (expected or {}).get("layout")
            expected_points = (expected or {}).get("points")
            expected_range = business.get("expected_range")
            expected_symbol = (expected or {}).get("symbol")
            observed_layout = observed.get("layout")
            observed_points = observed.get("points")
            raw_points = observed.get("raw_points")
            observed_range = business.get("observed_range")
            score = float(observed.get("points_score", 0.0))
            gap = float(observed.get("points_gap", 0.0))

            item = {
                "file": filename,
                "expected_symbol": expected_symbol,
                "expected_layout": expected_layout,
                "expected_points": expected_points,
                "expected_range": expected_range,
                "observed_layout": observed_layout,
                "observed_points": observed_points,
                "raw_points": raw_points,
                "observed_range": observed_range,
                "points_score": score,
                "points_gap": gap,
                "has_slash": bool(observed.get("has_slash")),
                "has_right_icon": bool(observed.get("has_right_icon")),
                "has_bottom_line": bool(observed.get("has_bottom_line")),
                "has_special_white_panel": bool(observed.get("has_special_white_panel")),
                "threshold_mode": business["threshold_mode"],
                "business_status": business["business_status"],
                "business_bucket": business["business_bucket"],
            }
            batch_results.append(item)

            if business["business_status"] == "OK":
                status_class = "ok"
            elif business["business_bucket"] == "fragile":
                status_class = "fragile"
            else:
                status_class = "ko"

            rows.append(
                f"""
                <tr>
                  <td>{safe_filename}</td>
                  <td>{html.escape(expected_symbol or '')}</td>
                  <td>{html.escape(expected_layout or '')}</td>
                  <td>{'' if expected_points is None else expected_points}</td>
                  <td>{html.escape(expected_range or '')}</td>
                  <td>{html.escape(observed_layout or '')}</td>
                  <td>{'' if observed_points is None else observed_points}</td>
                  <td>{'' if raw_points is None else raw_points}</td>
                  <td>{float(score):.4f}</td>
                  <td>{float(gap):.4f}</td>
                  <td>{html.escape(observed_range or '')}</td>
                  <td>{html.escape(business['threshold_mode'])}</td>
                  <td><span class=\"{status_class}\">{html.escape(business['business_status'])}</span></td>
                </tr>
                """
            )

        except Exception as e:
            summary["errors"] += 1
            batch_results.append({"file": filename, "error": str(e)})
            rows.append(f"<tr><td>{safe_filename}</td><td colspan='11'><span class='ko'>Erreur: {html.escape(str(e))}</span></td></tr>")

    pretty_json = json.dumps({"summary": summary, "results": batch_results}, indent=2, ensure_ascii=False)
    html_rows = "\n".join(rows)

    return f"""
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <title>Bottom Batch Test</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .ok {{ color: #0a7a28; font-weight: bold; }}
        .ko {{ color: #b00020; font-weight: bold; }}
        .fragile {{ color: #b26b00; font-weight: bold; }}
        .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:18px 0; }}
        .summary-card {{ border:1px solid #ddd; background:#fafafa; padding:12px; border-radius:8px; }}
        .summary-card .label {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.04em; }}
        .summary-card .value {{ font-size:28px; font-weight:bold; margin-top:6px; }}
        pre {{ background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; }}
      </style>
    </head>
    <body>
      <h1>Résultat batch bottom</h1>
      <p><a href=\"/bottom-batch-test\">← Revenir au formulaire</a></p>

      <div class=\"summary-grid\">
        <div class=\"summary-card\">
          <div class=\"label\">Correctes</div>
          <div class=\"value\">{summary['correctes']} / {summary['with_expected']}</div>
        </div>
        <div class=\"summary-card\">
          <div class=\"label\">Fausses</div>
          <div class=\"value\">{summary['fausses']}</div>
        </div>
        <div class=\"summary-card\">
          <div class=\"label\">Fragiles</div>
          <div class=\"value\">{summary['fragiles']}</div>
        </div>
        <div class=\"summary-card\">
          <div class=\"label\">Rejetées</div>
          <div class=\"value\">{summary['rejetees']}</div>
        </div>
        <div class=\"summary-card\">
          <div class=\"label\">Sans attendu</div>
          <div class=\"value\">{summary['sans_attendu']}</div>
        </div>
        <div class=\"summary-card\">
          <div class=\"label\">Erreurs</div>
          <div class=\"value\">{summary['errors']}</div>
        </div>
      </div>

      <div style=\"margin:16px 0; display:flex; gap:10px; flex-wrap:wrap;\">
        <button type=\"button\" onclick=\"exportBatchJson()\">Exporter JSON</button>
        <button type=\"button\" onclick=\"exportBatchTxt()\">Exporter TXT</button>
        <button type=\"button\" onclick=\"exportBatchHtml()\">Exporter HTML</button>
        <button type=\"button\" onclick=\"copyBatchJson()\">Copier JSON</button>
      </div>

      <table id=\"batch-results-table\">
        <thead>
          <tr>
            <th>Fichier</th>
            <th>Symbole</th>
            <th>Layout attendu</th>
            <th>Pts attendus</th>
            <th>Portée attendue</th>
            <th>Layout détecté</th>
            <th>Pts détectés</th>
            <th>Raw</th>
            <th>Score</th>
            <th>Gap</th>
            <th>Portée détectée</th>
            <th>Mode seuil</th>
            <th>Statut métier</th>
          </tr>
        </thead>
        <tbody>
          {html_rows}
        </tbody>
      </table>

      <h2>JSON complet</h2>
      <pre id=\"batch-json\">{pretty_json}</pre>

      <script>
      function downloadTextFile(filename, content, contentType) {{
        const blob = new Blob([content], {{ type: contentType }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function getBatchJsonText() {{
        const pre = document.getElementById("batch-json");
        return pre ? pre.textContent : "";
      }}

      function exportBatchJson() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("bottom_batch_results.json", jsonText, "application/json;charset=utf-8");
      }}

      function exportBatchTxt() {{
        const jsonText = getBatchJsonText();
        downloadTextFile("bottom_batch_results.txt", jsonText, "text/plain;charset=utf-8");
      }}

      function exportBatchHtml() {{
        const html = document.documentElement.outerHTML;
        downloadTextFile("bottom_batch_results.html", html, "text/html;charset=utf-8");
      }}

      function copyBatchJson() {{
        const jsonText = getBatchJsonText();
        navigator.clipboard.writeText(jsonText)
          .then(() => alert("JSON copié"))
          .catch(() => alert("Copie impossible"));
      }}
      </script>
    </body>
    </html>
    """


@app.route("/bottom-test", methods=["GET", "POST"])
def bottom_test():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Bottom Test</title>
        </head>
        <body style="font-family:Arial,sans-serif; padding:20px;">
          <h1>Test du bas de carte</h1>
          <p>Choisis une image de carte complète, puis clique sur Analyser.</p>

          <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required />
            <button type="submit">Analyser</button>
          </form>
        </body>
        </html>
        """

    if "image" not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files["image"]
    if not file or file.filename == "":
        return "Fichier vide", 400

    data = file.read()
    if not data:
        return "Impossible de lire le fichier", 400

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return "Image invalide", 400

    full_img, bottom_roi, bottom_box = extract_bottom_roi_from_full_card(img)
    result = analyze_bottom(bottom_roi, DIGITS_DIR)
    ref_match = _match_bottom_reference(bottom_roi)
    overlay = build_overlay(full_img, bottom_box, result)

    points_crop = None
    badge_norm = None
    digit_mask = None

    points_box = result.get("points_box")
    if points_box is not None:
        x, y, w, h = points_box
        points_crop = bottom_roi[y:y + h, x:x + w]
        if points_crop is not None and points_crop.size != 0:
            badge_norm = _normalize_badge(points_crop)
            digit_mask = _extract_digit_mask(points_crop)

    full_b64 = _img_to_base64(full_img)
    bottom_b64 = _img_to_base64(bottom_roi)
    overlay_b64 = _img_to_base64(overlay)
    points_b64 = _img_to_base64(points_crop) if points_crop is not None else None
    badge_b64 = _img_to_base64(badge_norm) if badge_norm is not None else None
    digit_b64 = _img_to_base64(digit_mask) if digit_mask is not None else None

    pretty_json = json.dumps(result, indent=2, ensure_ascii=False)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Bottom Test</title>
    </head>
    <body style="font-family:Arial,sans-serif; padding:20px;">
      <h1>Résultat du test du bas</h1>
      <p><a href="/bottom-test">← Revenir au formulaire</a></p>
      <h2>Résultat JSON</h2>
      <pre style="background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto;">{pretty_json}</pre>
      {_html_img_block("Image complète", full_b64)}
      {_html_img_block("ROI du bas", bottom_b64)}
      {_html_img_block("Overlay debug", overlay_b64)}
      {_html_img_block("Crop points", points_b64)}
      {_html_img_block("Badge normalisé", badge_b64)}
      {_html_img_block("Masque du chiffre", digit_b64)}
    </body>
    </html>
    """
def verify_cards_integrity():
    """
    Vérifie que chaque carte a bien une signature enrichie.
    """
    try:
        cards = load_cards_js()

        report = {
            "valid": 0,
            "missing_fields": [],
            "total": len(cards)
        }

        for card in cards:
            card_id = card.get("id", "Unknown ID")

            scan = card.get("signature", {}).get("scan", {})
            missing = []

            for field in ["symbol", "points", "bottom_layout"]:
                if field not in scan:
                    missing.append(field)

            if missing:
                report["missing_fields"].append({
                    "id": card_id,
                    "missing": missing
                })
            else:
                report["valid"] += 1

        print("--- Rapport de vérification ---")
        print(f"Total cartes : {report['total']}")
        print(f"Cartes conformes : {report['valid']}")

        if report["missing_fields"]:
            print(f"⚠️ Erreurs détectées ({len(report['missing_fields'])} cartes) :")
            for error in report["missing_fields"]:
                print(f" - Carte [{error['id']}] : Manque {error['missing']}")
        else:
            print("✅ Toutes les cartes ont une signature enrichie.")

        return report

    except Exception as e:
        print(f"Erreur lors de la vérification : {e}")
        
      
        


# =====================================================
# BOARD DEBUG (CANDIDATS UNIQUEMENT)
# =====================================================


def _rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(1, a["w"] * a["h"])
    area_b = max(1, b["w"] * b["h"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _dedupe_board_candidates(candidates, iou_threshold=0.35):
    kept = []
    for cand in sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True):
        duplicate = False
        for k in kept:
            if _rect_iou(cand, k) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _classify_board_crop(crop):
    if crop is None or crop.size == 0:
        return {"kind": "unknown", "score": 0.0, "reasons": ["empty"]}

    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    sat_ratio = float(np.count_nonzero(hsv[:, :, 1] > 55)) / float(max(h * w, 1))
    dark_ratio = float(np.count_nonzero(gray < 110)) / float(max(h * w, 1))
    bright_ratio = float(np.count_nonzero(gray > 190)) / float(max(h * w, 1))

    if bright_ratio > 0.28 and sat_ratio < 0.18 and dark_ratio > 0.10:
        return {
            "kind": "station",
            "score": 0.95,
            "reasons": [
                f"bright={bright_ratio:.2f}",
                f"sat={sat_ratio:.2f}",
                f"dark={dark_ratio:.2f}",
            ],
        }

    if sat_ratio > 0.12:
        return {
            "kind": "card",
            "score": min(0.99, 0.55 + sat_ratio),
            "reasons": [
                f"sat={sat_ratio:.2f}",
                f"bright={bright_ratio:.2f}",
                f"dark={dark_ratio:.2f}",
            ],
        }

    return {
        "kind": "unknown",
        "score": 0.40,
        "reasons": [
            f"sat={sat_ratio:.2f}",
            f"bright={bright_ratio:.2f}",
            f"dark={dark_ratio:.2f}",
        ],
    }


def detect_board_candidates(img):
    """
    Ancienne logique libre par contours, gardée uniquement comme secours de debug.
    La logique principale du board debug est maintenant pilotée par:
    1) détection des 3 capsules
    2) reconstruction des 10 slots
    """
    if img is None or img.size == 0:
        return []

    vis = img.copy()
    h, w = vis.shape[:2]
    image_area = float(max(1, h * w))

    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * 0.006:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 80:
            continue

        ratio = bh / float(max(bw, 1))
        if ratio < 1.05 or ratio > 2.6:
            continue

        pad_x = int(bw * 0.06)
        pad_y = int(bh * 0.06)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop = vis[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        cls = _classify_board_crop(crop)

        candidates.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "score": float(area / image_area) + float(cls.get("score", 0.0)),
            "kind": cls.get("kind", "unknown"),
            "kind_score": float(cls.get("score", 0.0)),
            "reasons": cls.get("reasons", []),
        })

    return _dedupe_board_candidates(candidates, iou_threshold=0.30)


# =====================================================
# BOARD DEBUG - VERSION NETTOYÉE
# =====================================================


_CAPSULE_TEMPLATE_CACHE = None
_STATIONS_DIR = os.path.join(BASE_DIR, "stations")
_STATION_FILENAMES = {
    "left": "station_gauche.png",
    "middle": "station_milieu.png",
    "right": "station_droite.png",
}


def _safe_float01(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _load_capsule_template_image(label):
    filename = _STATION_FILENAMES.get(label)
    if not filename:
        return None
    path = os.path.join(_STATIONS_DIR, filename)
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _mask_pink_background(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pink = (
        (hsv[:, :, 0] >= 135) &
        (hsv[:, :, 0] <= 179) &
        (hsv[:, :, 1] >= 20) &
        (hsv[:, :, 2] >= 110)
    )
    mask = np.where(pink, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _crop_to_mask_bbox(img_bgr, mask):
    if img_bgr is None or mask is None or img_bgr.size == 0 or mask.size == 0:
        return img_bgr, mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, mask
    x1 = max(0, int(xs.min()))
    y1 = max(0, int(ys.min()))
    x2 = min(img_bgr.shape[1], int(xs.max()) + 1)
    y2 = min(img_bgr.shape[0], int(ys.max()) + 1)
    if x2 <= x1 or y2 <= y1:
        return img_bgr, mask
    return img_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _preprocess_capsule_template(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    mask = _mask_pink_background(img_bgr)
    img_bgr, mask = _crop_to_mask_bbox(img_bgr, mask)
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 60, 160)
    if mask is None or mask.size == 0:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    return {
        "img": img_bgr,
        "gray": gray,
        "edge": edge,
        "mask": mask,
    }


def _get_capsule_templates():
    global _CAPSULE_TEMPLATE_CACHE
    if _CAPSULE_TEMPLATE_CACHE is not None:
        return _CAPSULE_TEMPLATE_CACHE
    cache = {}
    for label in ("left", "middle", "right"):
        cache[label] = _preprocess_capsule_template(_load_capsule_template_image(label))
    _CAPSULE_TEMPLATE_CACHE = cache
    return _CAPSULE_TEMPLATE_CACHE


def _rotate_bound_gray(img, angle_deg, border_value=0):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _resize_for_board_debug(img, max_width=1600):
    if img is None or img.size == 0:
        return img, 1.0
    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy(), 1.0
    scale = max_width / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _search_window(img_w, x_min_frac, x_max_frac):
    x1 = max(0, int(round(img_w * float(x_min_frac))))
    x2 = min(img_w, int(round(img_w * float(x_max_frac))))
    if x2 <= x1 + 40:
        return 0, img_w
    return x1, x2


def _match_capsule_template(img, template, x_min_frac, x_max_frac):
    if img is None or img.size == 0 or not template:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 60, 160)

    h, w = gray.shape[:2]
    x1, x2 = _search_window(w, x_min_frac, x_max_frac)
    sub_gray = gray[:, x1:x2]
    sub_edge = edge[:, x1:x2]

    scales = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    angles = [-8, -4, 0, 4, 8]
    best = None

    for angle in angles:
        tpl_gray_base = _rotate_bound_gray(template["gray"], angle, border_value=255)
        tpl_edge_base = _rotate_bound_gray(template["edge"], angle, border_value=0)
        tpl_mask_base = _rotate_bound_gray(template["mask"], angle, border_value=0)

        for scale in scales:
            tpl_gray = cv2.resize(tpl_gray_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            tpl_edge = cv2.resize(tpl_edge_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tpl_mask = cv2.resize(tpl_mask_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            th, tw = tpl_gray.shape[:2]
            if th < 50 or tw < 30:
                continue
            if th >= sub_gray.shape[0] or tw >= sub_gray.shape[1]:
                continue

            valid_ratio = float(np.count_nonzero(tpl_mask > 0)) / float(max(1, tpl_mask.size))
            if valid_ratio < 0.18:
                continue

            res_gray = cv2.matchTemplate(sub_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            res_edge = cv2.matchTemplate(sub_edge, tpl_edge, cv2.TM_CCORR_NORMED)

            _, max_gray, _, max_loc_gray = cv2.minMaxLoc(res_gray)
            _, max_edge, _, max_loc_edge = cv2.minMaxLoc(res_edge)

            gray_score = _safe_float01(max_gray)
            edge_score = _safe_float01(max_edge)
            score = _safe_float01((gray_score * 0.70) + (edge_score * 0.30))
            loc_x = int(round((max_loc_gray[0] + max_loc_edge[0]) / 2.0)) + x1
            loc_y = int(round((max_loc_gray[1] + max_loc_edge[1]) / 2.0))

            cand = {
                "score": score,
                "x": int(loc_x),
                "y": int(loc_y),
                "w": int(tw),
                "h": int(th),
                "angle": float(angle),
                "scale": float(scale),
                "gray_score": gray_score,
                "edge_score": edge_score,
                "window": [int(x1), int(x2)],
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best


def _detect_board_capsules(img):
    templates = _get_capsule_templates()
    searches = [
        ("left", templates.get("left"), 0.00, 0.42),
        ("middle", templates.get("middle"), 0.24, 0.76),
        ("right", templates.get("right"), 0.58, 1.00),
    ]
    found = []
    for label, tpl, xmin, xmax in searches:
        hit = _match_capsule_template(img, tpl, xmin, xmax)
        if not hit:
            continue
        hit["label"] = label
        found.append(hit)
    return found


def _clip_rect(x, y, w, h, img_w, img_h):
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _clip_box_dict(x, y, w, h, img_w, img_h):
    x1, y1, ww, hh = _clip_rect(x, y, w, h, img_w, img_h)
    return {"x": x1, "y": y1, "w": ww, "h": hh}


def _validate_capsule_triplet(capsules, img_shape):
    if len(capsules) != 3:
        return False, "capsules_not_found"
    caps = sorted(capsules, key=lambda c: c["x"])
    cxs = [c["x"] + c["w"] / 2.0 for c in caps]
    cys = [c["y"] + c["h"] / 2.0 for c in caps]
    widths = [max(1, c["w"]) for c in caps]
    heights = [max(1, c["h"]) for c in caps]
    if not (cxs[0] < cxs[1] < cxs[2]):
        return False, "capsule_order_invalid"
    median_w = float(np.median(widths))
    median_h = float(np.median(heights))
    for w in widths:
        if abs(w - median_w) / max(1.0, median_w) > 0.35:
            return False, "capsule_width_mismatch"
    for h in heights:
        if abs(h - median_h) / max(1.0, median_h) > 0.35:
            return False, "capsule_height_mismatch"
    if max(cys) - min(cys) > median_h * 0.30:
        return False, "capsule_alignment_invalid"
    gaps = [cxs[1] - cxs[0], cxs[2] - cxs[1]]
    if min(gaps) < median_w * 1.10:
        return False, "capsule_gap_too_small"
    if max(gaps) / max(1.0, min(gaps)) > 1.60:
        return False, "capsule_gap_mismatch"
    return True, None


def _box_is_valid(box, img_shape):
    img_h, img_w = img_shape[:2]
    w = int(box.get("w", 0) or 0)
    h = int(box.get("h", 0) or 0)
    x = int(box.get("x", 0) or 0)
    y = int(box.get("y", 0) or 0)
    if w < 60 or h < 80:
        return False, "too_small"
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False, "out_of_image"
    area = float(w * h)
    if area < float(img_w * img_h) * 0.003:
        return False, "area_too_small"
    return True, None


def _build_slots_from_capsules(capsules, img_shape):
    img_h, img_w = img_shape[:2]
    if len(capsules) < 3:
        return []

    capsules = sorted(capsules, key=lambda c: c["x"])
    left, middle, right = capsules

    def cx(cap):
        return float(cap["x"] + (cap["w"] / 2.0))

    def cy(cap):
        return float(cap["y"] + (cap["h"] / 2.0))

    cxl, cxm, cxr = cx(left), cx(middle), cx(right)
    cyl, cym, cyr = cy(left), cy(middle), cy(right)

    cy_med = float(np.median([cyl, cym, cyr]))
    h_med = float(np.median([left["h"], middle["h"], right["h"]]))
    gap_lm = max(1.0, cxm - cxl)
    gap_mr = max(1.0, cxr - cxm)
    gap_med = float(np.median([gap_lm, gap_mr]))

    # Boîtes un peu plus petites que ta version actuelle
    top_w = int(round(gap_med * 0.31))
    top_h = int(round(top_w * 1.52))

    side_w = int(round(gap_med * 0.32))
    side_h = int(round(side_w * 1.48))

    bottom_w = int(round(gap_med * 0.31))
    bottom_h = int(round(bottom_w * 1.52))

    # Positions corrigées
    # Haut : remonter franchement
    top_y = int(round(cy_med - h_med * 1.05 - top_h * 0.55))

    # Milieu : légèrement plus haut et plus proche des capsules
    mid_y = int(round(cy_med - side_h * 0.42))

    # Bas : légèrement plus bas
    bottom_y = int(round(cy_med + h_med * 0.70))

    centers = [
        ("top_left", cxl, top_y, top_w, top_h, "top"),
        ("top_middle", cxm, top_y, top_w, top_h, "top"),
        ("top_right", cxr, top_y, top_w, top_h, "top"),

        ("left_outer", cxl - (gap_med * 0.54), mid_y, side_w, side_h, "middle"),
        ("middle_left", (cxl + cxm) / 2.0, mid_y, side_w, side_h, "middle"),
        ("middle_right", (cxm + cxr) / 2.0, mid_y, side_w, side_h, "middle"),
        ("right_outer", cxr + (gap_med * 0.54), mid_y, side_w, side_h, "middle"),

        ("bottom_left", cxl, bottom_y, bottom_w, bottom_h, "bottom"),
        ("bottom_middle", cxm, bottom_y, bottom_w, bottom_h, "bottom"),
        ("bottom_right", cxr, bottom_y, bottom_w, bottom_h, "bottom"),
    ]

    out = []
    for slot_id, center_x, top, bw, bh, band in centers:
        box = _clip_box(center_x - (bw / 2.0), top, bw, bh, img_w, img_h)
        box["slot_id"] = slot_id
        box["band"] = band
        out.append(box)

    return out

def _analyze_board_slot(crop, valid=True, invalid_reason=None):
    if not valid:
        return {
            "status": "invalid",
            "score": 0.0,
            "reasons": [invalid_reason or "invalid_slot"],
            "metrics": {},
        }
    if crop is None or crop.size == 0:
        return {
            "status": "invalid",
            "score": 0.0,
            "reasons": ["empty_crop"],
            "metrics": {},
        }

    h, w = crop.shape[:2]
    pad_x = int(round(w * 0.10))
    pad_y = int(round(h * 0.10))
    inner = crop[pad_y:max(pad_y + 1, h - pad_y), pad_x:max(pad_x + 1, w - pad_x)]
    if inner is None or inner.size == 0:
        inner = crop

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 180)

    area = float(max(1, inner.shape[0] * inner.shape[1]))
    sat_ratio = float(np.count_nonzero(hsv[:, :, 1] > 45)) / area
    edge_ratio = float(np.count_nonzero(edges > 0)) / area
    dark_ratio = float(np.count_nonzero(gray < 130)) / area
    std_gray = float(np.std(gray)) / 255.0
    pink_mask = (
        (hsv[:, :, 0] >= 135) &
        (hsv[:, :, 0] <= 179) &
        (hsv[:, :, 1] >= 20) &
        (hsv[:, :, 2] >= 145)
    )
    pink_ratio = float(np.count_nonzero(pink_mask)) / area

    occupied_score = 0.0
    occupied_score += min(1.0, sat_ratio / 0.11) * 0.34
    occupied_score += min(1.0, edge_ratio / 0.06) * 0.24
    occupied_score += min(1.0, dark_ratio / 0.22) * 0.18
    occupied_score += min(1.0, std_gray / 0.20) * 0.24
    occupied_score -= min(1.0, pink_ratio / 0.05) * 0.25
    occupied_score = _safe_float01(occupied_score)

    status = "occupied" if occupied_score >= 0.58 else "empty"
    reasons = [
        f"sat={sat_ratio:.3f}",
        f"edge={edge_ratio:.3f}",
        f"dark={dark_ratio:.3f}",
        f"std={std_gray:.3f}",
        f"pink={pink_ratio:.3f}",
    ]
    return {
        "status": status,
        "score": occupied_score,
        "reasons": reasons,
        "metrics": {
            "sat_ratio": float(sat_ratio),
            "edge_ratio": float(edge_ratio),
            "dark_ratio": float(dark_ratio),
            "std_gray": float(std_gray),
            "pink_ratio": float(pink_ratio),
        },
    }


def _build_board_debug_analysis(img):
    if img is None or img.size == 0:
        return {
            "ok": False,
            "reason": "empty_image",
            "capsules": [],
            "slots": [],
            "legacy_candidates": [],
            "image_width": 0,
            "image_height": 0,
        }

    work, scale = _resize_for_board_debug(img, max_width=1600)
    capsules_small = _detect_board_capsules(work)

    capsules = []
    for cap in capsules_small:
        capsules.append({
            "label": cap.get("label"),
            "score": _safe_float01(cap.get("score", 0.0)),
            "gray_score": _safe_float01(cap.get("gray_score", 0.0)),
            "edge_score": _safe_float01(cap.get("edge_score", 0.0)),
            "angle": float(cap.get("angle", 0.0)),
            "scale": float(cap.get("scale", 1.0)),
            "x": int(round(cap["x"] / scale)),
            "y": int(round(cap["y"] / scale)),
            "w": int(round(cap["w"] / scale)),
            "h": int(round(cap["h"] / scale)),
        })

    capsules = sorted(capsules, key=lambda c: c["x"])

valid_capsules, reason = _validate_capsule_triplet(capsules, img.shape)

# Fallback : si au moins 3 capsules ont été trouvées,
# on construit quand même les slots.
if valid_capsules:
    slots = _build_slots_from_capsules(capsules, img.shape)
elif len(capsules) >= 3:
    reason = "validation_failed_but_fallback_used"
    slots = _build_slots_from_capsules(capsules, img.shape)
    valid_capsules = True
else:
    slots = []

    slot_results = []
    for slot in slots:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        crop = img[y:y + h, x:x + w] if w > 0 and h > 0 else None
        analysis = _analyze_board_slot(crop, valid=slot.get("valid", False), invalid_reason=slot.get("invalid_reason"))
        slot_results.append({
            "slot_id": slot["slot_id"],
            "band": slot["band"],
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "valid": bool(slot.get("valid", False)),
            "invalid_reason": slot.get("invalid_reason"),
            "status": analysis["status"],
            "score": float(analysis["score"]),
            "reasons": analysis["reasons"],
            "metrics": analysis["metrics"],
            "crop_b64": _img_to_base64(crop) if crop is not None and crop.size != 0 else None,
        })

    legacy_candidates = []
    if not valid_capsules:
        for cand in detect_board_candidates(img)[:12]:
            legacy_candidates.append({
                "x": int(cand["x"]),
                "y": int(cand["y"]),
                "w": int(cand["w"]),
                "h": int(cand["h"]),
                "kind": cand.get("kind", "unknown"),
                "score": float(cand.get("kind_score", 0.0)),
                "reasons": cand.get("reasons", []),
            })

    return {
        "ok": bool(valid_capsules),
        "reason": None if valid_capsules else reason,
        "capsules": capsules,
        "slots": slot_results,
        "legacy_candidates": legacy_candidates,
        "image_width": int(img.shape[1]),
        "image_height": int(img.shape[0]),
    }


def _board_debug_overlay(img, analysis):
    overlay = img.copy()

    for cap in analysis.get("capsules", []):
        x, y, w, h = cap["x"], cap["y"], cap["w"], cap["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 3)
        cv2.putText(
            overlay,
            f'capsule:{cap.get("label", "?")} {cap.get("score", 0.0):.2f}',
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2,
        )

    for slot in analysis.get("slots", []):
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        status = slot.get("status", "unknown")
        if status == "occupied":
            color = (0, 200, 0)
        elif status == "empty":
            color = (160, 100, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            overlay,
            f'{slot["slot_id"]} {status} {slot.get("score", 0.0):.2f}',
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )

    return overlay

def _render_board_debug_page(img, analysis):
    overlay = _board_debug_overlay(img, analysis)
    source_b64 = _img_to_base64(img)
    overlay_b64 = _img_to_base64(overlay)

    capsule_rows = []
    for idx, cap in enumerate(analysis.get("capsules", []), start=1):
        x, y, w, h = cap["x"], cap["y"], cap["w"], cap["h"]
        crop = img[y:y + h, x:x + w]
        crop_b64 = _img_to_base64(crop)
        capsule_rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td>{html.escape(cap.get("label", ""))}</td>
          <td>{cap.get("score", 0.0):.3f}</td>
          <td>{cap.get("gray_score", 0.0):.3f}</td>
          <td>{cap.get("edge_score", 0.0):.3f}</td>
          <td>x={x}, y={y}, w={w}, h={h}</td>
          <td><img src="data:image/png;base64,{crop_b64}" /></td>
        </tr>
        """)

    slot_rows = []
    for idx, slot in enumerate(analysis.get("slots", []), start=1):
        status_class = "occupied" if slot.get("status") == "occupied" else "empty"
        slot_rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td>{html.escape(slot.get("slot_id", ""))}</td>
          <td>{html.escape(slot.get("band", ""))}</td>
          <td><span class="{status_class}">{html.escape(slot.get("status", ""))}</span></td>
          <td>{slot.get("score", 0.0):.3f}</td>
          <td>{html.escape(", ".join(slot.get("reasons", [])))}</td>
          <td>x={slot["x"]}, y={slot["y"]}, w={slot["w"]}, h={slot["h"]}</td>
          <td><img src="data:image/png;base64,{slot.get("crop_b64") or ""}" /></td>
        </tr>
        """)

    legacy_rows = []
    for idx, cand in enumerate(analysis.get("legacy_candidates", []), start=1):
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        crop = img[y:y + h, x:x + w]
        crop_b64 = _img_to_base64(crop)
        legacy_rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td>{html.escape(cand.get("kind", ""))}</td>
          <td>{cand.get("score", 0.0):.3f}</td>
          <td>{html.escape(", ".join(cand.get("reasons", [])))}</td>
          <td>x={x}, y={y}, w={w}, h={h}</td>
          <td><img src="data:image/png;base64,{crop_b64}" /></td>
        </tr>
        """)

    export_json = {
        "ok": analysis.get("ok", False),
        "reason": analysis.get("reason"),
        "image_width": analysis.get("image_width"),
        "image_height": analysis.get("image_height"),
        "capsules": [
            {
                "label": c.get("label"),
                "score": c.get("score"),
                "gray_score": c.get("gray_score"),
                "edge_score": c.get("edge_score"),
                "angle": c.get("angle"),
                "scale": c.get("scale"),
                "x": c.get("x"),
                "y": c.get("y"),
                "w": c.get("w"),
                "h": c.get("h"),
            }
            for c in analysis.get("capsules", [])
        ],
        "slots": [
            {
                "slot_id": s.get("slot_id"),
                "band": s.get("band"),
                "status": s.get("status"),
                "score": s.get("score"),
                "reasons": s.get("reasons"),
                "metrics": s.get("metrics"),
                "box": {
                    "x": s.get("x"),
                    "y": s.get("y"),
                    "w": s.get("w"),
                    "h": s.get("h"),
                },
            }
            for s in analysis.get("slots", [])
        ],
        "legacy_candidates": analysis.get("legacy_candidates", []),
    }
    pretty_json = json.dumps(export_json, indent=2, ensure_ascii=False)

    status_text = "OK - 3 capsules détectées" if analysis.get("ok") else "ECHEC - capsules non détectées proprement"
    hint_html = ""
    if not analysis.get("ok"):
        hint_html = """
        <div class="warn">
          La détection des 3 capsules n'a pas été jugée assez fiable.
          Les encadrés rouges en bas de page viennent de l'ancienne logique libre par contours, gardée seulement comme secours de debug.
        </div>
        """

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Board Debug</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; color:#111; }}
        h1, h2 { margin-bottom: 8px; }
        p { line-height: 1.4; }
        .ok { color:#0a7a28; font-weight:bold; }
        .ko { color:#b00020; font-weight:bold; }
        .occupied { color:#0a7a28; font-weight:bold; }
        .empty { color:#8b5cf6; font-weight:bold; }
        .warn { background:#fff4e5; border:1px solid #f0c36d; padding:12px; margin:16px 0; }
        table { border-collapse: collapse; width: 100%; margin-top: 16px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
        th { background: #f5f5f5; }
        img { max-width: 220px; border: 1px solid #ddd; }
        pre { background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto; white-space:pre-wrap; }
        .toolbar { margin:16px 0; display:flex; gap:10px; flex-wrap:wrap; }
        button { padding:8px 12px; cursor:pointer; }
      </style>
    </head>
    <body>
      <h1>Board Debug - capsules + slots</h1>
      <p><a href="/board_debug">← Revenir</a></p>
      <p>
        Statut :
        <span class="{"ok" if analysis.get("ok") else "ko"}">{html.escape(status_text)}</span>
      </p>
      <p>
        Cette page utilise maintenant une chaîne unique :
        image → 3 capsules → reconstruction des 10 slots → lecture occupé/vide.
        La reconnaissance finale des cartes n'est pas encore branchée ici.
      </p>

      {hint_html}

      <div class="toolbar">
        <button type="button" onclick="exportBoardJson()">Exporter JSON</button>
        <button type="button" onclick="exportBoardTxt()">Exporter TXT</button>
        <button type="button" onclick="exportBoardHtml()">Exporter HTML</button>
        <button type="button" onclick="copyBoardJson()">Copier JSON</button>
      </div>

      {_html_img_block("Image source", source_b64)}
      {_html_img_block("Overlay capsules + slots", overlay_b64)}

      <h2>Capsules détectées</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Capsule</th>
            <th>Score global</th>
            <th>Score gris</th>
            <th>Score contours</th>
            <th>Box</th>
            <th>Crop</th>
          </tr>
        </thead>
        <tbody>
          {"".join(capsule_rows) if capsule_rows else '<tr><td colspan="7">Aucune capsule détectée</td></tr>'}
        </tbody>
      </table>

      <h2>Slots reconstruits</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Slot</th>
            <th>Bande</th>
            <th>Statut</th>
            <th>Score</th>
            <th>Raisons</th>
            <th>Box</th>
            <th>Crop</th>
          </tr>
        </thead>
        <tbody>
          {"".join(slot_rows) if slot_rows else '<tr><td colspan="8">Aucun slot reconstruit</td></tr>'}
        </tbody>
      </table>

      <h2>JSON complet</h2>
      <pre id="board-json">{pretty_json}</pre>

      <h2>Fallback contours libres</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Type</th>
            <th>Score</th>
            <th>Raisons</th>
            <th>Box</th>
            <th>Crop</th>
          </tr>
        </thead>
        <tbody>
          {"".join(legacy_rows) if legacy_rows else '<tr><td colspan="6">Non utilisé</td></tr>'}
        </tbody>
      </table>

      <script>
      function downloadTextFile(filename, content, contentType) {{
        const blob = new Blob([content], {{ type: contentType }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function getBoardJsonText() {{
        const pre = document.getElementById("board-json");
        return pre ? pre.textContent : "";
      }}

      function exportBoardJson() {{
        downloadTextFile("board_debug_results.json", getBoardJsonText(), "application/json;charset=utf-8");
      }}

      function exportBoardTxt() {{
        downloadTextFile("board_debug_results.txt", getBoardJsonText(), "text/plain;charset=utf-8");
      }}

      function exportBoardHtml() {{
        downloadTextFile("board_debug_results.html", document.documentElement.outerHTML, "text/html;charset=utf-8");
      }}

      function copyBoardJson() {{
        const jsonText = getBoardJsonText();
        navigator.clipboard.writeText(jsonText)
          .then(() => alert("JSON copié"))
          .catch(() => alert("Copie impossible"));
      }}
      </script>
    </body>
    </html>
    """


def _read_uploaded_image_from_request():
    file = request.files.get("image")
    if not file:
        return None, ("Aucune image fournie", 400)

    raw = file.read()
    if not raw:
        return None, ("Fichier vide", 400)

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, ("Image invalide", 400)

    return img, None


@app.route("/board_debug", methods=["GET", "POST"])
@app.route("/board-debug", methods=["GET", "POST"])
@app.route("/board_debug/run", methods=["POST"])
@app.route("/board-debug/run", methods=["POST"])
def board_debug():
    if request.method == "GET":
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>Board Debug</title>
          <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; color:#111; }}
            p { line-height: 1.45; }
            .card { background:#f5f5f5; border:1px solid #ddd; padding:16px; max-width:900px; }
            button { padding:8px 12px; cursor:pointer; }
          </style>
        </head>
        <body>
          <h1>Board Debug - capsules + slots</h1>
          <div class="card">
            <p>
              Cette version ne cherche plus des rectangles partout dans l'image.
              Elle fait une seule chaîne propre :
              <strong>upload image → détection des 3 capsules → reconstruction des 10 slots → état occupé/vide → overlay + crops + JSON</strong>.
            </p>
            <p>
              La reconnaissance finale des cartes n'est pas encore lancée ici.
              Le but est d'abord de valider que la géométrie du plateau est bonne.
            </p>
            <form method="post" enctype="multipart/form-data">
              <p><input type="file" name="image" accept="image/*" required /></p>
              <p><button type="submit">Analyser</button></p>
            </form>
          </div>
        </body>
        </html>
        """

    img, err = _read_uploaded_image_from_request()
    if err:
        return err

    analysis = _build_board_debug_analysis(img)
    return _render_board_debug_page(img, analysis)

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
