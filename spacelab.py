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


from bottom import (
    extract_bottom_roi_from_full_card,
    analyze_bottom,
    _normalize_badge,
    _extract_digit_mask,
    build_overlay,
)

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

_LEFT_CAPSULE_TEMPLATE_B64 = """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"""
_MID_CAPSULE_TEMPLATE_B64 = """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"""
_RIGHT_CAPSULE_TEMPLATE_B64 = """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"""


def _decode_inline_png_b64(b64_text):
    raw = base64.b64decode((b64_text or "").encode("utf-8"))
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_inline_png_b64_keep_alpha(data_b64):
    if not data_b64:
        return None
    arr = np.frombuffer(base64.b64decode(data_b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _preprocess_capsule_template(img):
    if img is None or img.size == 0:
        return None

    if len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3].copy()
    else:
        bgr = img[:, :, :3].copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        pink_mask = cv2.inRange(hsv, (145, 20, 120), (179, 255, 255))
        not_pink = cv2.bitwise_not(pink_mask)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 140)
        alpha = cv2.bitwise_or(not_pink, edge)

    alpha = cv2.threshold(alpha, 8, 255, cv2.THRESH_BINARY)[1]
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(bgr.shape[1] - 1, x2 + pad)
    y2 = min(bgr.shape[0] - 1, y2 + pad)

    crop_bgr = bgr[y1:y2 + 1, x1:x2 + 1]
    crop_alpha = alpha[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 50, 140)
    edge = cv2.bitwise_and(edge, crop_alpha)

    return {
        "bgr": crop_bgr,
        "gray": gray,
        "edge": edge,
        "mask": crop_alpha,
        "w": int(crop_bgr.shape[1]),
        "h": int(crop_bgr.shape[0]),
    }


def _get_capsule_templates():
    global _CAPSULE_TEMPLATE_CACHE
    if _CAPSULE_TEMPLATE_CACHE is not None:
        return _CAPSULE_TEMPLATE_CACHE

    raw = {
        "left": _decode_inline_png_b64_keep_alpha(_LEFT_CAPSULE_TEMPLATE_B64),
        "middle": _decode_inline_png_b64_keep_alpha(_MID_CAPSULE_TEMPLATE_B64),
        "right": _decode_inline_png_b64_keep_alpha(_RIGHT_CAPSULE_TEMPLATE_B64),
    }

    _CAPSULE_TEMPLATE_CACHE = {}
    for label, img in raw.items():
        _CAPSULE_TEMPLATE_CACHE[label] = _preprocess_capsule_template(img)
    return _CAPSULE_TEMPLATE_CACHE


def _resize_for_board_debug(img, max_width=1600):
    if img is None or img.size == 0:
        return img, 1.0

    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy(), 1.0

    scale = max_width / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale
def _rotate_bound(img, angle, border_value=0):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    cx, cy = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    if len(img.shape) == 2:
        return cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )

def _match_capsule_template(img, template, x_min_frac, x_max_frac):
    if template is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 50, 140)

    h, w = gray.shape[:2]
    x1 = max(0, int(round(w * float(x_min_frac))))
    x2 = min(w, int(round(w * float(x_max_frac))))
    if x2 <= x1 + 40:
        x1, x2 = 0, w

    sub_gray = gray[:, x1:x2]
    sub_edge = edge[:, x1:x2]

    base_gray = template["gray"]
    base_edge = template["edge"]
    base_mask = template["mask"]

    scales = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.45, 1.60]
    angles = [-8, -5, -3, 0, 3, 5, 8]

    best = None

    for angle in angles:
        gray_rot = _rotate_bound(base_gray, angle)
        edge_rot = _rotate_bound(base_edge, angle)
        mask_rot = _rotate_bound(base_mask, angle)
        mask_rot = cv2.threshold(mask_rot, 8, 255, cv2.THRESH_BINARY)[1]

        for scale in scales:
            tpl_gray = cv2.resize(gray_rot, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            tpl_edge = cv2.resize(edge_rot, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tpl_mask = cv2.resize(mask_rot, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tpl_mask = cv2.threshold(tpl_mask, 8, 255, cv2.THRESH_BINARY)[1]

            th, tw = tpl_gray.shape[:2]
            if th >= sub_gray.shape[0] or tw >= sub_gray.shape[1]:
                continue
            if th < 80 or tw < 40:
                continue
            if np.count_nonzero(tpl_mask) < (th * tw * 0.10):
                continue

            res_gray = cv2.matchTemplate(sub_gray, tpl_gray, cv2.TM_CCORR_NORMED, mask=tpl_mask)
            res_edge = cv2.matchTemplate(sub_edge, tpl_edge, cv2.TM_CCORR_NORMED, mask=tpl_mask)
            _, max_gray, _, max_loc_gray = cv2.minMaxLoc(res_gray)
            _, max_edge, _, max_loc_edge = cv2.minMaxLoc(res_edge)

            score = (float(max_gray) * 0.65) + (float(max_edge) * 0.35)
            loc_x = int(round((max_loc_gray[0] + max_loc_edge[0]) / 2.0)) + x1
            loc_y = int(round((max_loc_gray[1] + max_loc_edge[1]) / 2.0))

            cand = {
                "score": float(score),
                "x": int(loc_x),
                "y": int(loc_y),
                "w": int(tw),
                "h": int(th),
                "angle": float(angle),
                "scale": float(scale),
                "gray_score": float(max_gray),
                "edge_score": float(max_edge),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best


def _detect_board_capsules(img):
    templates = _get_capsule_templates()

    searches = [
        ("left", templates.get("left"), 0.00, 0.36),
        ("middle", templates.get("middle"), 0.26, 0.74),
        ("right", templates.get("right"), 0.58, 1.00),
    ]

    found = []
    for label, tpl, xmin, xmax in searches:
        hit = _match_capsule_template(img, tpl, xmin, xmax)
        if not hit:
            continue
        hit["label"] = label
        found.append(hit)

    found = sorted(found, key=lambda c: c["x"])
    if len(found) == 3:
        for idx, label in enumerate(["left", "middle", "right"]):
            found[idx]["label"] = label

    return found


def _clip_box(x, y, w, h, img_w, img_h):
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    return {
        "x": x1,
        "y": y1,
        "w": max(0, x2 - x1),
        "h": max(0, y2 - y1),
    }


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

    top_w = int(round(gap_med * 0.31))
    top_h = int(round(top_w * 1.52))
    side_w = int(round(gap_med * 0.32))
    side_h = int(round(side_w * 1.48))
    bottom_w = int(round(gap_med * 0.31))
    bottom_h = int(round(bottom_w * 1.52))

    top_y = int(round(cy_med - h_med * 1.05 - top_h * 0.55))
    mid_y = int(round(cy_med - side_h * 0.42))
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
        box = _clip_box_dict(center_x - (bw / 2.0), top, bw, bh, img_w, img_h)
        valid, invalid_reason = _box_is_valid(box, img_shape)
        box["slot_id"] = slot_id
        box["band"] = band
        box["valid"] = bool(valid)
        box["invalid_reason"] = invalid_reason
        out.append(box)

    return out


def _analyze_board_slot(crop):
    if crop is None or crop.size == 0:
        return {
            "status": "unknown",
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
        (hsv[:, :, 0] <= 175) &
        (hsv[:, :, 1] >= 20) &
        (hsv[:, :, 2] >= 145)
    )
    pink_ratio = float(np.count_nonzero(pink_mask)) / area

    occupied_score = 0.0
    occupied_score += min(1.0, sat_ratio / 0.10) * 0.35
    occupied_score += min(1.0, edge_ratio / 0.06) * 0.25
    occupied_score += min(1.0, dark_ratio / 0.22) * 0.20
    occupied_score += min(1.0, std_gray / 0.20) * 0.20
    occupied_score -= min(1.0, pink_ratio / 0.06) * 0.15

    occupied_score = max(0.0, min(1.0, occupied_score))
    status = "occupied" if occupied_score >= 0.52 else "empty"

    reasons = [
        f"sat={sat_ratio:.3f}",
        f"edge={edge_ratio:.3f}",
        f"dark={dark_ratio:.3f}",
        f"std={std_gray:.3f}",
        f"pink={pink_ratio:.3f}",
    ]

    return {
        "status": status,
        "score": float(occupied_score),
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
        analysis = _analyze_board_slot(
            crop,
            valid=slot.get("valid", False),
            invalid_reason=slot.get("invalid_reason"),
        )
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
        color = (0, 200, 0) if status == "occupied" else (160, 100, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            overlay,
            f'{slot["slot_id"]} {slot.get("score", 0.0):.2f}',
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )

    for cand in analysis.get("legacy_candidates", []):
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
    
