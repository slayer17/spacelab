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

_LEFT_CAPSULE_TEMPLATE_B64 = """iVBORw0KGgoAAAANSUhEUgAAAD0AAAB0CAIAAAC5V60aAAAxf0lEQVR4nL28+Y9t2XXft9Yez3ynmuu9emN3s/t1N5tkcxBFDZRiawhiRLaGKBOMGEh+MAwH
geMI+SX5BwIE+Sk/JI4RxxYCG7IJyZYoxaGsiRTZFKfuZg+v31SvXg13vueeaU8rP9R7r5ut7ibZIrJRuIVbqKrzOfuuvfba37XWwdd+97tq3cU26AsFv7Ej
hKC18bfH9mRe8nDPzTcu7e3s7yrg3fHS3R5/l6+tN0WR7F2+sLO15c6Wb965e7yaA4FS0cVrl3rDnjFmcjI9eXASyApk165f2djeZIyZqimnpfcBGRdapcNM
aNG2bUCKk6g36MEPPLAcL8B6FohpwXLtTGCdw84SBYNUkQkEkdISGOu8AFigdRQASHLej7NQd6tgOnIAgMAYcsY4ERGE4L0PLtJRb9BXShEBACDi41ciAgTG
2PnbH2qIk9f+3NbLqq4qHxopR5ACeOc7B8EBOGCWORTAg5c2pKiWkknL0JL1jjCkllUxc5JxZIyRVFZwIxjHIF3H2sbZwC8+9fwTTz8jpfxh4T6Iuzr5X25O
xe9/s33tru1HSQqmW3TPXH326tXrPOYG7ZoBJbHgKrY8MtZ57wLVbdc4q6OUE3AEhoQUFHP9zAzSLsmMFhWSyTZ2/+xrN2/ffqVp/8NPfOozP0puHhc37929
dacE3DnYv/DGqy9NT5e7B9egSHQWKw5KCIOMgAsNwVHGBFdigOBDQERgTARA54xpvanbrjtd88PbU9PeiVT13PN5V3evvfLlxcLkee/Jp5/+kXFX9urZg+PF
vUomK7rolU4IV0KKNE3jovCMxUIGD8ETeSJOcRqpLJaSMeeC6TprXec9gmAxclZ1XdO064XxXbQ1GkbJ9Uh30N6bH945unXzR8ndmk2kDeXP2ln14PCYi3i0
sT3oD5XShDxOcyAES568CyEwkHEaF5mSiG3jvLNd5711AYALgZxHS+Gw4MNU7R7s7u3sPZu+Mc7ivtJxsB0RfYgl+N7cUvOLl/ZWNZ2MV51pkzh55vnre9eu
OMmaqry6u9M2XWtbG5yHgEoyxZUSkqPnzHPgWmouVEAAhoEkG1Gqi4hvDXrbw0GSRj44Asj7/dH2zo8KGgAEB/Pxjz/z1HMv3j9b3Ts8unL1ClO6GIy4lNQ2
1tRd17bWeCCUgisoq1VdLwVHKbhABKWUQAaMEULwXVDe80SpRDOgply11rXImU7zvD/8UUEDgAgWolh7YpJIoMjSPteaAfPWBWtvvv6a1rGOE66UD65et6vl
sqnWQrB+rxgOB1mSRjoSTGCgYC3TUSArJSPCqqrLVXnv7lHd2o3tvY3tnR8lN1kOgmzVmLpmgFLEQmjGwDpD1jIiBsQRFGcouZKiq9YdEAfQSmVpmsaJ4ooB
I+998CiEV5LIV3VDrgPiTOiL1z6yeXA5SpK/CmgIoVnXgUII1DSNUCLhKILrkHyeppGOpNZcoLWILoqlJAaMI0dUWkmllOCDXsEZZnmWJamSSiCSJ0+OgmWc
UDAgb50LLvSLXl70k+2Ll596WkfRh4au1tXk5FS3LBkWTAsRhOj1BsS4bmyWpbnO0yTSScwFI4qyJHK2bY0xzlMIIQRE3BiOpJTIwHvvvQvBB8aCt9Y2znYC
PeMgGEfSwJnUEVdROhjuHRx8aGjv/fR03DxYjW9N9j/2ZLyRBxNEURQWQK+bLKdssBnHUsdKack5w5A5Y5ZluVqvq7ar58v5fNkrekVR6EgDMMY4AHjvnOu6
rnamQcm5YEIKJHTkO+tWdb0fx5zzD83dNi03OH7zOM/ywMD7ILQSURSjJymUFFYIZNwj2OADBs4AEVmkopAyIYxx3geKo0QJLZgCCCEggAfyIXQ+tIFa5DrN
8iSOwNGsm/3xn3zl5GzyEWRCiA/NXS5WvrHDwUa8VagkYpwjoLDnkJ6RZ96G4Lwl4wAZcoYcgTFQsZSSxcY564NgMnjmDQDjCAyIvAvOWNt13nZeMiGZc246
nr31xu23bt+dTOaL2bxcrfKi+HDcUsp42I94DJpzKZAQEMWdu3dHW1tKeSmds+VybgAQARkiQ84ZY1Ii44HAOmd9sAaRMQLwwQfvOQaGnoHl6IQADP7o3tFk
PHlwdHR2ctZ1jTPNzVdffu369Rd//LMfbt/hnEvNKAMSDJABAACKr339pWefe0bEKnjbdq6sVkAESJwR58AZMMGRMU/gvXeBGEdkzDvXdI3pjBQikjpPkjSO
OfLZrHzt9deP7t9dlpMQmp2NnV4uDm++8tIf9a8+9eRoc/NDcFer0je8XZt4q0AEAkAAcefo3v37s15xKc0uCN0HSAADgEV0nDnGAzIAhEAQQggEjCPjDJBC
SJgPnkTtRR0EOdbW1a07L7f+wcZenvb0nZvfMu1if/fgrbcOv/P1r1y8/sS//yu/8iGm/OTecfogrNbd1Z96hsccAAhI/PR/8Mt3X36lXtbGHUWZjdNeCN47
E4LlSIwDAhEQITA8JwZkoKQQSnBgQC64QN4ED13Trit44oWfvXjjmbTID7/77S/8n/9oAgvBxaJdfvPrX/7FX/7lD8EdkzxlayWRc8YQAgARCQ54+dIekEeG
/dEwThLvvOmMdw4BEJFCIArIUQoulWTs3PwBALz35J13AQmIaF36s9MuSqOdvb2tvT3wQYmRqUJn/bqrpvMT13Uyin4o9BBCMuwNJlb0JTIGgABAAOLk8M5W
P4ljxgVLEy9U45kHMhY8EAAgF1wIRCQAi2A5Q86Zc845K5UKZIH5ELx33tqqXE9OD+/Qn4t8MFpMF3FUYLDCewlmNZl/89/8/t6LL2zt7iqlfnD0wfYod9oj
MsEDAhEBgHD1ykRAgXGOjAPj6L23xljjyRMRKaW0EgRkbWetFYLrSLdtY4zZ2d3x7pzYG+Oa1kmdTB8cvv7q650JWVIMBz0tddFtiNl4Opv88//tH3/O/srH
f+yzG5tbRASIcRx/8PQTkY602EAPBAzhfC4BxKc+8VECqqrSmA6JKa49eBSckws8hBAo+Lq2FAJBQEQELrgE6Igwy3rOBu8IgANw2w8XD5758p995dvfeXk6
nT3z9LMvPPsTmxv761V77/b915vXv3nzG/kf/bsi0e3Vqz545/3+pSfyoviAXcl2tjxbVQ9WspdkWwUAIgQiwj/8v/7nrm2EFM7ZxXy+ubXprHXeR1oNBsNe
ryjX5Wq5NNZorZMkieMoiqJVWS4WiyRNOVdKpVwoIu49eU93795dLZec80F/sL97iUO0XrZVWVfl6s7x61nU7R5Exch3bnH33or005/+mV95+tnn3497Pl5M
XzmdvznJL21s3dhDICAiIpGmadNUQMQROWNXDg6MtW1TW2et7SaTsXUuhCC4ICLTGQBwznMhNza3hJDIFTIJIEJghFR16yTPuZaRVtubW0mcCFQAXul4eyvZ
2vXBjnf24ryApkn70ehbr95+5aU/yfPehUuX3pO7a7uuNqa1nogeGQlDFEmSKKkoeMH55mhj0OuXZUnOBe9N2zjrhJRpFCPj559s2xrrPBdCR5GOYmSKgPmA
xrmyaoyzQmuVRELy2hmzXg2KgYqFlCxWTMjY2UgK8MaCh/2t7cmkfuvotTdfvTDa2orj+L3sG3QUZf1cpRoQAQgJEUBorfu9fte1grEkio6PjpfLBSIyRHDE
AgpiAjgiB4aBwDrvvG2WZSDY28M4yYXSgNi27Xhymvf6XEsdRcCwXlemq6WKNRNSCRSBkHQc1fW8LI+taa5cLzY2ts/m08ObrxaDjSdvPJdl2buWqdJSbvdi
oVn/PHxHREAAIbnc2twyXRecd8Z87atfbZtmc3Nz2B8AEQSqG1tBBYKhlCh52xnr7HgyXleVdbSxsdkfDoVUne3Kukx6RRTHxAVyXozS8cnZumtJahQMgweO
/eFocrYcTxZNu97as0Kmlw+Kk5n99le+rHT81I0b73KROlJsQ+sk8hzCI30OAIQxxlvXNk1wXjC2Ndo+G4/PTqfjk4lAjIUmAE/BAVlOFsB5h8gZ571iNBpu
BoLT07MoifM8v/Hc07fv3E/7gyzvoVABIB/6plwqgbHU3hvjOud5mveuP3mDMYzT4c07h1n/4pXLB0dnq6//6Z9duHRJDb/n+FwuSr5EaoLaTOEdQp2o17U1
pm27ar1ezZez2Ww+m6/LSgqxPRoNez1CMM6SdzY471257pqmkVL6gPfvH+e9PEqi4F1Vr5kQ/WG/s1Y4n6VJpCIto1vLufEGUHiynpwJHlAm2SjLcsAExZTx
iPM4km49vf/Ff/Fbo+2tS9evXXnyyXM98ezojB2T9GJzkDBEBAICIBLlalWtK2Nd29ll0x6NJ+t5KVAkScaiogzcQWgdWeedD9b51oTWBBe8WLdiumCCF70i
z1Pr7Xw57492ys6tq5qrTOqUCyWVJnLWe4HgCZyntu5CsMaxONHGhK71nFsyPhhnq/XZm0vu3Ghza7S1CQDLWSmmvpcUjCGDc9MnABDL+aJcr1HIOOtt9npH
ixWtbR4X2zsHxe5OI1gAH0zLrIm8j411gZvAtNb94ebW5gYFW67Wcay4gKYqk6zHULdts1yudJwH0yVRGrq16ZzUPAR0nqqmq6uybu0mz9rOMWyCXVfLynRG
KdVNx81k4rr23B6yrICs1aliHB4vWAIQ8+m0sS7rD4qNjcHOzpL4SXJ/oPK9g8u9/f2SQQ2udq0IoYcs8XR4eO/ByXESRdevXjnY33v55W/cuXWnWs9HG72m
Ks9OHmzuX0VHdVW1TWubWkndNJXtLNMRBQzAHIHxXvhAyAMya5yry+Vs5X1QaWzmhBDOD/9EdHD5oNMrBoSMHs42IgCIxXze2FAZ3zK18DC8cHFw7cl8YzPl
qTtZ2zfv33erU9Xlo/z53QtXtva3L16eTsfr1Sp03e3bh6enk+V8tl5PHxyxpq2GWxc2ti9KHnkPtjO2swDeWS8YIAghdBznq3IBTAAXASCOk0IPykm3npdC
i7ptmq4hJPVItEgHSRor9CEQEEB4vC7LxaLxQNatmaij+NnrT8YyN6U9Gh8/mM4XwVSp9kXWKn5zPF2eLraKXpH00sBO5/dO7t8bn47rch5FsLnZO9jf51Em
kAEyh+Ct8c6hs+QIBCJxIbTWsZBaRlGUpIAsTfN+Nqink7qqo6xXrlcgmeplQikicsaXZxXVrYqYTGMEhud7JoDwPgRCzlVa9EdbO0qls6NZebxctq5M1EnE
RS+JEyU6Y8aL+WytN4PYHHBkQqqmboJzDEAL2cuyne1tR0IJRj641izases6CV4Ex0SERAyhaztrHeNC6SgEUkIRgYMASvBYEcBe1N/RPSFE8L6tzPG9CW/b
0W5PZrH3Ac4dCoIQQiom0n5/Z3tve7h9dOtwMqurxpNOVd6/GEes0KiQREWVqdfttKoIwjCNE6U5QqJ1hGmeyCSKFRcYmAAkzhX6pmsxOIYUKa4lJ28h+HVZ
tm2LEqVSznoG2HSdhaCLNI4TroB5H6RCxpwx9bqenk01UW+7AMa8dcgYMgQAkaVpi6LoDdKsqNfmOy+/jnt7/sIGV2kf9EHUaxmuma+S0G0PfCL8Yt0tFyK4
QjEteJRlZDGKmUBmui6AYCEkkWKFbE1gAApJcZAYnOsQqK7WXdvEKhZCrssuOG9c6wQmg16eZJLB3CxTQYgYiEzbcQZCSaYkAQYiJOIAACiGw9F43TGuT1fV
m/eOzfawt7ebjHY6hFvl7FvrN0wNXMpExz0Z54MkGW6Nb74WXH3JCUE0yHPbgVCQRJESynhgFDRHreIiFYKhQO9N29alNW0cq6px3jtABMCqXLcVgZYijSIu
tIqjQKt6KYQ6D0Sk5tuXRlIolcYEwBg/z8YBgciKvAxcKKVlciBxfFTL3C7lmsfRjWzrYr5bez/zdk6uDHZpant/3D+udhVTynhvVRwjSS5BCi249IEQMFhr
Gus8IIBAT9442yD4JMlgTgDAGAOApm68YUIDiyRnXKDgjSVi5zsl5zzvZ0mcegcAGAiY4EAEQAAk5uuyXpfOO27MTlxI1d1+/RtutRtduFz39g5VyrkHV4rl
GCaHtDhLPewksXXt7fGUOTubL61r4lQnwKVKXLCIAogTBQoEEBw5zoAxJCIdqaLIWIRpP1NKKM5kkhjGrDHWBBHpajFHjkpHAOCdq8sqdMiYEkqemzU8ciji
eDIOde2Xc7mey+Eol2rPw/R+1XauvZKv8iylLilncnwYn7zRn9/fdCyBeAZuatuoDmMRQOJQ8yEKoVLuOkBNKBEDogciH0ApyTh1xna2CxDSLCnyTAkZR7rg
vWkLti5t50WUzien+uJO9HDTgXKyNKuQ9vrZSJ7bBz3eL8u6lG0LnV0u5/XsrLe78+TlJ+9M65vHh6Xo1Xusq1Z+cTxcnvXLVbKYiGlVz4yNNQySVWA214JH
QWlQEaDmkgPTyBQTINAHb0JwXCkgaC111lhvExEpyZFIICY6mtdNaC3YICgs5+ODawfn3FLIdtGU41bJBIfEAOkdcaJIsmhdlrYN1JFZ1SYJO9Hwykeum6Pj
V1774yf9cVjanpVDphj07p3Z1voB57Kyikp8Ymfvwr4XYliMoqzXmsB4DFxxGUsmgMi5tutISB6CB8aSLEHuucJA1FXrqix7OPCtEQFTqRlR06y1UpHWIYSu
aZuy9dYh0EPRBjAAEBAQiKJIl8dTC9IRX9adqlTFB6PdywdSsLPX4Wu/nVx+ZvO5zw42Ly0PH3z1L145cuYSk/tdl1BwO8W1G9c6C7HMUp41izbvRXg+34wT
BSYk89JRaLuubjtPwIRwwXZVVS+r9apsRG0aJ4EJHQF5AzaLYq0jAEDGsiyNBan4YdyNiG/vl4fz+bQue7y/c/XitcsH0YXdwcXdaro4OVkvtp7C68/O204G
1+fd8OLGs3/tJzYWp6tX3xzPqn6ebY8u9UeXms5EXEVBmtJyzoRkyMi6rmnqztTW1j50ztVEuFg1RKYxVdPU9apqjEXOfTCmC4GHWnqrRT8vojgCAKnk/vV9
23kQnIjOixDgXNZEEEsKlOg4Svcubl/96I06iedd+WC9eiD0eudjbONCsrx3Wt+T48OtvZ39Tz/tvhWaW/fKWPP+4Eq6i7wvpVGCyQBSCa24kqxp1uPx+Gw8
RiSlxXI1W5VzAH/rLuOcAgZkIIAp5CiFJ2qNQ+a4CCGJsyyXWocQKAQI56oqEkIgwnDOTQAgsmIAqLd7u5tbO3GQswfLB74709r09nR8XYrtKGVlO727mrSS
b/RHozhf5r162Pp+wUmb2hNBIHLkkZHSgrNQV8uzs6Oz8TjLMy56XWfXlSHydtEKhSrWaZZGcaIlWgqtc50nHlhngtJxVORCiM4YX9vF7VNUSm/0pOTwqO4D
zvWqvWxEcnDp4Om0v/fKnfGt8Ty5djXvbY+y7ZEqCvC3OlhtXVouRfPWKefNswdXm91p2XZS8XY5WZ2AlDJIzoC87ZSWPti6WfnQjTZ6w+FQ6STvFZeuXJNa
W28AA1dcaMUCdbPJqq5XTet5GumUXFmwhGUJl6IrV27ZLt44EsM+K1LJkMJD8eQcXJRHk4PdqxKyo7V7mSP72U/XTvdFLxNJptpYdsO8p85efU7+6/3tW8GE
avkE8AuKyfV0cvPl6kGc9Ioi6/XSPO8N+1GWrMvldD6ZTE77gwFAaNta6iROUiF1BETgPIIH6GzXWTBlWXUdqJ5QMS7LnWI0a9ZbAM5519lgDDBAhoDny5Hw
EbkoV/V4q5s0k663kV7bM5FkLTPOHIfFXe9bbn5q+YVfXf6Th7+uAODupSv6n57emM6tnXUnzYN+UQxGo8HWFleirpuuM11nGRcbm1uDwWC+WIbQhcADEQD4
4BygD+h9kEIOt3dXzXRh0NiWu7boJfy8vIYIEXkW8VQzwQDw8U55Ti6iJFVStXVb2XFQoArlSaLWiU56Slxbf+lnl/8EvnfEovv1T738j8RPgoirxXxYFP1B
P+33oyg21hpjrfOcy8FglBX5bDnrTAPMRqil0pGWKBSitMZV1Vpy7V3ouoYLxV0j0w2lFQAordlARS886ZRkkSIiAmBvnzBBZMM+B5YzFVp/dvOWJqf6O7xX
MCkSnnxy/I/hvUaqzGefWNz2z3T9YphEaZ5hFAuhiJhxgQiVTtIsZxxX5bztqiRVzuskSVmcK4FK64jrFpgzoWtNZ0goxbsqTi+K86BKCKZBbhQewSHz9Giy
Hw4SKonWXbc1TAsZ4Z232m+8kj0Tddg/MWLi7xXm1ntyA0AOt2/fvxwLl6rADAFDFnhnvXPEhOYCAuG6Wp+dPgDWSZV3nXSmLFdzKdNefytPB1pIV3kfyFPo
bMPWqzSOz7nbpvVrS0uj8wQi9WhBEgEgASAIDxRHkQDc3hhdurrzxX/3B2/dfXMjGXKVLpan7wcNAJwF473xTeLYkEURBwK4c+fu5mjIuZxOp2+++VaSAIDf
39+6dv0C5/Tg/oPDu4c+CO9CrnPBRV03znpP1NlONFUcx+da+Gwym7w19cfV5Y8/kWwKJt6dbhZFPizL9fHx7XW13Nw5ePrqp99462Z49aUkSTRgfUElwrwn
91mlEbu8RLG1uZHtifXKbfrVfD49q3tZ5Gz6xptvXLq8zaVKkjzSGedhZ+eCksXZeLleV/cOD3ENfh1CF5zxjlnmnM6Tc/vuJYWTZk4VETFExth5duzx1VkS
Z1pLwna2mhwez0ajJz915SM3mB09+G5ydvO71XPvCe0DfPnOkLCrA9VFFl27ku/tsFDybkl27W1tTF3XpTEmipIkKbRKGeim9vNFVa4b4wIBOue7trONN5W3
lRNK6SwTQoQQYqGhDXESSyk4Y5whQ0TE800eAAQFJxWLWQw8TYuejJRMMihGTVeVFF7X/95FXm75N75nURD81jd741anMbBYCSWY4Exit6ylYBBctV56225u
DpACINRVM5nMne2mk8VqWQPKvOgX/UFZLTvbtC50HqWQUZRmUSKEJCKlle4lcU+rWLHzpNj5CQ3O90wS63JKUvSGW8PNvbzYWi7uHq8WUOz7/kXbVaeAvyf+
zovqDw/Wf5jxJhAwBETY6BFWSUCVplHqHJuN2/l43UG/GKzmq6ppsyy5snVwenZSN83R/fvTySnnXHI56A2itCh6G1Ika1Z1IdSMGSVVolQKsYoY5wDAY9W/
PELyXHACwhDw8VYJRIBitRoXW/sH155K+6M3X3/lrVtv6Usf7/pP2KgH1LbTO7abv7n9c6/7T4xvvv61r3zj7/740Qv7zU9eWX27FnOrFlDr+e1xVNoAzqu4
2Lh3eMwZ29vb3d/bm45PvLFt1XZaDkejrc3dne1dJnXVuumkXLdt40MnuJeaJ3GUo2CCMeasI0KmBCMG5wkdAIRzNfbRutze3X/q0g173L3xze+c2qp38Ize
3cZ+tA52sajy4UjcW69OJkNpNSUdmX/53a3n9+5yBj+z98Y/v/PxcrXYFFm70P2d3auXL2olrbGB43nOdmtjJ497WsfD4Wg4HDHOrDOm9bYzRC5OEg9L8IGz
0Ed+XY9AK0Qs50ta4NnRMu/LZJgyJR6GJg/TgAAAItL5q7fv1U4ZlePoKg5HaRplUTVETIlNFkRxb9n4pj5ZLaeMqqN576v3R5+5OH12ePynZ7MJZsPNnTTL
EpVmcdw1TbA+krHggjzs7x4ES1rGSZxJpTvbNtYY70zXUfBxlroA0lHKgvShjWTNUQGsl2s4cu39hZK9qB8TPoI+j2IJAEh0a3Piqer35PZOlm0GHk2IjmsH
DDkqzg0b9RrlZydny3odrNESv7369Iv0RYH+5w9u/tbdn0SVpcVGkfcU443zkYq9hdPjyXrVaKkYkeRMSo4AVdus2zogcaFUlEkVt8AIgHnrXWvlIHAGAHXd
8NJRa88ltQDv2N8fx9/rxZptb+P2Vjvqe2ApY6VnSyuIsMewJzSqQIy5knkZKT4Az7cuPjVLzVb9/1xOx5fz2XSe725sZ1khhVBc7W5fmM7m8+m6XNYUGoC1
FJ1WQQioGtsaDqB7g90sHZKIamAdQ7LGOMalZPgovo6FGEQy0edSJj2e8UdOXBBT69NFEHNw0ZRxv7lxoMVnOLCAJ4YducDXK1XO+nWn9SDff3pdjnuD1B38
mn/9zzjVP77xjf/1L6LNweDC9jZTqsh6z3zk2fl8ZTqXZrquHxh3wuUyjoyKyLnMu95qCZEejYab41Y1HjogoNbYILQ6L9jKiowLroep7kco2HkE+73xCYjN
q9fo8EFz67ZarS8/85EJtAufdSGOiGFoN9qZPToMp4ejWPWvXrsp+eWrO9vb2yiyk/iv79f/ajtefGRwbzLdXlbtoNdHIEm+1xswZEmiy8qtG5cWvbwg5N5Z
dCaxZhWMr8pyPJs70ykOBryx7WNxJ8tznatgg6MQ8KFC9dgLPuSW/f6NUW989+bh2eHyrVo/eWNObi5FQiyeH4U734oV7F7bjbUs55PgawjRnVu3jg4fdNXF
fpKmovrrV976zTs3VuuqtS5WkfeoNWeA3nnr0Die8ASVdKEz5I1FYyh0FmE9G6+sbQRDhh4AkzTlnBOR71yzMlIonglgcC58vXMgAJudHkWx2LtyYXNv05SL
9d3bajnl85P26FZ7dDgS4urebn9QlNXy8OgOhc601YP79+7eunl8dPzSydMAMIjb54dvrcr1eLZwPnjvAYAhCz5YR00HtREtpZ73UQ2Nk4GEkjrSEslLwZAF
7zvOIR0UiBhCsJWZ3x63kzq4gASIj63+3LsiIIr53ZtniRjsbG9cvLbC+MHppGeA3P3Q2kzIixcv5Xk8W5zO7h+tZ9PNYY7gbGfAI0d+e3XpY/ZmIRcvbrz8
e8tPnpzKWMc8oGICJSEC55LJxJHuQqSUkJI56riIcp0lcSokB06taaxr47hXDAeMseBDV7aL4wUnJXZiJvBxjhjoPNN9ni821f37h6fW9zZ3r119ktWmPHxT
m3ZjuLW58yQRf+O1m6Zbcuc3+70skcF7JlCi0FJHOp7qXyjCb0a8e1J/7dXlT6aT6SgvkHkELziL42QjFRhxZCIAD4TGAueR0jEBlutl3ZSmafI0vnj5IB32
OOfO2GB8sI4YhRAwEHDEtw37ockIEHpd1fbouCqbrsg1+Hg4UM4IzuezcRUWTV16V0lBWZqmqVxMZs4FAgvOCaKJubgpLmZw+FTyrVvtC/NVvLc1ijgXDBHA
1GuVKqlQas4EzicL23VKaYfi7HT6xte+s791sP/E5Weee/rgYzfSQY8z1rUmzpONC1v5dl9oBQwJ4F3OBACEVwkQoHOuXNa2SwSP8kIhErHGUnAWMQRyzntj
YHk0Xc3meZykWqN3HYXlfHKn/7ln5W9ydM+kX/l2+9dmi8kgSyMpQnDlapzzTErNhAIL69k4WFsb/+Z377x56/6lT33sl379P7LOaKWSfsEZAwDBeW+rl0WJ
TCPLIAABBXoUeT+OC8V83XEp41gqZKHrfOAhihwT1oeybdZ12zYVosszLQQrV1WaZAcX9jXny8l4OZ/Fcey2r7T6o1Hzrcv61ZvdC+OZihQAce8MByuoFSFQ
06zK8o3vvFyug3UKePbsZz774uc+N9wcxnF8LuOfD+89hUBE9DggOT8xvB0PAhGI09kqivSQQCaAwbeWgg8oZGv8YlWWVb0ul7083tgY5FnkLRXDol/00du5
d6vVIk7ibFHcGX36Kfg2Q3qCfen33vzkan6Sas7Ia0nLedAKQzCz+eL0aKqL/Y29g972waXrT+xd3P/LFUrVqvJnrZt12cUNLOTbNkIA7yjxEIt1K+omGOvz
OOFMCehsIC5qY+flal3Xq+U8TaM0K4oiBeLB09npWTBNWa6cbU9O7s9Wi1ek9le2b2ydXM4eLI++/Pp3kyKWscJYoe/WEr01jSN+4zM/f/HGj11+6vneYBi9
V5UMACwmC3NracYtK5I4yxERiJCAHouaAADAkEtA4QmsDY2xdWPXdbuqmuW6XlV13bbW+jhOi7wX6yTWyXAw8sHXbQ0c4jyywVjXENg/P7pgPSLC33phbqzV
Ohr2hovpMo7y4WhHypxHo8//8n/5/Kd/amt37/2gAcB0rXfEBOfy7Y+CvucbAABTKmJCeWLGUeeodVR1bt2YddNWrWmdJ8bTrIjjDIFFOo6iZF1Vy3JJHJJe
Gph33gD4zquv3hsBwBNb9sZuPZ7M7t49Wi6q2ay6dzidzk3e302LgdL6gwvvvA1cCJXEQgoG7DyQOpeRCeDx6ZgRARELxPy5tCt0QGEDGEcuAIFArqI0A2Tz
xco5QsajOBZatq6drmaNaTw4wIAYXjraqQwDgF96oazablmbor/JeVpWId+4/PQnP8+FfOcSfM+R94tib5Dt91Wi+Hm5FCI9SjM8nn/Wdda54IkBl0wlMs6Z
igMKByygQCGRSy6jujUnJ+O2tavV2hMAZ62z67ZCAUxgIOe9Q6a+fGcTAPb67jPX29qTIdl0wFV/dOmpq89/Qir1fYtMR1uj3l4/3yu4PG8URHwP9w3MueA8
uIA2MI8ycB1QOuCOmAN0gCikC7gqm/F02TTm6PhsMlvUrQXG4jTLegVXMgAhC1Lwb90fLmoBAH/z422ep3VgQWeXnvnolWdfuP70Mx9MDABEFOmIcc6lICCE
d5yG4XuOD0JITYCtDX7ddtarWnjynXONo8Z4792wyKvOooemC6t123StUDaKIS/yotBVvW6aTimdZ3no/O23Dn//L3q/+rnpIHG/9uP5q80nLl29+uSzz27t
7X1faACgQJP74/VRGSVJ/0IfGIbzmsHzrNS5enJ+bjgdT4XUcRwVRRQX/aIoqqY2qyW1BlB48Mt189K3XsmE0J4Ui30wztrWNqtVOxZBxlIluglhfXoye3Bm
Z2b4E/9Jw38n9vc/tffmCv9GsbXT39wsej9QL6tzbj5elCfLfOjzvRwDEnsYmzw0sEeeUHzsYy8+OD5dlMvTyXyxrtNsAYgAgamoF8ecYxbFECgE7yV6JXiz
VqbMWB5J73399AvPjHY3EcL4/uHr8N1waXT9F3+OiafhL/6+hOaG/LN7/sLDrNL3GyEE03bWGGLEFEOGwB5rD49M5JGxi+effx7Fa+JMV20biDxwrVUU6zjW
SaSiSO3v7gkuOmtc8DpS7eSkOTsqhM9kmE3uX9y/3N/bUBL7WVY3ZrVSUZYXF3/RHP4zGv/5BfNH95qfnE53kiTpfb8pJyJrndKK9Vicx4wzxjAAALzjRP8w
nQYizZKNjZGOYxuCJ2JCaC21ljqSsZKRls/ceCZNctd2ru5yKY+S9K7rFDURGNsY8L6uKsx0mqVb/eHy3plbVHiA8mO/0f3+LzFw1+3v3plcKIoiTdMP7oZB
RMZYMepREqIsZpwhMkTC8LZHefucNh6fKo05SxwBcqZ0rLWQgisl4kgoIba3+lrFqxU24EkwipAUuMZY10gKmodUEtjWL+usIivY0AsAYMPn+IVfCPd/d9t8
9W75+em0l2XZcPh9WgKVUsOdYeg8ECA89IGPSjQfvj+PsYTWbGNzu6zb2aLsnDXdOtJFlqZ5GkdSIHjbmtl0tlwsGEM56HswHkzXlr5actfyuuRVaJ07O1vc
Ws7zophE/pxOfPQfmKM/QHJPmN95bX41TdPvy902LdWBexRaImMERAEgPNTqz7PF5wKh2NvdYJx3plES4yipmhaDlYwEkreN4Pjg8M7p+Gy1WsWRts3wdHy/
c5V3FdVzFdp6+qAqvQPR1KyuTTKI2aO+bZZfYld/Nbz1z4b25bh+dbUqVqtVnufvt/WEEEzT1fdLVlPUS+LdHo8VY4EIKRB4AnzYvQMIYn9/a7lcAVnBQpYl
1nbBdRBccF1Xr+NIjc+WZ9OxtRaKvCrZajW1ofZu7epFgo66ddfUTGbcpbbsSrGG8LhYDsSzf8/c/QK46nr7hVfKG8fHx0mScM7fE50xFqVx6RftvO7WnQFM
9wvOMHTWLluzNsCQAUglWCaZVMIFZ62h4AVHwZHIB2+dM9YagEDkkyQebQxGo34SR5wBg+BMa9ouUrESkeRaq1gy2Zbr+3dud03z2OuxeFN85O8AQO5uZ+VX
Tk9Pm6YJ77ixd3Fn/UyNIjFUDly1KjvT+WBN160Xq9mDyex4PHswXk4WbdOKw/sP5stl0xnrQ9XUxBhn0kMgoDhNkixLejmTQkopkDvX9af9pVk2FiBEeXFR
pRd8qFBKajtTVcenp+Ssd048shb+1H/h3vyn0E2vtr/z9fZjDx48ODg4eM8SdQDgnO8+sb/qL1ZnS0cE3HsEz71PiAZwHmCFFCgmcfOtu8474ygQrqpaSBlp
HQA67zjDyWLemq6zFgiF0IJju+64BRkEiqIYXY56l0EYCg4WR65r8iROkuSdBxOUmXj277mv/49JON2q//j4ONrc3NRav19gKJXsbQ/SYRYCccEZYyGE/m7f
+/Do3hjjXDQGPQlkgjFGiMS4Dcy3njPHOQL5uu3a1hBxpaSWIrSetW0wjWCst7HBewOlOdlGRvMk0lv9wWhz611M/Nqvudf+d6gOr5jfO6k/eXZ2JqXM8/w9
uQFACPFuT6/1u43KeSAQyCQTkjERAhjTVXW1KsvFarUs151xgIJQtgZWpalWa1uXYCsuXTZMRJao3kDlAxSq67o0y8v1+l3cyKT86H8LACosL3RfOjs7W61W
57LWhx6C2xUwZO68l5I8hRCs88bZznvrvY+iROsUQJkObNPJ9bEOk0jUeR7rwsuEeMRd4I3tpvPF1rMvVFX9ly/DLv4C9G/A4pVL5t8eLT+7XG70+/00Tf8K
3OUbFBwF68kHAkMBEYicN23TVF3b2ih1cYEs6jpwxnB3xtUyz2BzlzM1Y6wfnOuqerGcz5tu88KB/EufKZwXMX7sN+yX/jNB7VX3B6fzC1mW/ZW477/yJc0p
VSAEWmItEeecceAQUvIxeF+jXTHjmbEQXEhlx/Mw2tzY2VYmnM7O7GKBXUVNXfW3dw+efCp5Hxq+/WNu+3N0+id73R/fX/z0NEkGg8EHWPn34f7bv/63vUqD
ToXAyNWmm0axEpJ78s7ZztgQkEgAKAYKkbdUOlglERskmQ5iMWmWR/PJuFytuuc/+YlnP/Giit+3T1688A/tF/+Ugb9q/vXt1cXxePyXG19+UO4OB0AcHQbr
WwMWMluHKBHARGN83QIRO28/CwGJiGPCQHZdmCyDN009K+t5uzqbB6EvPfEUl+oDev344Bl/6W+Eu1/Y7L52uP7ZySTe3d39AJ/4Qdy33rgt0UvwNtil7xKu
AwWpJWNobNu1XQgsgAggnGeBWMJlyqRgROACWFZ13EOajLInrj3x4ov8+/Vty+f/m+7w32CwV5t/9Wp5ZTqdbm1t/VBtjQ+5B/sXMJAkWIbOhPJq1Ds+mzbG
AwEEwTAOCBQAkAvBgfEAvGGcM2QYOBJDz5Bv72zuf/z5g8vXv+/DcDDd59f/U//G/zFwr6XVt2/dkkVRfAhufBwt/Agf8vHBg7p599ufB7cu+YWX0n/41Ec+
srOzk/yQT484LzPA/9+gAQD1gD/9XwFA7u9vmpfOzs6qqvqh/8nLf/6tuEjzfqEj7Y2dnUxM3QXvAFFG0WB3Q2nljF0tVuV8RUCAqKKoGORZkRHR4my2ni6d
cYwxlKK3Pcx6GSI267perIbbG2k/19G7PTq5pv2dn8F23LDRS73/4fLVJy5evBj9MA++YPqYZCWYlxgEGqbGqI+cumP0oY8WTHspHGMt8iWIMfAzEhNQKyY7
yb0QTsiSqxNS95184PUUlVUsCOYFrxgdmvaN+d1vvj4/m77rPI8ils/91wAQh+lO+0ez2Ww+n/8gZ/63uW3nXRecca5ztnOmteett855Z4K3wRnvWuc656z3
NngbvA/eem+8s94b7wMEevjlXfA22M51tenK7uSVQzWhUNqu6d51YX7lb0F+FQAut19cTo8mk8kPFbHg2avHLJKoBDIEF0JjyAUIARgyLUUkCSA47633zgMQ
IOOScymY5AjgW+s7R+fPhOKMRxIlR0Ayzi7b9d3p/PRs8PSuvlDsHbxbsnKHX3R/+ncB4I7++dn2f3zlypXd3d0fkFvIPDovnAUA5ChSDeeS7XnBB0MAYIIj
ZxwEnecPER8JjsAjyZWgxxWJDws4kGnJhsx6O+qLZKuH+j38urj4c370MZp+46L5f0/Xnz89TUajkZTyB3ESDBh7yHKeiGAIgqPkKDjw83JxBIbAGTLGGEPG
kDFgjwA5Q8GZ5Cg5coaMPVYiUfJ0Ix8cbKWD7C8vzfMhX/jvAICTuVj9znK5nE6nzrkfaL4f6rSPxaxzieLxLRPgec0nET1O255rRo8BGcDjhomHDQhEgAQU
PLEQgOjx7bx72jZfZHs/Gx78293uT+9Xnz88jPM8F0KcXz+MX6Lm7H243yEunwv6+La6Qo9++uiG3iaHR7Va8CiXSw+JzwtbgIIP67Ml1V12oSeG75sYER/9
B92DLyGEy/UXvjMeVVUVRZEQwt/+LfvV3wB6n0P0O8jg8aUf5yYe5Sce1STQw+4IJDp/xcdZDHhYCkWBiIhC8MbN743Hr96rJiW9zxkeAFjvCX7lbwLApvtW
4e8cHR2VZelu/YsPgH7E/c5B70gZPgZ+dGP4nr8Z6OHro/s+d4rkA7Mh5iqKtHqvw8TjIZ/7+8Q0Alzvfns2m3Vv/t/uq//9B0DDeX7nXSAA75rm86mFx9Vk
jxpS6NEMv/0Kj4yEiMgH7ilPYyaYkB+oaCa7/In/HAAG/ual1W/2bv1P7+i0fJ8/mR0tH83iu7erd8zuY+N+O3HxjrQL0uO00eO0F5HrnHuwVIjxtZHeyhn/
oCCbzKr77Z8Gu6J3farvM/4/utxRXmy2f/AAAAAASUVORK5CYII="""
_MID_CAPSULE_TEMPLATE_B64 = """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"""
_RIGHT_CAPSULE_TEMPLATE_B64 = """iVBORw0KGgoAAAANSUhEUgAAAD0AAABzCAIAAACkUp2iAAArCUlEQVR4nMW82Zsk13EvFhFny72qunqbHQtBEAAByhQl2bRk3U3+7De/+z/zu7dnL0++4nd5
fXW1kbREACQAAhgMZqanl+racjtr+KF6emZAcGhC8uXpr6s6s7IzfxkZESdOxC8Kf/I3P62buqwKIhq6fn25Yk6ImBd5M2m00UPXdZutGx0AkhST/T2TZ877
zXrTb1tiIKKyrutJo42RSmZFppSC/58H/uTv/j6llFICAEJAJNx9woljJAQpBApiRAZQUh3euNlMp/8JkL18yF/82/8YU4ohBOdj8AioELSAjDjXWGeqLnRV
ZSoTIfpt252++nZ48929oxu/X+iy3W9SiokTMxCDElIIKSikZDu/Xawv/Omq6dVsryhLLar48V/9T+bjt97+L//bN979PiL+3nAf3rwhhBBEAASRU4wco4Ag
2LsuW67X5+thw9HWxaGuZoWub4ZP338/BFPWs1uvvvZ7w53nRV1VRmlEij4O/dB3HSSJqDBPotxXEwe19vXe0OzlZVaLGX902V6sLp+c/j5xI6IxJjcZADp0
zjpABCQUJE1ZTQ+SkFhINa1TVQ1GNcWemh74ltfni98XaACQSMQALvjggx2s804QSa2MksgZeIcQ2JDJtBYiJS/KTGRy+/j8/PHj3yfuru+6bmtHOwxDdKEu
69lkmhtdZJoggi+JHeuktCYBlh2ISBms1hdffvYr55zW+veDWwjyzjs7Dn3PIdZFabTMjFICIYAmMiSs98NmG8fkZEgYMdeT48P9mzeFEP90BCklZk4xpZR2
DooEEREA7HYyMwICgpQCEHc7ZV2WWFdVUVRlF71vqibPjRJEzClFiBESC0REAimzOkct9m/e2Kf9W69855+Cu+/65fmSgSeTCUbYLtbrs6UgqbVp9ifFrBRS
jJtxc7pygyVElenmeKJy7UffLTZSS2GyrMiyqiiiD0brTGsphGSOkYA5pQQIkqTUStdlktAcSBAwPzr6xqDHcTw7ObVPhqzIAzqOabzou5NWoPDK6ag0Ks6U
uxzbhxvXWyFEXmbOZBgx9NZejHIchhSClFIJkSsthVBCKCkVYkRGIWJKngN5giBNwExKrdCis/3wjXF3my1t08nffHbj2/eKohCZMvNmYhQwCCDKVGSGEEGT
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
rkJggg=="""


def _decode_inline_png_b64(b64_text):
    raw = base64.b64decode((b64_text or "").encode("utf-8"))
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _get_capsule_templates():
    global _CAPSULE_TEMPLATE_CACHE
    if _CAPSULE_TEMPLATE_CACHE is not None:
        return _CAPSULE_TEMPLATE_CACHE

    _CAPSULE_TEMPLATE_CACHE = {
        "left": _decode_inline_png_b64(_LEFT_CAPSULE_TEMPLATE_B64),
        "middle": _decode_inline_png_b64(_MID_CAPSULE_TEMPLATE_B64),
        "right": _decode_inline_png_b64(_RIGHT_CAPSULE_TEMPLATE_B64),
    }
    return _CAPSULE_TEMPLATE_CACHE


def _rotate_bound(img, angle_deg):
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
        borderValue=(255, 255, 255),
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


def _match_capsule_template(img, template, x_min_frac, x_max_frac):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 80, 180)

    h, w = gray.shape[:2]
    x1 = max(0, int(round(w * float(x_min_frac))))
    x2 = min(w, int(round(w * float(x_max_frac))))
    if x2 <= x1 + 20:
        x1, x2 = 0, w

    sub_gray = gray[:, x1:x2]
    sub_edge = edge[:, x1:x2]

    scales = [0.75, 0.85, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40]
    angles = [-10, -6, -3, 0, 3, 6, 10]

    best = None

    for angle in angles:
        rotated = _rotate_bound(template, angle)
        tpl_gray_base = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        tpl_edge_base = cv2.Canny(tpl_gray_base, 80, 180)

        for scale in scales:
            tpl_gray = cv2.resize(
                tpl_gray_base,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            tpl_edge = cv2.resize(
                tpl_edge_base,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

            th, tw = tpl_gray.shape[:2]
            if th >= sub_gray.shape[0] or tw >= sub_gray.shape[1]:
                continue
            if th < 40 or tw < 20:
                continue

            res_gray = cv2.matchTemplate(sub_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            res_edge = cv2.matchTemplate(sub_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)

            _, max_gray, _, max_loc_gray = cv2.minMaxLoc(res_gray)
            _, max_edge, _, max_loc_edge = cv2.minMaxLoc(res_edge)

            score = (float(max_gray) + float(max_edge)) / 2.0
            loc_x = int(round((max_loc_gray[0] + max_loc_edge[0]) / 2.0)) + x1
            loc_y = int(round((max_loc_gray[1] + max_loc_edge[1]) / 2.0))

            cand = {
                "score": score,
                "x": loc_x,
                "y": loc_y,
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
        ("left", templates["left"], 0.00, 0.52),
        ("middle", templates["middle"], 0.18, 0.82),
        ("right", templates["right"], 0.48, 1.00),
    ]

    found = []
    for label, tpl, xmin, xmax in searches:
        if tpl is None or tpl.size == 0:
            continue

        hit = _match_capsule_template(img, tpl, xmin, xmax)
        if not hit:
            continue

        hit["label"] = label
        found.append(hit)

    found.sort(key=lambda c: c["x"])

    if len(found) == 3:
        found[0]["label"] = "left"
        found[1]["label"] = "middle"
        found[2]["label"] = "right"

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

    xs = np.array([c["x"] for c in capsules], dtype=np.float32)
    ys = np.array([c["y"] for c in capsules], dtype=np.float32)
    ws = np.array([c["w"] for c in capsules], dtype=np.float32)
    hs = np.array([c["h"] for c in capsules], dtype=np.float32)
    cxs = xs + (ws / 2.0)

    slot_w = int(round(float(np.median(ws)) * 1.05))
    slot_h = int(round(float(np.median(hs)) * 0.92))
    mid_slot_h = int(round(slot_h * 0.92))

    top_y = int(round(float(np.median(ys)) - (slot_h * 0.60)))
    mid_y = int(round(float(np.median(ys)) + (float(np.median(hs)) * 0.12)))
    bottom_y = int(round(float(np.median(ys)) + (float(np.median(hs)) * 0.95)))

    left_cx = float(cxs[0])
    mid_cx = float(cxs[1])
    right_cx = float(cxs[2])

    centers = [
        ("top_left", left_cx, top_y, slot_w, slot_h, "top"),
        ("top_middle", mid_cx, top_y, slot_w, slot_h, "top"),
        ("top_right", right_cx, top_y, slot_w, slot_h, "top"),
        ("left_outer", left_cx - (slot_w * 1.55), mid_y, slot_w, mid_slot_h, "middle"),
        ("middle_left", (left_cx + mid_cx) / 2.0, mid_y, slot_w, mid_slot_h, "middle"),
        ("middle_right", (mid_cx + right_cx) / 2.0, mid_y, slot_w, mid_slot_h, "middle"),
        ("right_outer", right_cx + (slot_w * 1.25), mid_y, slot_w, mid_slot_h, "middle"),
        ("bottom_left", left_cx, bottom_y, slot_w, slot_h, "bottom"),
        ("bottom_middle", mid_cx, bottom_y, slot_w, slot_h, "bottom"),
        ("bottom_right", right_cx, bottom_y, slot_w, slot_h, "bottom"),
    ]

    out = []
    for slot_id, cx, y, w, h, band in centers:
        box = _clip_box(cx - (w / 2.0), y, w, h, img_w, img_h)
        box["slot_id"] = slot_id
        box["band"] = band
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
        }

    work, scale = _resize_for_board_debug(img, max_width=1600)
    capsules_small = _detect_board_capsules(work)

    capsules = []
    for cap in capsules_small:
        capsules.append({
            "label": cap.get("label"),
            "score": float(cap.get("score", 0.0)),
            "gray_score": float(cap.get("gray_score", 0.0)),
            "edge_score": float(cap.get("edge_score", 0.0)),
            "angle": float(cap.get("angle", 0.0)),
            "scale": float(cap.get("scale", 1.0)),
            "x": int(round(cap["x"] / scale)),
            "y": int(round(cap["y"] / scale)),
            "w": int(round(cap["w"] / scale)),
            "h": int(round(cap["h"] / scale)),
        })

    capsules = sorted(capsules, key=lambda c: c["x"])

    slots = _build_slots_from_capsules(capsules, img.shape) if len(capsules) >= 3 else []

    slot_results = []
    for slot in slots:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        crop = img[y:y + h, x:x + w]
        analysis = _analyze_board_slot(crop)
        slot_results.append({
            "slot_id": slot["slot_id"],
            "band": slot["band"],
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "status": analysis["status"],
            "score": float(analysis["score"]),
            "reasons": analysis["reasons"],
            "metrics": analysis["metrics"],
            "crop_b64": _img_to_base64(crop),
        })

    legacy_candidates = []
    if len(capsules) < 3:
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
        "ok": len(capsules) >= 3,
        "reason": None if len(capsules) >= 3 else "capsules_not_found",
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
        body { font-family: Arial, sans-serif; padding: 20px; color:#111; }
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
            body { font-family: Arial, sans-serif; padding: 20px; color:#111; }
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
    
