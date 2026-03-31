import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import base64
import shutil
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
        return None, None

    panel = _extract_symbol_panel(zone)

    if len(panel.shape) == 3:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    else:
        gray = panel.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = _remove_border_touching_components(bin_img, min_area=3)

    if np.count_nonzero(bin_img) == 0:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    bin_img = _keep_main_components(bin_img, max_components=8, min_ratio=0.07)

    return _normalize_binary_to_canvas(bin_img), panel


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

    refs = {
        "icons": {},
        "cards": {
            "SCIENTIFIQUE": [],
            "ASTRONAUTE": [],
            "MECANICIEN": [],
            "MEDECIN": [],
        }
    }

    for name in ["SCIENTIFIQUE", "ASTRONAUTE", "MECANICIEN", "MEDECIN"]:
        path = os.path.join(SYMBOLS_DIR, f"{name.lower()}.png")
        tpl = cv2.imread(path, cv2.IMREAD_COLOR)
        refs["icons"][name] = _normalize_symbol_template(tpl)

    try:
        cards = load_cards_js()
    except Exception:
        cards = []

    for card in cards:
        card_id = card.get("id")
        symbol_name = card.get("symbol")

        if not card_id or symbol_name not in refs["cards"]:
            continue

        path = find_card_image(card_id)
        if path is None:
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            continue

        zone = _extract_symbol_zone_from_card(img)
        if zone is None or zone.size == 0:
            continue

        mask, _ = _normalize_symbol_scan(zone)
        if mask is None:
            continue

        refs["cards"][symbol_name].append({
            "id": card_id,
            "mask": mask
        })

    _SYMBOL_REFS_CACHE = refs
    return refs


def detect_symbol(zone):
    refs = _load_symbol_references()

    empty_debug = {
        "top_candidates": [],
        "winner_references": [],
        "runner_up": None
    }

    if zone is None or zone.size == 0:
        return None, 0.0, 0.0, empty_debug

    variants = _extract_symbol_zone_variants(zone)
    if not variants:
        return None, 0.0, 0.0, empty_debug

    best_result = None
    best_scan_mask = None

    for variant in variants:
        scan_mask, _ = _normalize_symbol_scan(variant)
        if scan_mask is None:
            continue

        per_symbol = {}

        for symbol_name in ["SCIENTIFIQUE", "ASTRONAUTE", "MECANICIEN", "MEDECIN"]:
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
                "support": support
            }

        ranked = sorted(per_symbol.values(), key=lambda d: d["score"], reverse=True)
        if not ranked:
            continue

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        gap = float(best["score"] - (second["score"] if second else 0.0))

        winner_name = best["name"]

        winner_card_scores = []
        for card_ref in refs["cards"].get(winner_name, []):
            ref_mask = card_ref.get("mask")
            if ref_mask is None:
                continue

            s = _score_symbol_masks(scan_mask, ref_mask)
            winner_card_scores.append({
                "card_id": card_ref.get("id"),
                "score": float(s)
            })

        winner_card_scores.sort(key=lambda d: d["score"], reverse=True)

        result = {
            "raw_name": winner_name,
            "score": float(best["score"]),
            "gap": gap,
            "top_candidates": [
                {
                    "best_kind": "card" if best["best_source"] else "icon",
                    "best_source": best["best_source"] or best["name"].lower(),
                    "name": best["name"],
                    "score": float(best["score"]),
                    "support": int(best["support"])
                }
            ],
            "winner_references": winner_card_scores[:5],
            "runner_up": {
                "name": second["name"],
                "score": float(second["score"])
            } if second else None
        }

        if best_result is None:
            best_result = result
            best_scan_mask = scan_mask
        else:
            prev = (best_result["score"], best_result["gap"])
            cur = (result["score"], result["gap"])
            if cur > prev:
                best_result = result
                best_scan_mask = scan_mask

    if best_result is None:
        return None, 0.0, 0.0, empty_debug

    return (
        best_result["raw_name"],
        float(best_result["score"]),
        float(best_result["gap"]),
        {
            "top_candidates": best_result.get("top_candidates", []),
            "winner_references": best_result.get("winner_references", []),
            "runner_up": best_result.get("runner_up")
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

    symbol_name = raw_symbol_name
    if symbol_score < 0.58 or symbol_gap < 0.025:
        symbol_name = None

    symbol_sig = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "name": symbol_name,
        "raw_name": raw_symbol_name,
        "score": float(symbol_score),
        "gap": float(symbol_gap),
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
        "has_special_white_panel": bool(bottom_layout.get("has_special_white_panel", False))
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

        if warped is None or warped.size == 0:
            warped = img.copy()

        if warped is not None and warped.size != 0:
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

        # Sauvegarde de sécurité avant réécriture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{CARDS_JS_PATH}.bak_{timestamp}"
        shutil.copyfile(CARDS_JS_PATH, backup_path)

        global _SYMBOL_REFS_CACHE
        _SYMBOL_REFS_CACHE = None

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
    scan_mask, panel = _normalize_symbol_scan(zone)
    raw_name, score, gap, symbol_debug = detect_symbol(zone)

    pretty_json = json.dumps({
        "raw_name": raw_name,
        "score": score,
        "gap": gap,
        "top_candidates": symbol_debug.get("top_candidates", []),
        "winner_references": symbol_debug.get("winner_references", []),
        "runner_up": symbol_debug.get("runner_up")
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
# BOTTOM TEST
# =====================================================

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
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
