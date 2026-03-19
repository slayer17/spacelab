import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# =====================================================
# OUTILS DE BASE
# =====================================================

def _clip_box(x, y, w, h, max_w, max_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))

    if x + w > max_w:
        w = max_w - x
    if y + h > max_h:
        h = max_h - y

    w = max(1, w)
    h = max(1, h)
    return x, y, w, h


def _offset_box(box, dx, dy):
    if box is None:
        return None
    x, y, w, h = box
    return int(x + dx), int(y + dy), int(w), int(h)


def _save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# =====================================================
# EXTRACTION ROI BAS
# =====================================================

def extract_bottom_roi_from_full_card(img):
    """
    Reprend la même logique de base que le projet principal :
    - resize en 200x300
    - ROI du bas à gauche
    """
    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]

    x1 = int(w * 0.00)
    x2 = int(w * 0.55)
    y1 = int(h * 0.82)
    y2 = int(h * 1.00)

    roi = img[y1:y2, x1:x2]
    return img, roi, (x1, y1, x2 - x1, y2 - y1)


# =====================================================
# LECTURE DES CHIFFRES
# =====================================================

def _normalize_badge(img_or_mask, target=96):
    if img_or_mask is None or img_or_mask.size == 0:
        return None

    if len(img_or_mask.shape) == 3:
        gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_mask.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    _, white_mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 20:
        return None

    x, y, w, h = cv2.boundingRect(best)
    crop = gray[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None

    canvas = np.zeros((target, target), dtype=np.uint8)

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    scale = min((target - 12) / cw, (target - 12) / ch)
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))

    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _extract_digit_mask(img_or_mask, target=64):
    """
    Extrait seulement la forme noire du chiffre.
    Cette fonction est volontairement isolée pour pouvoir être modifiée
    sans toucher au reste du projet.
    """
    if img_or_mask is None or img_or_mask.size == 0:
        return None

    if len(img_or_mask.shape) == 3:
        gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_mask.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    _, white_mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 20:
        return None

    badge_fill = np.zeros_like(white_mask)
    cv2.drawContours(badge_fill, [best], -1, 255, thickness=-1)

    x, y, w, h = cv2.boundingRect(best)
    crop_gray = gray[y:y + h, x:x + w]
    crop_badge = badge_fill[y:y + h, x:x + w]
    if crop_gray is None or crop_gray.size == 0:
        return None

    inner_badge = cv2.erode(crop_badge, np.ones((3, 3), np.uint8), iterations=1)
    badge_pixels = crop_gray[inner_badge > 0]
    if badge_pixels.size == 0:
        return None

    dark_threshold = np.percentile(badge_pixels, 45)
    dark_mask = (crop_gray < dark_threshold).astype(np.uint8) * 255
    digit_mask = cv2.bitwise_and(dark_mask, inner_badge)

    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, kernel)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = max(6, int(digit_mask.shape[0] * digit_mask.shape[1] * 0.01))
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not kept:
        return None

    clean = np.zeros_like(digit_mask)
    cv2.drawContours(clean, kept, -1, 255, thickness=-1)

    ys, xs = np.where(clean > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = clean[y1:y2 + 1, x1:x2 + 1]
    if crop is None or crop.size == 0:
        return None

    canvas = np.zeros((target, target), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    scale = min((target - 10) / cw, (target - 10) / ch)
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _binary_mask_score(a, b):
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
    if full_a is None or full_b is None:
        return 0.0

    diff = cv2.absdiff(full_a, full_b)
    diff_score = 1.0 - (float(np.mean(diff)) / 255.0)

    blur_a = cv2.GaussianBlur(full_a, (3, 3), 0)
    blur_b = cv2.GaussianBlur(full_b, (3, 3), 0)
    diff2 = cv2.absdiff(blur_a, blur_b)
    structure_score = 1.0 - (float(np.mean(diff2)) / 255.0)

    full_score = float((diff_score * 0.30) + (structure_score * 0.20))

    if digit_a is None or digit_b is None:
        return full_score

    digit_score = _binary_mask_score(digit_a, digit_b)
    return float(full_score + (digit_score * 0.50))


def detect_digit(zone, digits_dir):
    if zone is None or zone.size == 0:
        return {
            "digit": None,
            "score": 0.0,
            "gap": 0.0,
            "scores": []
        }

    scan_badge = _normalize_badge(zone)
    if scan_badge is None:
        return {
            "digit": None,
            "score": 0.0,
            "gap": 0.0,
            "scores": []
        }

    scan_digit = _extract_digit_mask(zone)
    scores = []

    for n in range(1, 11):
        path = Path(digits_dir) / f"{n}.png"
        tpl = cv2.imread(str(path))
        if tpl is None or tpl.size == 0:
            continue

        tpl_badge = _normalize_badge(tpl)
        if tpl_badge is None:
            continue

        tpl_digit = _extract_digit_mask(tpl)
        score = _digit_score(scan_badge, tpl_badge, scan_digit, tpl_digit)
        scores.append({"digit": n, "score": float(score)})

    if not scores:
        return {
            "digit": None,
            "score": 0.0,
            "gap": 0.0,
            "scores": []
        }

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    best = scores[0]
    second = scores[1] if len(scores) > 1 else {"score": 0.0}

    return {
        "digit": int(best["digit"]),
        "score": float(best["score"]),
        "gap": float(best["score"] - second["score"]),
        "scores": scores
    }


# =====================================================
# ANALYSE DU BAS
# =====================================================

def _make_bottom_light_mask(zone):
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
    Cherche le grand panneau noir du bas.
    Si ça échoue, on renvoie un fallback fixe.
    """
    if bottom_zone is None or bottom_zone.size == 0:
        return None

    gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    fx = int(w * 0.02)
    fy = int(h * 0.08)
    fw = int(w * 0.50)
    fh = int(h * 0.84)
    return _clip_box(fx, fy, fw, fh, w, h)


def _find_special_white_panel_box(bottom_zone):
    if bottom_zone is None or bottom_zone.size == 0:
        return None

    mask, gray = _make_bottom_light_mask(bottom_zone)
    if mask is None:
        return None

    h, w = gray.shape[:2]
    dark_ratio = float(np.count_nonzero(gray < 90)) / float(max(gray.size, 1))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def _find_points_badge_in_black_panel(panel_zone):
    """
    Cherche la zone du badge points dans un panneau noir classique.

    Nouvelle version :
    - crop plus large
    - plus haut
    - presque toute la hauteur
    - pour ne plus couper le badge
    """
    if panel_zone is None or panel_zone.size == 0:
        return None, None

    ph, pw = panel_zone.shape[:2]
    if ph == 0 or pw == 0:
        return None, None

    # Zone gauche plus large qu'avant
    x = int(pw * 0.00)
    y = int(ph * 0.04)
    w = int(pw * 0.46)
    h = int(ph * 0.92)

    x, y, w, h = _clip_box(x, y, w, h, pw, ph)

    crop = panel_zone[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None, None

    return crop, (x, y, w, h)


def _find_slash_box(panel_zone):
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    return x + x1, y, bw, bh


def _find_right_icon_box(panel_zone):
    """
    Cherche la grande icône blanche à droite.

    Nouvelle version :
    - zone plus à droite
    - filtres plus stricts
    - évite de prendre le bord décoratif
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.64)
    x2 = pw
    y1 = int(ph * 0.08)
    y2 = int(ph * 0.72)

    zone = panel_zone[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.08:
            continue

        if bw < zw * 0.22:
            continue

        if bh < zh * 0.28:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 0.55 or ratio > 1.50:
            continue

        # On évite les éléments trop collés au bord droit du décor
        if (x + bw) >= (zw - 1):
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

    Nouvelle version :
    - plus stricte
    - évite de prendre les bords du panneau
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.52)
    x2 = pw
    y1 = int(ph * 0.68)
    y2 = ph

    zone = panel_zone[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.015:
            continue

        if bw < zw * 0.24:
            continue

        if bh > zh * 0.32:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 2.4:
            continue

        # On évite les traits collés au bas ou au bord
        if (y + bh) >= (zh - 1):
            continue

        score = (ratio * 1.5) + (area_ratio * 6.0)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    return (x + x1, y + y1, bw, bh)

def analyze_bottom(bottom_zone, digits_dir):
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
        "special_box": None,
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
    if badge_crop is not None and badge_box_local is not None:
        digit_res = detect_digit(badge_crop, digits_dir)
        result["raw_points"] = digit_res["digit"]
        result["points_score"] = float(digit_res["score"])
        result["points_gap"] = float(digit_res["gap"])

        if digit_res["score"] >= 0.72 or (digit_res["score"] >= 0.60 and digit_res["gap"] >= 0.02):
            result["points"] = digit_res["digit"]

        bx, by, bw, bh = badge_box_local
        result["points_box"] = _offset_box((bx, by, bw, bh), px, py)
        result["digit_scores"] = digit_res["scores"]

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

    if result["points"] is not None and not result["has_slash"] and not result["has_right_icon"]:
        result["layout"] = "NUMBER_ONLY"
    elif result["points"] is not None and result["has_slash"] and result["has_right_icon"] and result["has_bottom_line"]:
        result["layout"] = "NUMBER_ICON_LINE"
    elif result["points"] is not None and result["has_slash"] and result["has_right_icon"]:
        result["layout"] = "NUMBER_ICON"
    else:
        result["layout"] = "BLACK_PANEL"

    return result


# =====================================================
# DEBUG VISUEL
# =====================================================

def draw_box(img, box, color, label):
    if box is None:
        return
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img,
        label,
        (x, max(15, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def build_overlay(full_img, bottom_box, result):
    overlay = full_img.copy()

    draw_box(overlay, bottom_box, (255, 0, 255), "BOTTOM")

    bx, by, _, _ = bottom_box

    if result.get("panel_box") is not None:
        draw_box(overlay, _offset_box(result["panel_box"], bx, by), (255, 255, 255), "PANEL")

    if result.get("special_box") is not None:
        draw_box(overlay, _offset_box(result["special_box"], bx, by), (0, 255, 255), "SPECIAL")

    if result.get("points_box") is not None:
        draw_box(overlay, _offset_box(result["points_box"], bx, by), (255, 255, 255), "POINTS")

    if result.get("slash_box") is not None:
        draw_box(overlay, _offset_box(result["slash_box"], bx, by), (220, 220, 220), "SLASH")

    if result.get("right_icon_box") is not None:
        draw_box(overlay, _offset_box(result["right_icon_box"], bx, by), (0, 255, 255), "ICON")

    if result.get("bottom_line_box") is not None:
        draw_box(overlay, _offset_box(result["bottom_line_box"], bx, by), (0, 255, 0), "LINE")

    return overlay


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Debug du ROI bottom de Space Lab")
    parser.add_argument("--image", required=True, help="image d'une carte complète, ou d'un crop du bas")
    parser.add_argument("--mode", choices=["full", "bottom"], default="full", help="full = carte complète, bottom = crop du bas")
    parser.add_argument("--digits-dir", default=None, help="dossier digits contenant 1.png à 10.png")
    parser.add_argument("--out-dir", default="bottom_debug_out", help="dossier de sortie pour les images debug")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    digits_dir = Path(args.digits_dir) if args.digits_dir else (Path(__file__).resolve().parent / "digits")

    if not digits_dir.exists():
        raise FileNotFoundError(
            f"Dossier digits introuvable : {digits_dir}\n"
            f"Ajoute --digits-dir /chemin/vers/digits"
        )

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image : {image_path}")

    if args.mode == "full":
        full_img, bottom_roi, bottom_box = bottom.extract_bottom_roi_from_full_card(img)
result = bottom.analyze_bottom(bottom_roi, DIGITS_DIR)
overlay = bottom.build_overlay(full_img, bottom_box, result)
    else:
        full_img = img.copy()
        bottom_roi = img.copy()
        h, w = img.shape[:2]
        bottom_box = (0, 0, w, h)

    result = analyze_bottom(bottom_roi, digits_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_image(out_dir / "01_full_or_input.png", full_img)
    _save_image(out_dir / "02_bottom_roi.png", bottom_roi)

    if result.get("panel_box") is not None:
        x, y, w, h = result["panel_box"]
        _save_image(out_dir / "03_panel.png", bottom_roi[y:y + h, x:x + w])

    if result.get("points_box") is not None:
        x, y, w, h = result["points_box"]
        points_crop = bottom_roi[y:y + h, x:x + w]
        _save_image(out_dir / "04_points_crop.png", points_crop)

        badge_norm = bottom._normalize_badge(points_crop)
        digit_mask = bottom._extract_digit_mask(points_crop)
        if badge_norm is not None:
            _save_image(out_dir / "05_points_badge_norm.png", badge_norm)

        digit_mask = _extract_digit_mask(points_crop)
        if digit_mask is not None:
            _save_image(out_dir / "06_points_digit_mask.png", digit_mask)

    overlay = build_overlay(full_img, bottom_box, result)
    _save_image(out_dir / "99_overlay.png", overlay)

    json_result = {
        "image": str(image_path),
        "mode": args.mode,
        "digits_dir": str(digits_dir),
        "bottom_box": bottom_box,
        "result": result,
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)

    print(json.dumps(json_result, indent=2, ensure_ascii=False))
    print()
    print(f"Images debug écrites dans : {out_dir}")
    print(f"Overlay principal : {out_dir / '99_overlay.png'}")


if __name__ == "__main__":
    main()
