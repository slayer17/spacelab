#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpaceLab - backend Flask propre, autonome et robuste.

Objectifs :
- éviter les erreurs de syntaxe / indentation
- garder une route /upload stable
- toujours renvoyer un JSON cohérent
- ne plus bloquer sur "main_card_not_detected" : fallback sur toute l'image
- rester autonome : pas d'import cassant vers bottom.py ou autres fichiers annexes

Dépendances :
    pip install flask opencv-python numpy

Lancement :
    python spacelab.py

Variables utiles :
    PORT=5000 python spacelab.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEBUG_DIR = BASE_DIR / "_debug"
DEBUG_DIR.mkdir(exist_ok=True)

WARP_PATH = str(DEBUG_DIR / "last_warp.jpg")
STANDARD_CARD_W = 200
STANDARD_CARD_H = 300
DIGITS_DIR = BASE_DIR / "digits"
SYMBOLS_DIR = BASE_DIR / "symbols"

CARD_DB_CANDIDATES = [
    BASE_DIR / "cards_db.json",
    BASE_DIR / "card_db.json",
    BASE_DIR / "cards.json",
    BASE_DIR / "references.json",
    BASE_DIR / "cards.js",
    BASE_DIR / "static" / "cards.js",
]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


# -----------------------------------------------------
# UTILITAIRES GÉNÉRAUX
# -----------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = minimum
    return max(minimum, min(maximum, v))


def ensure_color_image(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("image vide")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("quad invalide")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_quad_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    max_width = max(max_width, 2)
    max_height = max(max_height, 2)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def warp_quad(img: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    try:
        return four_point_transform(img, quad)
    except Exception:
        return None


def load_optional_card_db() -> List[Dict[str, Any]]:
    for path in CARD_DB_CANDIDATES:
        if not path.exists():
            continue

        try:
            raw = path.read_text(encoding="utf-8").strip()

            # Cas JSON classique
            if path.suffix.lower() == ".json":
                data = json.loads(raw)
            else:
                # Cas cards.js du type : window.CARDS = [...]
                txt = raw

                if txt.startswith("window.CARDS"):
                    txt = txt.split("=", 1)[1].strip()

                if txt.endswith(";"):
                    txt = txt[:-1].strip()

                data = json.loads(txt)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                for key in ("cards", "items", "data"):
                    if isinstance(data.get(key), list):
                        return data[key]

        except Exception as e:
            print(f"load_optional_card_db ERROR on {path}: {e}")

    return []

CARD_DB = load_optional_card_db()


def _normalize_symbol_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    mapping = {
        "astronaute": "ASTRONAUTE",
        "mecanicien": "MECANICIEN",
        "mécanicien": "MECANICIEN",
        "medecin": "MEDECIN",
        "médecin": "MEDECIN",
        "scientifique": "SCIENTIFIQUE",
    }
    return mapping.get(name, (raw or "INCONNU").upper())


def _normalize_digit_template(img: np.ndarray, target: int = 64) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(th > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = th[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = min((target - 8) / max(w, 1), (target - 8) / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def load_digit_templates() -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not DIGITS_DIR.exists():
        return out
    for path in sorted(DIGITS_DIR.glob('*.png')):
        try:
            digit = int(path.stem)
        except Exception:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        norm = _normalize_digit_template(img)
        if norm is not None:
            out[digit] = norm
    return out


def _normalize_symbol_template(img: np.ndarray, target: int = 64) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=alpha)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(th > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = th[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = min((target - 8) / max(w, 1), (target - 8) / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def load_symbol_templates() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not SYMBOLS_DIR.exists():
        return out
    for path in sorted(SYMBOLS_DIR.glob('*.png')):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        norm = _normalize_symbol_template(img)
        if norm is not None:
            out[_normalize_symbol_name(path.stem)] = norm
    return out


DIGIT_TEMPLATES = load_digit_templates()
SYMBOL_TEMPLATES = load_symbol_templates()


# -----------------------------------------------------
# DÉTECTION DE CARTE
# -----------------------------------------------------

def contour_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if box is None or len(box) != 4:
        return None
    return np.array(box, dtype=np.float32)


def detect_main_card(img: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        img = ensure_color_image(img)
        h, w = img.shape[:2]
        if h < 5 or w < 5:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 60, 160)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0
        image_area = float(w * h)

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < image_area * 0.08:
                continue

            quad = contour_to_quad(cnt)
            if quad is None:
                continue

            x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
            if bw < 20 or bh < 20:
                continue

            rect_area = float(bw * bh)
            fill_ratio = area / max(rect_area, 1.0)
            if fill_ratio < 0.45:
                continue

            score = area * fill_ratio
            if score > best_score:
                best_score = score
                best = {
                    "x": int(x),
                    "y": int(y),
                    "w": int(bw),
                    "h": int(bh),
                    "quad": [[int(px), int(py)] for px, py in order_quad_points(quad)],
                }

        return best
    except Exception:
        return None


# -----------------------------------------------------
# ANALYSE VISUELLE SIMPLE
# -----------------------------------------------------

def crop_rel(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = img.shape[:2]
    xa = clamp_int(round(x1 * w), 0, w - 1)
    ya = clamp_int(round(y1 * h), 0, h - 1)
    xb = clamp_int(round(x2 * w), xa + 1, w)
    yb = clamp_int(round(y2 * h), ya + 1, h)
    crop = img[ya:yb, xa:xb].copy()
    roi = {"x": xa, "y": ya, "w": xb - xa, "h": yb - ya}
    return crop, roi


def image_to_vector(gray: np.ndarray, size: Tuple[int, int] = (16, 16)) -> List[int]:
    if gray is None or gray.size == 0:
        return [0] * (size[0] * size[1])
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return [int(v) for v in small.flatten().tolist()]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def normalize_card_image(img: np.ndarray) -> np.ndarray:
    img = ensure_color_image(img)
    return cv2.resize(img, (STANDARD_CARD_W, STANDARD_CARD_H), interpolation=cv2.INTER_AREA)


def _normalize_binary_crop(gray: np.ndarray, target: int = 64) -> Optional[np.ndarray]:
    if gray is None or gray.size == 0:
        return None
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(th > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = th[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = min((target - 8) / max(w, 1), (target - 8) / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _clip_box_tuple(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    if x + w > max_w:
        w = max_w - x
    if y + h > max_h:
        h = max_h - y
    return x, y, max(1, w), max(1, h)


def _offset_roi(box: Optional[Tuple[int, int, int, int]], dx: int, dy: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None:
        return None
    x, y, w, h = box
    return int(x + dx), int(y + dy), int(w), int(h)


def _find_largest_contour_box(mask: np.ndarray, area_ratio_min: float = 0.05) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or mask.size == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = mask.shape[:2]
    zone_area = float(max(1, h * w))
    candidates = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= zone_area * area_ratio_min:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        candidates.append((area, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x), int(y), int(bw), int(bh)


def _make_light_mask(zone: np.ndarray) -> Optional[np.ndarray]:
    if zone is None or zone.size == 0:
        return None
    if len(zone.shape) == 3:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    else:
        gray = zone.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _find_black_panel_box(bottom_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if bottom_zone is None or bottom_zone.size == 0:
        return None
    gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    zone = gray[:, :max(1, int(w * 0.92))]
    mask = np.zeros_like(zone)
    mask[zone < 150] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = zone.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, zw * zh))
        ratio = bw / float(max(1, bh))
        if area_ratio < 0.10:
            continue
        if bw < zw * 0.35 or bh < zh * 0.38:
            continue
        if ratio < 1.0 or ratio > 5.5:
            continue
        score = (area_ratio * 4.0) + (bw / float(max(1, zw))) + (bh / float(max(1, zh)))
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x), int(y), int(bw), int(bh)


def _find_special_white_panel_box(bottom_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if bottom_zone is None or bottom_zone.size == 0:
        return None
    h, w = bottom_zone.shape[:2]
    zone = bottom_zone[:, :max(1, int(w * 0.70))]
    mask = _make_light_mask(zone)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = zone.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, zw * zh))
        ratio = bw / float(max(1, bh))
        if area_ratio < 0.10:
            continue
        if bw < zw * 0.28 or bh < zh * 0.30:
            continue
        if ratio < 0.80 or ratio > 3.20:
            continue
        score = (area_ratio * 4.0) + (bh / float(max(1, zh)))
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x), int(y), int(bw), int(bh)


def _find_points_badge_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    ph, pw = panel_zone.shape[:2]
    x1 = 0
    x2 = max(1, int(pw * 0.42))
    y1 = int(ph * 0.04)
    y2 = ph
    zone = panel_zone[y1:y2, x1:x2]
    if zone.size == 0:
        return None
    mask = _make_light_mask(zone)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = zone.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, zw * zh))
        ratio = bw / float(max(1, bh))
        if area_ratio < 0.03:
            continue
        if bw < zw * 0.16 or bh < zh * 0.20:
            continue
        if ratio < 0.45 or ratio > 1.80:
            continue
        score = (area_ratio * 4.0) + (bh / float(max(1, zh)))
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x + x1), int(y + y1), int(bw), int(bh)


def _extract_digit_mask_from_badge(badge_crop: np.ndarray, target: int = 64) -> Optional[np.ndarray]:
    if badge_crop is None or badge_crop.size == 0:
        return None
    if len(badge_crop.shape) == 3:
        gray = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = badge_crop.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, white_mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 20:
        return None
    x, y, w, h = cv2.boundingRect(best)
    crop_gray = gray[y:y + h, x:x + w]
    crop_badge = white_mask[y:y + h, x:x + w]
    inner_badge = cv2.erode(crop_badge, np.ones((2, 2), np.uint8), iterations=1)
    badge_pixels = crop_gray[inner_badge > 0]
    if badge_pixels.size == 0:
        return None
    dark_threshold = np.percentile(badge_pixels, 42)
    digit_mask = np.zeros_like(crop_gray, dtype=np.uint8)
    digit_mask[crop_gray < dark_threshold] = 255
    digit_mask = cv2.bitwise_and(digit_mask, inner_badge)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(digit_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    digit = digit_mask[y1:y2, x1:x2]
    if digit.size == 0:
        return None
    h0, w0 = digit.shape[:2]
    scale = min((target - 8) / max(w0, 1), (target - 8) / max(h0, 1))
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _detect_digit_from_badge(badge_crop: np.ndarray) -> Dict[str, Any]:
    result = {
        "digit": 0,
        "raw_digit": 0,
        "score": 0.0,
        "gap": 1.0,
        "found": False,
        "scores": [],
        "mean": 0.0,
        "std": 0.0,
    }
    if badge_crop is None or badge_crop.size == 0:
        return result
    gray = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2GRAY) if len(badge_crop.shape) == 3 else badge_crop.copy()
    result["mean"] = safe_float(np.mean(gray))
    result["std"] = safe_float(np.std(gray))

    digit_mask = _extract_digit_mask_from_badge(badge_crop)
    if digit_mask is None:
        fallback = detect_points_from_bottom(gray)
        fallback["scores"] = []
        return fallback

    if DIGIT_TEMPLATES:
        vec = digit_mask.flatten().astype(np.float32).tolist()
        scored = []
        for digit, templ in DIGIT_TEMPLATES.items():
            score = cosine_similarity(vec, templ.flatten().astype(np.float32).tolist())
            scored.append((float(score), int(digit)))
        scored.sort(reverse=True)
        best_score, best_digit = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        gap = float(best_score - second_score)
        result.update({
            "digit": int(best_digit),
            "raw_digit": int(best_digit),
            "score": float(best_score),
            "gap": gap,
            "found": bool(best_score >= 0.72 or (best_score >= 0.60 and gap >= 0.02)),
            "scores": [{"digit": int(d), "score": float(s)} for s, d in scored[:5]],
        })
        if not result["found"]:
            result["digit"] = 0
        return result

    fallback = detect_points_from_bottom(gray)
    fallback["scores"] = []
    return fallback


def _find_slash_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    gray = cv2.cvtColor(panel_zone, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(th, 1, np.pi / 180, threshold=20, minLineLength=max(10, panel_zone.shape[1] // 7), maxLineGap=6)
    if lines is None:
        return None
    best = None
    best_len = -1.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line.tolist())
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2 and abs(dy) < 2:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if not (20 <= angle <= 70):
            continue
        length = math.hypot(dx, dy)
        x = min(x1, x2)
        y = min(y1, y2)
        bw = abs(dx) + 1
        bh = abs(dy) + 1
        if length > best_len:
            best_len = length
            best = (x, y, bw, bh)
    return best


def _find_right_icon_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.70)
    zone = panel_zone[:, x1:pw]
    mask = _make_light_mask(zone)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = zone.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, zw * zh))
        ratio = bw / float(max(1, bh))
        if area_ratio < 0.02:
            continue
        if bw < zw * 0.18 or bh < zh * 0.18:
            continue
        if ratio < 0.45 or ratio > 1.80:
            continue
        if (x + bw) >= (zw - 1):
            continue
        score = (area_ratio * 4.0) + (bh / float(max(1, zh)))
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x + x1), int(y), int(bw), int(bh)


def _find_bottom_line_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.52)
    y1 = int(ph * 0.68)
    zone = panel_zone[y1:ph, x1:pw]
    mask = _make_light_mask(zone)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = zone.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, zw * zh))
        ratio = bw / float(max(1, bh))
        if area_ratio < 0.015:
            continue
        if bw < zw * 0.24 or bh > zh * 0.34:
            continue
        if ratio < 2.2:
            continue
        if (y + bh) >= (zh - 1):
            continue
        score = (ratio * 1.4) + (area_ratio * 6.0)
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return int(x + x1), int(y + y1), int(bw), int(bh)


def analyze_bottom_zone(bottom_zone: np.ndarray) -> Dict[str, Any]:
    result = {
        "layout": "UNKNOWN",
        "points": 0,
        "raw_points": 0,
        "points_score": 0.0,
        "points_gap": 1.0,
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
        "digit_scores": [],
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
        gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
        result.update(detect_points_from_bottom(gray))
        result["layout"] = "UNKNOWN"
        return result

    result["panel_box"] = panel_box
    px, py, pw, ph = panel_box
    panel_zone = bottom_zone[py:py + ph, px:px + pw]

    points_box_local = _find_points_badge_box(panel_zone)
    if points_box_local is not None:
        bx, by, bw, bh = points_box_local
        badge_crop = panel_zone[by:by + bh, bx:bx + bw]
        digit = _detect_digit_from_badge(badge_crop)
        result["raw_points"] = int(digit.get("raw_digit", 0))
        result["points_score"] = float(digit.get("score", 0.0))
        result["points_gap"] = float(digit.get("gap", 1.0))
        result["digit_scores"] = digit.get("scores", [])
        if digit.get("found"):
            result["points"] = int(digit.get("digit", 0))
        result["points_box"] = _offset_roi(points_box_local, px, py)

    slash_box_local = _find_slash_box(panel_zone)
    if slash_box_local is not None:
        result["has_slash"] = True
        result["slash_box"] = _offset_roi(slash_box_local, px, py)

    right_box_local = _find_right_icon_box(panel_zone)
    if right_box_local is not None:
        result["has_right_icon"] = True
        result["right_icon_box"] = _offset_roi(right_box_local, px, py)

    line_box_local = _find_bottom_line_box(panel_zone)
    if line_box_local is not None:
        result["has_bottom_line"] = True
        result["bottom_line_box"] = _offset_roi(line_box_local, px, py)

    if result["points"] and not result["has_slash"] and not result["has_right_icon"]:
        result["layout"] = "NUMBER_ONLY"
    elif result["points"] and result["has_slash"] and result["has_right_icon"] and result["has_bottom_line"]:
        result["layout"] = "NUMBER_ICON_LINE"
    elif result["points"] and result["has_slash"] and result["has_right_icon"]:
        result["layout"] = "NUMBER_ICON"
    elif result["has_right_icon"]:
        result["layout"] = "ICON_ONLY"
    else:
        result["layout"] = "BLACK_PANEL"

    return result


def detect_dominant_color_name(img: np.ndarray) -> Dict[str, Any]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (s > 40) & (v > 40)
    if not np.any(mask):
        return {
            "detected": "INCONNU",
            "debug": {"BLEU": 0, "VERT": 0, "JAUNE": 0, "ROUGE": 0},
            "mean": safe_float(np.mean(v)),
            "std": safe_float(np.std(v)),
            "color": [safe_float(np.mean(img[:, :, c])) for c in range(3)],
        }

    hue = h[mask]
    scores = {
        "ROUGE": int(np.sum((hue <= 10) | (hue >= 170))),
        "JAUNE": int(np.sum((hue >= 16) & (hue <= 40))),
        "VERT": int(np.sum((hue >= 41) & (hue <= 85))),
        "BLEU": int(np.sum((hue >= 86) & (hue <= 135))),
    }
    detected = max(scores, key=scores.get)
    return {
        "detected": detected,
        "debug": scores,
        "mean": safe_float(np.mean(v)),
        "std": safe_float(np.std(v)),
        "color": [safe_float(np.mean(img[:, :, c])) for c in range(3)],
    }


def detect_points_from_bottom(bottom_gray: np.ndarray) -> Dict[str, Any]:
    if bottom_gray is None or bottom_gray.size == 0:
        return {"digit": 0, "raw_digit": 0, "score": 0.0, "gap": 1.0, "found": False, "mean": 0.0, "std": 0.0}

    blur = cv2.GaussianBlur(bottom_gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    components = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= max(12, (bottom_gray.shape[0] * bottom_gray.shape[1]) // 250):
            components += 1

    digit = 0
    score = 0.0
    if components == 1:
        digit = 1
        score = 0.35
    elif components == 2:
        digit = 3
        score = 0.42
    elif components == 3:
        digit = 8
        score = 0.30

    return {
        "digit": int(digit),
        "raw_digit": int(digit),
        "score": float(score),
        "gap": float(max(0.0, 1.0 - score)),
        "found": bool(score >= 0.40),
        "mean": safe_float(np.mean(bottom_gray)),
        "std": safe_float(np.std(bottom_gray)),
    }


def detect_bottom_layout(bottom_gray: np.ndarray, symbol_gray: np.ndarray) -> Dict[str, Any]:
    if bottom_gray is None or bottom_gray.size == 0:
        return {
            "layout": "UNKNOWN",
            "has_slash": False,
            "has_right_icon": False,
            "has_bottom_line": False,
            "has_special_white_panel": False,
            "points": 0,
            "raw_points": 0,
            "points_score": 0.0,
            "points_gap": 1.0,
            "range": "",
            "target": "",
            "ref_card_id": None,
            "ref_score": 0.0,
            "ref_gap": 1.0,
        }

    h, w = bottom_gray.shape[:2]
    blur = cv2.GaussianBlur(bottom_gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lines = cv2.HoughLinesP(th, 1, np.pi / 180, threshold=20, minLineLength=max(10, w // 7), maxLineGap=5)
    has_slash = False
    has_bottom_line = False

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line.tolist())
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 2 and abs(dy) < 2:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if 20 <= angle <= 70:
                has_slash = True
            if angle <= 10 and max(y1, y2) >= int(h * 0.72):
                has_bottom_line = True

    right_zone = th[:, int(w * 0.72):]
    has_right_icon = bool(np.mean(right_zone > 0) > 0.04)
    has_special_white_panel = bool(np.mean(bottom_gray > 220) > 0.22)
    points = detect_points_from_bottom(bottom_gray)

    if has_right_icon and points["digit"] > 0:
        layout = "NUMBER_ICON"
    elif points["digit"] > 0:
        layout = "NUMBER_ONLY"
    elif has_right_icon:
        layout = "ICON_ONLY"
    else:
        layout = "UNKNOWN"

    return {
        "layout": layout,
        "has_slash": has_slash,
        "has_right_icon": has_right_icon,
        "has_bottom_line": has_bottom_line,
        "has_special_white_panel": has_special_white_panel,
        "points": int(points["digit"]),
        "raw_points": int(points["raw_digit"]),
        "points_score": float(points["score"]),
        "points_gap": float(points["gap"]),
        "range": "GLOBAL" if has_slash else "",
        "target": "",
        "ref_card_id": None,
        "ref_score": float(points["score"]),
        "ref_gap": float(points["gap"]),
    }


def detect_symbol_name(symbol_gray: np.ndarray) -> Dict[str, Any]:
    vector = image_to_vector(symbol_gray, size=(16, 16))
    mean = safe_float(np.mean(symbol_gray))
    std = safe_float(np.std(symbol_gray))

    best_name = "INCONNU"
    best_score = 0.0
    runner_up = {"name": None, "score": 0.0}

    for item in CARD_DB:
        ref = item.get("symbol_vector")
        name = item.get("symbol_name") or item.get("symbol") or item.get("target") or "INCONNU"
        if isinstance(ref, list) and len(ref) == len(vector):
            score = cosine_similarity(vector, ref)
            if score > best_score:
                runner_up = {"name": best_name, "score": best_score}
                best_name = str(name)
                best_score = float(score)

    threshold_mode = "accepted" if best_score >= 0.65 else "weak" if best_score >= 0.45 else "unknown"

    return {
        "name": best_name,
        "raw_name": best_name,
        "score": float(best_score),
        "gap": float(max(0.0, 1.0 - best_score)),
        "mean": mean,
        "std": std,
        "mode": "card_db" if CARD_DB else "none",
        "threshold_mode": threshold_mode,
        "runner_up": runner_up,
        "top_candidates": [],
        "winner_references": [],
    }


def compute_signature(warped: np.ndarray) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        warped = ensure_color_image(warped)
        h, w = warped.shape[:2]
        if h < 20 or w < 20:
            return None, []

        color_crop, roi_color = crop_rel(warped, 0.00, 0.00, 0.38, 0.18)
        symbol_crop, roi_symbol = crop_rel(warped, 0.00, 0.14, 0.24, 0.34)
        bottom_crop, roi_bottom = crop_rel(warped, 0.00, 0.78, 0.55, 1.00)
        global_crop, roi_global = crop_rel(warped, 0.00, 0.00, 1.00, 1.00)

        rois = [
            {"type": "COLOR", **roi_color},
            {"type": "SYMBOL", **roi_symbol},
            {"type": "BOTTOM", **roi_bottom},
            {"type": "GLOBAL", **roi_global},
        ]

        color_info = detect_dominant_color_name(color_crop)
        symbol_gray = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY)
        bottom_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
        global_gray = cv2.cvtColor(global_crop, cv2.COLOR_BGR2GRAY)

        symbol_info = detect_symbol_name(symbol_gray)
        bottom_layout = detect_bottom_layout(bottom_gray, symbol_gray)
        points_info = detect_points_from_bottom(bottom_gray)

        signature = {
            "color": color_info,
            "symbol": symbol_info,
            "bottom": {
                "vector": image_to_vector(bottom_gray, size=(16, 16)),
                "mean": safe_float(np.mean(bottom_gray)),
                "std": safe_float(np.std(bottom_gray)),
            },
            "global": {
                "vector": image_to_vector(global_gray, size=(16, 16)),
                "mean": safe_float(np.mean(global_gray)),
                "std": safe_float(np.std(global_gray)),
            },
            "points": points_info,
            "bottom_layout": bottom_layout,
        }

        return signature, rois
    except Exception:
        return None, []


# -----------------------------------------------------
# MATCH FINAL
# -----------------------------------------------------

def _score_candidate(sig: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    score = 0.0
    details: List[str] = []

    color_name = (sig.get("color", {}) or {}).get("detected")
    symbol_name = (sig.get("symbol", {}) or {}).get("name")
    layout = (sig.get("bottom_layout", {}) or {}).get("layout")
    points = (sig.get("bottom_layout", {}) or {}).get("points")

    expected_color = item.get("color_name") or item.get("color") or item.get("couleur")
    expected_symbol = item.get("symbol_name") or item.get("symbol") or item.get("target")
    expected_layout = item.get("bottom_layout") or item.get("layout")
    expected_points = item.get("points")

    if expected_color and color_name == expected_color:
        score += 30
        details.append("color_exact")
    if expected_symbol and symbol_name == expected_symbol:
        score += 25
        details.append("symbol_exact")
    if expected_layout and layout == expected_layout:
        score += 10
        details.append("layout_exact")
    if expected_points is not None and int(points or 0) == int(expected_points):
        score += 8
        details.append("points_exact")

    ref_global = item.get("global_vector")
    sig_global = (sig.get("global", {}) or {}).get("vector")
    global_visual = 0.0
    if isinstance(ref_global, list) and isinstance(sig_global, list) and len(ref_global) == len(sig_global):
        global_visual = cosine_similarity(sig_global, ref_global)
        score += 20 * global_visual
        details.append(f"global_visual={global_visual:.3f}")

    ref_bottom = item.get("bottom_vector")
    sig_bottom = (sig.get("bottom", {}) or {}).get("vector")
    bottom_visual = 0.0
    if isinstance(ref_bottom, list) and isinstance(sig_bottom, list) and len(ref_bottom) == len(sig_bottom):
        bottom_visual = cosine_similarity(sig_bottom, ref_bottom)
        score += 15 * bottom_visual
        details.append(f"bottom_visual={bottom_visual:.3f}")

    return {
        "card_id": item.get("card_id") or item.get("id"),
        "score": float(score),
        "details": details,
        "global_visual": float(global_visual),
        "bottom_visual": float(bottom_visual),
        "expected_bottom": {
            "layout": expected_layout,
            "points": expected_points,
            "target": expected_symbol,
            "range": item.get("range", ""),
            "has_special_white_panel": bool(item.get("has_special_white_panel", False)),
            "has_slash": bool(item.get("has_slash", False)),
            "has_right_icon": bool(item.get("has_right_icon", False)),
            "has_bottom_line": bool(item.get("has_bottom_line", False)),
        },
    }


def resolve_final_card(sig: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    color_name = (sig.get("color", {}) or {}).get("detected") or "INCONNU"
    symbol_name = (sig.get("symbol", {}) or {}).get("name") or "INCONNU"
    layout = (sig.get("bottom_layout", {}) or {}).get("layout") or "UNKNOWN"
    points = int((sig.get("bottom_layout", {}) or {}).get("points") or 0)

    if not CARD_DB:
        final_card_id = None
        if color_name != "INCONNU" and symbol_name != "INCONNU" and points > 0:
            final_card_id = f"{color_name}_{symbol_name}_{points}"
        return {
            "candidate_cards": [],
            "color_name": color_name,
            "symbol_name": symbol_name,
            "symbol_source": (sig.get("symbol", {}) or {}).get("threshold_mode"),
            "bottom_layout": layout,
            "points": points,
            "final_card_id": final_card_id,
            "final_status": "partial" if final_card_id else "unknown",
            "final_score": 0.0,
            "final_gap": 0.0,
            "reason": "no_card_db_loaded",
        }

    candidates: List[Dict[str, Any]] = []
    for item in CARD_DB:
        candidate = _score_candidate(sig, item)
        if candidate.get("card_id"):
            candidates.append(candidate)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        return {
            "candidate_cards": [],
            "color_name": color_name,
            "symbol_name": symbol_name,
            "symbol_source": (sig.get("symbol", {}) or {}).get("threshold_mode"),
            "bottom_layout": layout,
            "points": points,
            "final_card_id": None,
            "final_status": "unknown",
            "final_score": 0.0,
            "final_gap": 0.0,
            "reason": "no_candidate",
        }

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {"score": 0.0}
    gap = float(best["score"] - second["score"])

    if best["score"] >= 55 and gap >= 8:
        status = "accepted"
        reason = "strong_unique_match"
    elif best["score"] >= 40:
        status = "candidate"
        reason = "usable_match"
    else:
        status = "unknown"
        reason = "weak_match"

    return {
        "candidate_cards": candidates[:5],
        "color_name": color_name,
        "symbol_name": symbol_name,
        "symbol_source": (sig.get("symbol", {}) or {}).get("threshold_mode"),
        "bottom_layout": layout,
        "points": points,
        "final_card_id": best.get("card_id"),
        "final_status": status,
        "final_score": float(best["score"]),
        "final_gap": gap,
        "reason": reason,
    }


# -----------------------------------------------------
# RÉPONSES JSON STABLES
# -----------------------------------------------------

def empty_upload_payload(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "rects": [],
        "signature": None,
        "rois": [],
        "card_match": None,
        "final_card_id": None,
        "final_status": None,
        "final_score": 0.0,
        "final_gap": 0.0,
        "color_name": None,
        "symbol_name": None,
        "bottom_layout": None,
        "points": None,
        "error": error,
    }
# -----------------------------------------------------
# BOARD MODE
# -----------------------------------------------------

BOARD_SLOT_LAYOUT = [
    ("top_left",     0.20, 0.03, 0.33, 0.28),
    ("top_middle",   0.42, 0.03, 0.55, 0.28),
    ("top_right",    0.65, 0.03, 0.78, 0.28),

    ("left_outer",   0.08, 0.31, 0.19, 0.62),
    ("middle_left",  0.28, 0.32, 0.40, 0.61),
    ("middle_right", 0.53, 0.32, 0.65, 0.60),
    ("right_outer",  0.78, 0.31, 0.89, 0.62),

    ("bottom_left",  0.18, 0.70, 0.31, 0.98),
    ("bottom_middle",0.44, 0.68, 0.54, 0.96),
    ("bottom_right", 0.66, 0.69, 0.79, 0.98),
]


def _board_crop_box(img: np.ndarray, xr1: float, yr1: float, xr2: float, yr2: float) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = img.shape[:2]
    x1 = clamp_int(int(round(w * xr1)), 0, w - 1)
    y1 = clamp_int(int(round(h * yr1)), 0, h - 1)
    x2 = clamp_int(int(round(w * xr2)), x1 + 1, w)
    y2 = clamp_int(int(round(h * yr2)), y1 + 1, h)
    crop = img[y1:y2, x1:x2].copy()
    return crop, {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _box_from_margins(box: Dict[str, int], left: float, top: float, right: float, bottom: float) -> Dict[str, int]:
    w = int(box["w"])
    h = int(box["h"])
    x = int(round(left * w))
    y = int(round(top * h))
    x2 = int(round(w * (1.0 - right)))
    y2 = int(round(h * (1.0 - bottom)))
    x2 = max(x + 1, x2)
    y2 = max(y + 1, y2)
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def _default_board_inner_box(slot_id: str, box: Dict[str, int]) -> Dict[str, int]:
    if slot_id.startswith("top_"):
        return _box_from_margins(box, left=0.16, top=0.00, right=0.20, bottom=0.05)
    if slot_id.startswith("bottom_"):
        return _box_from_margins(box, left=0.12, top=0.00, right=0.12, bottom=0.06)
    if slot_id in ("left_outer", "right_outer"):
        return _box_from_margins(box, left=0.05, top=0.00, right=0.05, bottom=0.04)
    return _box_from_margins(box, left=0.12, top=0.00, right=0.12, bottom=0.05)


def _projection_bounds(values: np.ndarray, threshold_ratio: float = 0.35, pad: int = 6) -> Optional[Tuple[int, int]]:
    if values is None or len(values) == 0:
        return None

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None

    win = max(5, int(arr.size * 0.05))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smooth = np.convolve(arr, kernel, mode="same")

    vmax = float(np.max(smooth))
    vmed = float(np.median(smooth))
    if vmax <= 1e-6:
        return None

    threshold = vmed + (vmax - vmed) * threshold_ratio
    idx = np.where(smooth >= threshold)[0]
    if idx.size == 0:
        return None

    start = max(0, int(idx[0]) - pad)
    end = min(arr.size - 1, int(idx[-1]) + pad)
    return start, end


def _largest_contour_box(mask: np.ndarray) -> Optional[Dict[str, int]]:
    if mask is None or mask.size == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    mh, mw = mask.shape[:2]
    full_area = float(max(1, mh * mw))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 30:
            continue

        area = float(cv2.contourArea(cnt))
        rect_area = float(max(1, w * h))
        fill_ratio = area / rect_area
        area_ratio = rect_area / full_area
        aspect = w / float(max(1, h))

        if fill_ratio < 0.25:
            continue
        if area_ratio < 0.10 or area_ratio > 0.92:
            continue
        if aspect < 0.35 or aspect > 0.95:
            continue

        score = area * (0.50 + fill_ratio)
        if score > best_score:
            best_score = score
            best = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return best


def _merge_local_boxes(primary: Dict[str, int], secondary: Dict[str, int], alpha: float = 0.65) -> Dict[str, int]:
    beta = 1.0 - alpha
    x = int(round(primary["x"] * alpha + secondary["x"] * beta))
    y = int(round(primary["y"] * alpha + secondary["y"] * beta))
    w = int(round(primary["w"] * alpha + secondary["w"] * beta))
    h = int(round(primary["h"] * alpha + secondary["h"] * beta))
    return {"x": x, "y": y, "w": max(1, w), "h": max(1, h)}


def _clip_local_box(local_box: Dict[str, int], crop_shape: Tuple[int, int, int]) -> Dict[str, int]:
    h, w = crop_shape[:2]
    x = clamp_int(local_box.get("x", 0), 0, max(0, w - 1))
    y = clamp_int(local_box.get("y", 0), 0, max(0, h - 1))
    bw = clamp_int(local_box.get("w", 1), 1, max(1, w - x))
    bh = clamp_int(local_box.get("h", 1), 1, max(1, h - y))
    return {"x": x, "y": y, "w": bw, "h": bh}


def _local_box_to_abs(base_box: Dict[str, int], local_box: Dict[str, int]) -> Dict[str, int]:
    return {
        "x": int(base_box["x"] + local_box["x"]),
        "y": int(base_box["y"] + local_box["y"]),
        "w": int(local_box["w"]),
        "h": int(local_box["h"]),
    }


def _refine_board_card_box(slot_id: str, crop: np.ndarray, base_box: Dict[str, int]) -> Dict[str, Any]:
    default_box = _default_board_inner_box(slot_id, base_box)
    default_box = _clip_local_box(default_box, crop.shape)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    pad_x = max(4, int(base_box["w"] * 0.03))
    pad_y = max(4, int(base_box["h"] * 0.03))
    inner = edges[pad_y:max(pad_y + 1, edges.shape[0] - pad_y), pad_x:max(pad_x + 1, edges.shape[1] - pad_x)]
    projection_box = None

    if inner.size > 0:
        xproj = np.mean(inner > 0, axis=0).astype(np.float32)
        yproj = np.mean(inner > 0, axis=1).astype(np.float32)
        xb = _projection_bounds(xproj, threshold_ratio=0.33, pad=max(4, int(base_box["w"] * 0.02)))
        yb = _projection_bounds(yproj, threshold_ratio=0.33, pad=max(4, int(base_box["h"] * 0.02)))
        if xb and yb:
            projection_box = {
                "x": int(xb[0] + pad_x),
                "y": int(yb[0] + pad_y),
                "w": int(xb[1] - xb[0] + 1),
                "h": int(yb[1] - yb[0] + 1),
            }
            projection_box = _clip_local_box(projection_box, crop.shape)

    foreground_mask = np.zeros_like(gray)
    foreground_mask[(gray < 215) | (hsv[:, :, 1] > 45)] = 255
    k = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, k)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, k)
    contour_box = _largest_contour_box(foreground_mask)

    candidate = None
    if projection_box and contour_box:
        candidate = _merge_local_boxes(projection_box, contour_box, alpha=0.55)
    elif projection_box:
        candidate = projection_box
    elif contour_box:
        candidate = contour_box

    if candidate is not None:
        candidate = _clip_local_box(candidate, crop.shape)
        aspect = candidate["w"] / float(max(1, candidate["h"]))
        area_ratio = (candidate["w"] * candidate["h"]) / float(max(1, base_box["w"] * base_box["h"]))
        if 0.35 <= aspect <= 0.95 and 0.12 <= area_ratio <= 0.90:
            local_box = _merge_local_boxes(candidate, default_box, alpha=0.72)
        else:
            local_box = default_box
    else:
        local_box = default_box

    local_box = _clip_local_box(local_box, crop.shape)
    abs_box = _local_box_to_abs(base_box, local_box)
    tight_crop = crop[local_box["y"]:local_box["y"] + local_box["h"], local_box["x"]:local_box["x"] + local_box["w"]].copy()

    return {
        "local_box": local_box,
        "abs_box": abs_box,
        "tight_crop": tight_crop,
        "default_local_box": default_box,
        "projection_local_box": projection_box,
        "contour_local_box": contour_box,
    }


def _board_slot_occupied(crop: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    if crop is None or crop.size == 0:
        return False, {"sat_ratio": 0.0, "edge_ratio": 0.0, "dark_ratio": 0.0}

    h, w = crop.shape[:2]
    pad_x = max(1, int(w * 0.08))
    pad_y = max(1, int(h * 0.08))
    inner = crop[pad_y:h - pad_y, pad_x:w - pad_x]
    if inner.size == 0:
        inner = crop

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 180)

    area = float(max(1, inner.shape[0] * inner.shape[1]))
    sat_ratio = float(np.count_nonzero(hsv[:, :, 1] > 45)) / area
    edge_ratio = float(np.count_nonzero(edges > 0)) / area
    dark_ratio = float(np.count_nonzero(gray < 180)) / area

    occupied = (
        sat_ratio >= 0.05 or
        edge_ratio >= 0.025 or
        dark_ratio >= 0.18
    )

    return occupied, {
        "sat_ratio": float(sat_ratio),
        "edge_ratio": float(edge_ratio),
        "dark_ratio": float(dark_ratio),
    }


def _extract_ref_scan(card_item: Dict[str, Any]) -> Dict[str, Any]:
    return (((card_item or {}).get("signature") or {}).get("scan") or {})


def _match_card_from_signature(sig: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not sig or not CARD_DB:
        return None

    query_color = (((sig.get("color") or {}).get("detected")) or "").upper()
    query_symbol = ((((sig.get("symbol") or {}).get("name")) or ((sig.get("symbol") or {}).get("raw_name")) or "")).upper()
    query_points = int((((sig.get("bottom_layout") or {}).get("points")) or 0))
    query_layout = (((sig.get("bottom_layout") or {}).get("layout")) or "").upper()

    query_global_vec = (((sig.get("global") or {}).get("vector")) or [])
    query_bottom_vec = (((sig.get("bottom") or {}).get("vector")) or [])

    candidates = []

    for item in CARD_DB:
        card_id = item.get("id")
        if not card_id:
            continue

        ref_scan = _extract_ref_scan(item)

        ref_color = str(
            item.get("couleur")
            or item.get("color")
            or item.get("color_name")
            or ((ref_scan.get("color") or {}).get("detected"))
            or ""
        ).upper()

        ref_symbol = str(
            item.get("symbol")
            or item.get("symbol_name")
            or ""
        ).upper()

        ref_points = int(item.get("points") or ((ref_scan.get("bottom_layout") or {}).get("points")) or 0)
        ref_layout = str(
            ((ref_scan.get("bottom_layout") or {}).get("layout"))
            or item.get("layout")
            or item.get("bottom_layout")
            or ""
        ).upper()

        ref_global_vec = (((ref_scan.get("global") or {}).get("vector")) or [])
        ref_bottom_vec = (((ref_scan.get("bottom") or {}).get("vector")) or [])

        score = 0.0
        details = []

        if query_color and ref_color and query_color == ref_color:
            score += 25.0
            details.append("color_exact")

        if query_symbol and query_symbol != "INCONNU" and ref_symbol and query_symbol == ref_symbol:
            score += 22.0
            details.append("symbol_exact")

        if query_points > 0 and ref_points > 0 and query_points == ref_points:
            score += 10.0
            details.append("points_exact")

        if query_layout and ref_layout and query_layout == ref_layout:
            score += 8.0
            details.append("layout_exact")

        global_sim = 0.0
        if query_global_vec and ref_global_vec and len(query_global_vec) == len(ref_global_vec):
            global_sim = cosine_similarity(query_global_vec, ref_global_vec)
            score += 18.0 * global_sim
            details.append(f"global={global_sim:.3f}")

        bottom_sim = 0.0
        if query_bottom_vec and ref_bottom_vec and len(query_bottom_vec) == len(ref_bottom_vec):
            bottom_sim = cosine_similarity(query_bottom_vec, ref_bottom_vec)
            score += 14.0 * bottom_sim
            details.append(f"bottom={bottom_sim:.3f}")

        candidates.append({
            "card_id": card_id,
            "score": float(score),
            "details": details,
            "global_similarity": float(global_sim),
            "bottom_similarity": float(bottom_sim),
            "color": ref_color,
            "symbol": ref_symbol,
            "points": ref_points,
            "layout": ref_layout,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {"score": 0.0}
    gap = float(best["score"] - second["score"])

    status = "accepted" if best["score"] >= 28.0 and gap >= 3.0 else "candidate" if best["score"] >= 18.0 else "unknown"

    return {
        "final_card_id": best["card_id"] if status != "unknown" else None,
        "final_status": status,
        "final_score": float(best["score"]),
        "final_gap": gap,
        "candidates": candidates[:5],
    }


def _analyze_single_slot(slot_id: str, crop: np.ndarray, box: Dict[str, int]) -> Dict[str, Any]:
    occupied, metrics = _board_slot_occupied(crop)
    refined = _refine_board_card_box(slot_id, crop, box)
    tight_box = refined["abs_box"]

    slot = {
        "slot_id": slot_id,
        "x": tight_box["x"],
        "y": tight_box["y"],
        "w": tight_box["w"],
        "h": tight_box["h"],
        "occupied": bool(occupied),
        "metrics": metrics,
        "signature": None,
        "match": None,
        "raw_box": box,
        "local_box": refined["local_box"],
    }

    if not occupied:
        return slot

    tight_crop = refined.get("tight_crop")
    if tight_crop is None or tight_crop.size == 0:
        tight_crop = crop.copy()

    rect = detect_main_card(tight_crop)
    if rect is None:
        h, w = tight_crop.shape[:2]
        rect = {
            "x": 0,
            "y": 0,
            "w": int(w),
            "h": int(h),
            "quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        }

    quad = np.array(rect["quad"], dtype=np.float32)
    warped = warp_quad(tight_crop, quad)
    if warped is None or warped.size == 0:
        warped = tight_crop.copy()

    sig, rois = compute_signature(warped)
    slot["signature"] = sig
    slot["rois"] = rois

    if sig is not None:
        slot["match"] = _match_card_from_signature(sig)

    return slot


def analyze_board(img: np.ndarray) -> Dict[str, Any]:
    slots = []
    board_matches = []
    rects = []

    for slot_id, xr1, yr1, xr2, yr2 in BOARD_SLOT_LAYOUT:
        crop, box = _board_crop_box(img, xr1, yr1, xr2, yr2)
        slot = _analyze_single_slot(slot_id, crop, box)
        slots.append(slot)

        rects.append({
            "x": slot["x"],
            "y": slot["y"],
            "w": slot["w"],
            "h": slot["h"],
            "slot_id": slot_id,
        })

        if slot.get("occupied") and slot.get("match"):
            board_matches.append({
                "slot_id": slot_id,
                "final_card_id": (slot["match"] or {}).get("final_card_id"),
                "final_status": (slot["match"] or {}).get("final_status"),
                "final_score": (slot["match"] or {}).get("final_score", 0.0),
                "final_gap": (slot["match"] or {}).get("final_gap", 0.0),
                "signature": slot.get("signature"),
                "candidates": (slot["match"] or {}).get("candidates", []),
            })

    return {
        "board_analysis": {
            "slots": slots,
            "slots_count": len(slots),
            "occupied_count": sum(1 for s in slots if s.get("occupied")),
        },
        "board_matches": board_matches,
        "rects": rects,
    }

# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------

@app.get("/")
def index():
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return (
        "<h1>SpaceLab backend OK</h1>"
        "<p>Utilise <code>POST /upload</code> avec un champ fichier <code>image</code>.</p>"
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "spacelab",
            "card_db_loaded": bool(CARD_DB),
            "card_db_size": len(CARD_DB),
        }
    )


@app.post("/upload")
def upload():
    try:
        if "image" not in request.files:
            return jsonify(empty_upload_payload("missing_image_field"))

        file = request.files["image"]
        if file is None or not getattr(file, "filename", ""):
            return jsonify(empty_upload_payload("empty_upload"))

        mode = str(request.form.get("mode") or "BOARD").upper()

        data = np.frombuffer(file.read(), np.uint8)
        if data.size == 0:
            return jsonify(empty_upload_payload("empty_file"))

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(empty_upload_payload("image_decode_failed"))

        img = ensure_color_image(img)

        # -----------------------------
        # MODE BOARD
        # -----------------------------
        if mode == "BOARD":
            board_result = analyze_board(img)
            return jsonify({
                "rects": board_result["rects"],
                "signature": None,
                "rois": [],
                "card_match": None,
                "final_card_id": None,
                "final_status": None,
                "final_score": 0.0,
                "final_gap": 0.0,
                "color_name": None,
                "symbol_name": None,
                "bottom_layout": None,
                "points": None,
                "board_analysis": board_result["board_analysis"],
                "board_matches": board_result["board_matches"],
                "error": None,
            })

        # -----------------------------
        # MODE SINGLE CARD
        # -----------------------------
        h, w = img.shape[:2]
        rect = detect_main_card(img)

        if rect is None:
            rect = {
                "x": 0,
                "y": 0,
                "w": int(w),
                "h": int(h),
                "quad": [
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1],
                ],
            }

        quad = np.array(rect["quad"], dtype=np.float32)
        warped = warp_quad(img, quad)
        if warped is None or warped.size == 0:
            warped = img.copy()

        try:
            cv2.imwrite(WARP_PATH, warped)
        except Exception:
            pass

        sig, rois = compute_signature(warped)
        card_match = resolve_final_card(sig) if sig is not None else None

        return jsonify({
            "rects": [rect],
            "signature": sig,
            "rois": rois,
            "card_match": card_match,
            "final_card_id": (card_match or {}).get("final_card_id"),
            "final_status": (card_match or {}).get("final_status"),
            "final_score": float((card_match or {}).get("final_score", 0.0)),
            "final_gap": float((card_match or {}).get("final_gap", 0.0)),
            "color_name": (card_match or {}).get("color_name") or (((sig or {}).get("color") or {}).get("detected")),
            "symbol_name": (card_match or {}).get("symbol_name") or (((sig or {}).get("symbol") or {}).get("name")),
            "bottom_layout": (card_match or {}).get("bottom_layout") or (((sig or {}).get("bottom_layout") or {}).get("layout")),
            "points": (card_match or {}).get("points") if card_match else (((sig or {}).get("bottom_layout") or {}).get("points")),
            "board_analysis": None,
            "board_matches": [],
            "error": None,
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        payload = empty_upload_payload(f"upload_exception: {e}")
        payload["board_analysis"] = None
        payload["board_matches"] = []
        return jsonify(payload)


# -----------------------------------------------------
# CARD OVERRIDES - ROI FIX
# -----------------------------------------------------

def _roi_dict(name: str, box: Tuple[int, int, int, int]) -> Dict[str, int]:
    x, y, w, h = box
    return {"type": name, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _crop_box(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    x, y, w, h = _clip_box_tuple(x, y, w, h, img.shape[1], img.shape[0])
    return img[y:y+h, x:x+w].copy()


def _find_symbol_box(card_norm: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    h, w = card_norm.shape[:2]
    sx1, sy1, sx2, sy2 = 0, int(h * 0.12), int(w * 0.32), int(h * 0.38)
    search = card_norm[sy1:sy2, sx1:sx2]
    if search.size == 0:
        box = _clip_box_tuple(0, int(h * 0.14), int(w * 0.24), int(h * 0.20), w, h)
        return box, _crop_box(card_norm, box)

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Chercher une forme contrastée dans la zone symbole plutôt que garder un crop fixe aveugle.
    th_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th_light = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(th_dark, th_light)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    sh, sw = gray.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if bw < max(10, int(sw * 0.18)) or bh < max(10, int(sh * 0.18)):
            continue
        if x + bw >= sw - 1 or y + bh >= sh - 1:
            continue
        ratio = bw / float(max(1, bh))
        if ratio < 0.45 or ratio > 1.75:
            continue
        area_ratio = area / float(max(1, sw * sh))
        score = area_ratio * 4.0 + (bh / float(max(1, sh)))
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        box = _clip_box_tuple(0, int(h * 0.14), int(w * 0.24), int(h * 0.20), w, h)
        return box, _crop_box(card_norm, box)

    x, y, bw, bh = best
    pad_x = max(3, int(bw * 0.18))
    pad_y = max(3, int(bh * 0.18))
    box = _clip_box_tuple(sx1 + x - pad_x, sy1 + y - pad_y, bw + pad_x * 2, bh + pad_y * 2, w, h)
    return box, _crop_box(card_norm, box)


def _build_bottom_rois(card_norm: np.ndarray) -> Tuple[Dict[str, Any], List[Dict[str, int]], np.ndarray]:
    h, w = card_norm.shape[:2]
    bx1, by1, bx2, by2 = 0, int(h * 0.82), int(w * 0.55), h
    bottom_box = _clip_box_tuple(bx1, by1, bx2 - bx1, by2 - by1, w, h)
    bottom_crop = _crop_box(card_norm, bottom_box)
    bottom_info = analyze_bottom_zone(bottom_crop)

    rois = [_roi_dict('BOTTOM', bottom_box)]
    mapping = [
        ('panel_box', 'BOTTOM_PANEL'),
        ('special_box', 'BOTTOM_SPECIAL'),
        ('points_box', 'POINTS_BADGE'),
        ('slash_box', 'BOTTOM_SLASH'),
        ('right_icon_box', 'BOTTOM_RIGHT_ICON'),
        ('bottom_line_box', 'BOTTOM_LINE'),
    ]
    for key, name in mapping:
        box = bottom_info.get(key)
        if box is None:
            continue
        ox, oy, ow, oh = box
        abs_box = _clip_box_tuple(bottom_box[0] + ox, bottom_box[1] + oy, ow, oh, w, h)
        rois.append(_roi_dict(name, abs_box))

    # bottom vector prioritaire : panneau noir si trouvé, sinon toute la zone bottom.
    panel_box = bottom_info.get('panel_box')
    if panel_box is not None:
        px, py, pw, ph = panel_box
        panel_crop = bottom_crop[py:py+ph, px:px+pw].copy()
        panel_gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY) if panel_crop.size else cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
    else:
        panel_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)

    bottom_signature = {
        'vector': image_to_vector(panel_gray, size=(16, 16)),
        'mean': safe_float(np.mean(panel_gray)),
        'std': safe_float(np.std(panel_gray)),
    }
    return bottom_info, rois, bottom_crop, bottom_signature


def _normalize_points_payload(points_info: Dict[str, Any], bottom_info: Dict[str, Any]) -> Dict[str, Any]:
    digit = int(bottom_info.get('points') or points_info.get('digit') or 0)
    raw_digit = int(bottom_info.get('raw_points') or points_info.get('raw_digit') or digit)
    score = float(bottom_info.get('points_score', points_info.get('score', 0.0)))
    gap = float(bottom_info.get('points_gap', points_info.get('gap', 1.0)))
    found = bool(digit > 0 and (score >= 0.40 or bottom_info.get('points_box') is not None))
    return {
        'digit': digit,
        'raw_digit': raw_digit,
        'score': score,
        'gap': gap,
        'found': found,
        'mean': float(points_info.get('mean', 0.0)),
        'std': float(points_info.get('std', 0.0)),
        'scores': bottom_info.get('digit_scores', []),
    }


def detect_main_card(img: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        img = ensure_color_image(img)
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 55, 165)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

        bright = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours = []
        for mask in (edges, bright):
            cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cts)

        image_area = float(w * h)
        best = None
        best_score = -1.0

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < image_area * 0.06:
                continue

            quad = contour_to_quad(cnt)
            if quad is None:
                continue

            ordered = order_quad_points(quad)
            width_a = np.linalg.norm(ordered[2] - ordered[3])
            width_b = np.linalg.norm(ordered[1] - ordered[0])
            height_a = np.linalg.norm(ordered[1] - ordered[2])
            height_b = np.linalg.norm(ordered[0] - ordered[3])
            card_w = float(max(width_a, width_b, 1.0))
            card_h = float(max(height_a, height_b, 1.0))
            portrait_ratio = max(card_h / card_w, card_w / card_h)
            if card_h <= card_w:
                continue
            if portrait_ratio < 1.20 or portrait_ratio > 1.95:
                continue

            x, y, bw, bh = cv2.boundingRect(ordered.astype(np.int32))
            if bw < 40 or bh < 60:
                continue
            rect_area = float(max(1, bw * bh))
            fill_ratio = area / rect_area
            if fill_ratio < 0.38:
                continue

            ratio_penalty = abs(portrait_ratio - 1.5)
            ratio_score = max(0.0, 1.0 - (ratio_penalty / 0.55))
            center_x = x + bw / 2.0
            center_y = y + bh / 2.0
            dx = abs(center_x - (w / 2.0)) / max(1.0, w / 2.0)
            dy = abs(center_y - (h / 2.0)) / max(1.0, h / 2.0)
            center_score = max(0.0, 1.0 - (dx * 0.35 + dy * 0.20))
            score = area * (0.45 + fill_ratio) * (0.60 + 0.40 * ratio_score) * (0.65 + 0.35 * center_score)

            if score > best_score:
                best_score = score
                best = {
                    'x': int(x),
                    'y': int(y),
                    'w': int(bw),
                    'h': int(bh),
                    'quad': [[int(px), int(py)] for px, py in ordered],
                }
        return best
    except Exception:
        return None


def compute_signature(warped: np.ndarray) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        warped = ensure_color_image(warped)
        if warped.shape[0] < 20 or warped.shape[1] < 20:
            return None, []

        card_norm = normalize_card_image(warped)
        h, w = card_norm.shape[:2]

        color_box = _clip_box_tuple(0, 0, int(w * 0.38), int(h * 0.18), w, h)
        global_box = _clip_box_tuple(0, 0, w, h, w, h)
        symbol_box, symbol_crop = _find_symbol_box(card_norm)

        color_crop = _crop_box(card_norm, color_box)
        global_crop = _crop_box(card_norm, global_box)
        bottom_info, bottom_rois, bottom_crop, bottom_signature = _build_bottom_rois(card_norm)

        rois = [
            _roi_dict('COLOR', color_box),
            _roi_dict('SYMBOL', symbol_box),
            _roi_dict('GLOBAL', global_box),
        ] + bottom_rois

        color_info = detect_dominant_color_name(color_crop)
        symbol_gray = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY) if symbol_crop.size else np.zeros((8, 8), dtype=np.uint8)
        bottom_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY) if bottom_crop.size else np.zeros((8, 8), dtype=np.uint8)
        global_gray = cv2.cvtColor(global_crop, cv2.COLOR_BGR2GRAY)

        symbol_info = detect_symbol_name(symbol_gray)
        points_info = detect_points_from_bottom(bottom_gray)
        points_payload = _normalize_points_payload(points_info, bottom_info)

        bottom_layout = {
            'layout': bottom_info.get('layout', 'UNKNOWN'),
            'has_slash': bool(bottom_info.get('has_slash')),
            'has_right_icon': bool(bottom_info.get('has_right_icon')),
            'has_bottom_line': bool(bottom_info.get('has_bottom_line')),
            'has_special_white_panel': bool(bottom_info.get('has_special_white_panel')),
            'points': int(points_payload.get('digit', 0)),
            'raw_points': int(points_payload.get('raw_digit', 0)),
            'points_score': float(points_payload.get('score', 0.0)),
            'points_gap': float(points_payload.get('gap', 1.0)),
            'range': 'GLOBAL' if bottom_info.get('has_slash') else '',
            'target': '',
            'ref_card_id': None,
            'ref_score': float(points_payload.get('score', 0.0)),
            'ref_gap': float(points_payload.get('gap', 1.0)),
        }

        signature = {
            'color': color_info,
            'symbol': symbol_info,
            'bottom': bottom_signature,
            'global': {
                'vector': image_to_vector(global_gray, size=(16, 16)),
                'mean': safe_float(np.mean(global_gray)),
                'std': safe_float(np.std(global_gray)),
            },
            'points': points_payload,
            'bottom_layout': bottom_layout,
            'normalized_size': {'w': w, 'h': h},
        }
        return signature, rois
    except Exception:
        return None, []

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
