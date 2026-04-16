#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

CARD_W = 200
CARD_H = 300
WARP_PATH = str(DEBUG_DIR / "last_warp.jpg")

CARD_DB_CANDIDATES = [
    BASE_DIR / "cards_db.json",
    BASE_DIR / "card_db.json",
    BASE_DIR / "cards.json",
    BASE_DIR / "references.json",
    BASE_DIR / "cards.js",
    BASE_DIR / "static" / "cards.js",
]
SYMBOLS_DIR_CANDIDATES = [BASE_DIR / "symbols", BASE_DIR / "static" / "symbols"]
DIGITS_DIR_CANDIDATES = [BASE_DIR / "digits", BASE_DIR / "static" / "digits"]
CARDS_DIR_CANDIDATES = [BASE_DIR / "cards", BASE_DIR / "static" / "cards"]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


# -----------------------------------------------------
# UTILS
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


def save_debug_image(name: str, img: Optional[np.ndarray]) -> Optional[str]:
    if img is None or getattr(img, "size", 0) == 0:
        return None
    path = DEBUG_DIR / name
    try:
        cv2.imwrite(str(path), img)
        return str(path)
    except Exception:
        return None


def crop_box(img: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, Dict[str, int]]:
    ih, iw = img.shape[:2]
    x = clamp_int(x, 0, max(0, iw - 1))
    y = clamp_int(y, 0, max(0, ih - 1))
    w = clamp_int(w, 1, max(1, iw - x))
    h = clamp_int(h, 1, max(1, ih - y))
    return img[y:y + h, x:x + w].copy(), {"x": x, "y": y, "w": w, "h": h}


def crop_rel(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = img.shape[:2]
    xa = clamp_int(round(x1 * w), 0, w - 1)
    ya = clamp_int(round(y1 * h), 0, h - 1)
    xb = clamp_int(round(x2 * w), xa + 1, w)
    yb = clamp_int(round(y2 * h), ya + 1, h)
    crop = img[ya:yb, xa:xb].copy()
    roi = {"x": xa, "y": ya, "w": xb - xa, "h": yb - ya}
    return crop, roi


def resize_keep(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


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


def _side_lengths(quad: np.ndarray) -> Tuple[float, float]:
    q = order_quad_points(quad)
    tl, tr, br, bl = q
    width = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0)
    height = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0)
    return width, height


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
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def warp_quad(img: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    try:
        return four_point_transform(img, quad)
    except Exception:
        return None


def normalize_card_image(img: np.ndarray) -> np.ndarray:
    img = ensure_color_image(img)
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return resize_keep(img, CARD_W, CARD_H)


def image_to_vector(gray: np.ndarray, size: Tuple[int, int] = (16, 16)) -> List[int]:
    if gray is None or gray.size == 0:
        return [0] * (size[0] * size[1])
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return [int(v) for v in small.flatten().tolist()]


def compute_basic_signature(img: np.ndarray, size: Tuple[int, int] = (32, 32)) -> Optional[Dict[str, Any]]:
    if img is None or img.size == 0:
        return None
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "mean": safe_float(np.mean(gray)),
        "std": safe_float(np.std(gray)),
        "color": [
            safe_float(np.mean(small[:, :, 0])),
            safe_float(np.mean(small[:, :, 1])),
            safe_float(np.mean(small[:, :, 2])),
        ],
        "vector": image_to_vector(gray, size=(16, 16)),
    }


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


def euclidean_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    dist = float(np.sqrt(np.mean((va - vb) ** 2)))
    return 1.0 / (1.0 + dist)


def mean_std_score(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    if not a or not b:
        return 0.0
    if a.get("mean") is None or b.get("mean") is None:
        return 0.0
    d_mean = abs(float(a.get("mean", 0.0)) - float(b.get("mean", 0.0)))
    d_std = abs(float(a.get("std", 0.0)) - float(b.get("std", 0.0)))
    return max(0.0, 1.0 - (d_mean + d_std) / 510.0)


def patch_score(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    if not a or not b:
        return 0.0
    vec_a = a.get("vector") if isinstance(a, dict) else None
    vec_b = b.get("vector") if isinstance(b, dict) else None
    stats_score = mean_std_score(a, b)
    if isinstance(vec_a, list) and isinstance(vec_b, list) and len(vec_a) == len(vec_b):
        vec_score = euclidean_similarity(vec_a, vec_b)
        return (vec_score * 0.75) + (stats_score * 0.25)
    return stats_score


# -----------------------------------------------------
# LOAD DB / ASSETS
# -----------------------------------------------------

def find_existing_dir(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def load_optional_card_db() -> List[Dict[str, Any]]:
    for path in CARD_DB_CANDIDATES:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if path.suffix.lower() == ".json":
                data = json.loads(raw)
            else:
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
SYMBOLS_DIR = find_existing_dir(SYMBOLS_DIR_CANDIDATES)
DIGITS_DIR = find_existing_dir(DIGITS_DIR_CANDIDATES)
CARDS_DIR = find_existing_dir(CARDS_DIR_CANDIDATES)


def _card_symbol_name(card_item: Dict[str, Any]) -> str:
    return str(card_item.get("symbol") or card_item.get("symbol_name") or "").upper().strip()


def _card_color_name(card_item: Dict[str, Any]) -> str:
    return str(card_item.get("couleur") or card_item.get("color") or card_item.get("color_name") or "").upper().strip()


# -----------------------------------------------------
# CARD DETECTION
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


def _candidate_score_from_quad(quad: np.ndarray, contour_area: float, image_w: int, image_h: int) -> Optional[Tuple[float, Dict[str, Any]]]:
    rect = order_quad_points(quad)
    x, y, bw, bh = cv2.boundingRect(rect.astype(np.int32))
    if bw < 24 or bh < 24:
        return None

    image_area = float(max(1, image_w * image_h))
    box_area = float(max(1, bw * bh))
    area_ratio = contour_area / image_area
    fill_ratio = contour_area / box_area
    if area_ratio < 0.04 or area_ratio > 0.96:
        return None
    if fill_ratio < 0.40:
        return None

    width, height = _side_lengths(rect)
    if width <= 1 or height <= 1:
        return None
    ratio = max(width, height) / max(1.0, min(width, height))
    portrait = height >= width
    portrait_score = 1.0 - min(abs(ratio - 1.50) / 0.65, 1.0)
    if not portrait:
        portrait_score *= 0.65
    if ratio < 1.10 or ratio > 2.40:
        return None

    border_touch = 0.0
    border_pad_x = max(8, int(image_w * 0.015))
    border_pad_y = max(8, int(image_h * 0.015))
    if x <= border_pad_x:
        border_touch += 0.35
    if y <= border_pad_y:
        border_touch += 0.35
    if x + bw >= image_w - border_pad_x:
        border_touch += 0.35
    if y + bh >= image_h - border_pad_y:
        border_touch += 0.35

    fullframe_penalty = 0.0
    if bw >= image_w * 0.96:
        fullframe_penalty += 1.8
    elif bw >= image_w * 0.90:
        fullframe_penalty += 1.1
    if bh >= image_h * 0.98:
        fullframe_penalty += 1.8
    elif bh >= image_h * 0.92:
        fullframe_penalty += 1.0

    center_x = x + bw / 2.0
    center_y = y + bh / 2.0
    center_score_x = 1.0 - min(abs(center_x - (image_w / 2.0)) / float(max(image_w * 0.55, 1.0)), 1.0)
    center_score_y = 1.0 - min(abs(center_y - (image_h / 2.0)) / float(max(image_h * 0.55, 1.0)), 1.0)

    score = (
        (area_ratio * 7.0)
        + (fill_ratio * 2.8)
        + (portrait_score * 3.8)
        + (center_score_x * 0.8)
        + (center_score_y * 0.8)
        - border_touch
        - fullframe_penalty
    )

    payload = {
        "x": int(x),
        "y": int(y),
        "w": int(bw),
        "h": int(bh),
        "quad": [[int(px), int(py)] for px, py in rect],
        "metrics": {
            "area_ratio": area_ratio,
            "fill_ratio": fill_ratio,
            "ratio": ratio,
            "portrait": portrait,
            "border_touch": border_touch,
            "fullframe_penalty": fullframe_penalty,
            "score": score,
        },
    }
    return score, payload


def detect_main_card(img: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        img = ensure_color_image(img)
        ih, iw = img.shape[:2]
        if ih < 10 or iw < 10:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        _, th_light = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        _, th_dark = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_or(edges, th_dark)
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(th_light))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Dict[str, Any]] = None
        best_score = -1e9

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < (iw * ih) * 0.03:
                continue
            quad = contour_to_quad(cnt)
            if quad is None:
                continue
            scored = _candidate_score_from_quad(quad, area, iw, ih)
            if scored is None:
                continue
            score, payload = scored
            if score > best_score:
                best_score = score
                best = payload

        return best
    except Exception as e:
        print("detect_main_card ERROR", e)
        return None


# -----------------------------------------------------
# COLOR DETECTION
# -----------------------------------------------------

def detect_dominant_color_name(img: np.ndarray) -> Dict[str, Any]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s > 35) & (v > 40)
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


# -----------------------------------------------------
# SYMBOL DETECTION
# -----------------------------------------------------

def _normalize_binary_mask(mask: np.ndarray, target: int = 96) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th > 0) > 0.55:
        binary = cv2.bitwise_not(th)
    else:
        binary = th
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    h, w = binary.shape[:2]
    cx0 = w / 2.0
    cy0 = h / 2.0
    keep = np.zeros_like(binary)
    best_score = -1e9
    chosen: List[int] = []
    min_area = max(8, int(h * w * 0.01))
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        touches_border = x <= 0 or y <= 0 or (x + ww) >= w - 1 or (y + hh) >= h - 1
        cx, cy = centroids[label]
        center_score = 1.0 - min(abs(cx - cx0) / max(w * 0.5, 1.0), 1.0)
        center_score += 1.0 - min(abs(cy - cy0) / max(h * 0.5, 1.0), 1.0)
        fill_ratio = area / float(max(1, ww * hh))
        ratio = ww / float(max(1, hh))
        shape_score = 1.0 - min(abs(fill_ratio - 0.40) / 0.40, 1.0)
        shape_score += 1.0 - min(abs(ratio - 1.0) / 1.0, 1.0)
        score = center_score * 2.0 + shape_score * 1.2 + min(area / float(h * w * 0.18), 1.0)
        if touches_border:
            score -= 0.8
        if score > best_score:
            best_score = score
        chosen.append((score, label))
    if not chosen:
        return None
    chosen.sort(reverse=True, key=lambda t: t[0])
    threshold = max(0.25, chosen[0][0] * 0.48)
    for score, label in chosen:
        if score >= threshold:
            keep[labels == label] = 255
    ys, xs = np.where(keep > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1
    crop = keep[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    canvas = np.zeros((target, target), dtype=np.uint8)
    scale = min((target - 10) / float(max(cw, 1)), (target - 10) / float(max(ch, 1)))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _extract_symbol_mask_from_roi(roi: np.ndarray, target: int = 96) -> Optional[np.ndarray]:
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    gray = cv2.equalizeHist(cv2.GaussianBlur(gray, (3, 3), 0))
    _, inv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)
    mask = cv2.bitwise_or(inv_otsu, adap)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _normalize_binary_mask(mask, target=target)


def _mask_similarity(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    iou = inter / union if union > 0 else 0.0
    cos = cosine_similarity(a.flatten().astype(np.float32).tolist(), b.flatten().astype(np.float32).tolist())
    ax = np.mean(a > 0, axis=0)
    bx = np.mean(b > 0, axis=0)
    ay = np.mean(a > 0, axis=1)
    by = np.mean(b > 0, axis=1)
    proj_x = max(0.0, 1.0 - float(np.mean(np.abs(ax - bx))))
    proj_y = max(0.0, 1.0 - float(np.mean(np.abs(ay - by))))
    contour_score = 0.0
    try:
        ca, _ = cv2.findContours((a * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cb, _ = cv2.findContours((b * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ca and cb:
            sa = max(ca, key=cv2.contourArea)
            sb = max(cb, key=cv2.contourArea)
            dist = float(cv2.matchShapes(sa, sb, cv2.CONTOURS_MATCH_I1, 0.0))
            contour_score = max(0.0, 1.0 - min(dist / 3.0, 1.0))
    except Exception:
        contour_score = 0.0
    return float((cos * 0.38) + (iou * 0.22) + (proj_x * 0.14) + (proj_y * 0.14) + (contour_score * 0.12))


def _load_symbol_template_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            return _normalize_binary_mask(alpha, target=96)
        if img.ndim == 3:
            return _extract_symbol_mask_from_roi(img, target=96)
        return _normalize_binary_mask(img, target=96)
    except Exception:
        return None


def _symbol_crop_from_reference_card(img: np.ndarray) -> Optional[np.ndarray]:
    card = normalize_card_image(img)
    roi, _ = crop_rel(card, 0.00, 0.10, 0.34, 0.40)
    return _extract_symbol_mask_from_roi(roi, target=96)


def build_symbol_templates() -> Dict[str, List[np.ndarray]]:
    templates: Dict[str, List[np.ndarray]] = {}
    if SYMBOLS_DIR is not None:
        for path in SYMBOLS_DIR.glob("*.png"):
            name = path.stem.upper().strip()
            mask = _load_symbol_template_image(path)
            if mask is not None:
                templates.setdefault(name, []).append(mask)
    if CARDS_DIR is not None and CARD_DB:
        for card in CARD_DB:
            symbol_name = _card_symbol_name(card)
            card_id = str(card.get("id") or "").lower()
            if not symbol_name or not card_id:
                continue
            for suffix in (".jpeg", ".jpg", ".png"):
                p = CARDS_DIR / f"{card_id}{suffix}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is None or img.size == 0:
                        break
                    mask = _symbol_crop_from_reference_card(img)
                    if mask is not None:
                        templates.setdefault(symbol_name, []).append(mask)
                    break
    return templates


SYMBOL_TEMPLATES = build_symbol_templates()


def detect_symbol_name(card_norm: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, int], Optional[np.ndarray]]:
    search_zone, search_box = crop_rel(card_norm, 0.00, 0.10, 0.34, 0.40)
    search_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.equalizeHist(cv2.GaussianBlur(search_gray, (3, 3), 0))
    _, th = cv2.threshold(search_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    zh, zw = search_gray.shape[:2]
    best_local = None
    best_score = -1e9
    min_area = max(12, int(zh * zw * 0.01))
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label]
        center_score = 1.0 - min(abs(cx - (zw * 0.32)) / max(zw * 0.45, 1.0), 1.0)
        center_score += 1.0 - min(abs(cy - (zh * 0.45)) / max(zh * 0.45, 1.0), 1.0)
        fill_ratio = area / float(max(1, w * h))
        ratio = w / float(max(1, h))
        shape_score = 1.0 - min(abs(fill_ratio - 0.45) / 0.45, 1.0)
        shape_score += 1.0 - min(abs(ratio - 1.0) / 1.2, 1.0)
        score = center_score * 2.1 + shape_score * 1.6 + min(area / float(zh * zw * 0.10), 1.0)
        if x <= 0 or y <= 0 or x + w >= zw - 1 or y + h >= zh - 1:
            score -= 0.5
        if score > best_score:
            best_score = score
            pad_x = max(3, int(w * 0.22))
            pad_y = max(3, int(h * 0.22))
            rx = max(0, x - pad_x)
            ry = max(0, y - pad_y)
            rw = min(zw - rx, w + 2 * pad_x)
            rh = min(zh - ry, h + 2 * pad_y)
            best_local = {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)}

    if best_local is None:
        best_local = {"x": int(zw * 0.06), "y": int(zh * 0.15), "w": int(zw * 0.46), "h": int(zh * 0.44)}

    roi, _ = crop_box(search_zone, best_local["x"], best_local["y"], best_local["w"], best_local["h"])
    symbol_mask = _extract_symbol_mask_from_roi(roi, target=96)

    candidates = []
    for name, masks in SYMBOL_TEMPLATES.items():
        best = 0.0
        for tpl in masks:
            score = _mask_similarity(symbol_mask, tpl)
            if score > best:
                best = score
        candidates.append({"name": name, "score": float(best)})
    candidates.sort(key=lambda c: c["score"], reverse=True)

    if candidates:
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else {"score": 0.0, "name": None}
        score = float(best["score"])
        gap = float(score - float(second.get("score", 0.0)))
        raw_name = str(best.get("name") or "INCONNU")
        if score >= 0.62 and gap >= 0.02:
            threshold_mode = "accepted"
            name = raw_name
        elif score >= 0.54:
            threshold_mode = "weak"
            name = raw_name
        else:
            threshold_mode = "unknown"
            name = "INCONNU"
    else:
        score = 0.0
        gap = 0.0
        raw_name = "INCONNU"
        name = "INCONNU"
        threshold_mode = "unknown"
        second = {"name": None, "score": 0.0}

    abs_box = {
        "x": int(search_box["x"] + best_local["x"]),
        "y": int(search_box["y"] + best_local["y"]),
        "w": int(best_local["w"]),
        "h": int(best_local["h"]),
    }

    info = {
        "name": name,
        "raw_name": raw_name,
        "score": score,
        "gap": gap,
        "mean": safe_float(np.mean(search_gray)),
        "std": safe_float(np.std(search_gray)),
        "mode": "templates" if SYMBOL_TEMPLATES else "none",
        "threshold_mode": threshold_mode,
        "runner_up": {"name": second.get("name"), "score": float(second.get("score", 0.0))},
        "top_candidates": candidates[:4],
        "winner_references": [],
    }
    return info, abs_box, symbol_mask


# -----------------------------------------------------
# DIGITS / BOTTOM ANALYSIS
# -----------------------------------------------------

def _clip_box_tuple(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    if x + w > max_w:
        w = max_w - x
    if y + h > max_h:
        h = max_h - y
    return max(0, x), max(0, y), max(1, w), max(1, h)


def _offset_box(box: Optional[Tuple[int, int, int, int]], dx: int, dy: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None:
        return None
    x, y, w, h = box
    return int(x + dx), int(y + dy), int(w), int(h)


def _normalize_badge(img_or_mask: np.ndarray, target: int = 96) -> Optional[np.ndarray]:
    if img_or_mask is None or img_or_mask.size == 0:
        return None
    gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY) if len(img_or_mask.shape) == 3 else img_or_mask.copy()
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
    ch, cw = crop.shape[:2]
    scale = min((target - 12) / max(cw, 1), (target - 12) / max(ch, 1))
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _extract_digit_mask(img_or_mask: np.ndarray, target: int = 64) -> Optional[np.ndarray]:
    if img_or_mask is None or img_or_mask.size == 0:
        return None
    gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY) if len(img_or_mask.shape) == 3 else img_or_mask.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, white_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
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
    inner_badge = cv2.erode(crop_badge, np.ones((2, 2), np.uint8), iterations=1)
    badge_pixels = crop_gray[inner_badge > 0]
    if badge_pixels.size == 0:
        return None
    dark_threshold = np.percentile(badge_pixels, 42)
    digit_mask = np.zeros_like(crop_gray, dtype=np.uint8)
    digit_mask[crop_gray < dark_threshold] = 255
    digit_mask = cv2.bitwise_and(digit_mask, inner_badge)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _normalize_binary_mask(digit_mask, target=target)


def _digit_score(scan_badge: Optional[np.ndarray], tpl_badge: Optional[np.ndarray], scan_digit: Optional[np.ndarray], tpl_digit: Optional[np.ndarray]) -> float:
    badge_score = _mask_similarity(scan_badge, tpl_badge)
    digit_score = _mask_similarity(scan_digit, tpl_digit)
    return float((badge_score * 0.38) + (digit_score * 0.62))


def detect_digit(zone: np.ndarray) -> Dict[str, Any]:
    if zone is None or zone.size == 0 or DIGITS_DIR is None:
        return {"digit": None, "score": 0.0, "gap": 0.0, "scores": []}
    scan_badge = _normalize_badge(zone)
    scan_digit = _extract_digit_mask(zone)
    if scan_badge is None:
        return {"digit": None, "score": 0.0, "gap": 0.0, "scores": []}
    scores = []
    for n in range(1, 11):
        path = DIGITS_DIR / f"{n}.png"
        tpl = cv2.imread(str(path))
        if tpl is None or tpl.size == 0:
            continue
        tpl_badge = _normalize_badge(tpl)
        tpl_digit = _extract_digit_mask(tpl)
        score = _digit_score(scan_badge, tpl_badge, scan_digit, tpl_digit)
        scores.append({"digit": n, "score": float(score)})
    if not scores:
        return {"digit": None, "score": 0.0, "gap": 0.0, "scores": []}
    scores.sort(key=lambda x: x["score"], reverse=True)
    best = scores[0]
    second = scores[1] if len(scores) > 1 else {"score": 0.0}
    return {
        "digit": int(best["digit"]),
        "score": float(best["score"]),
        "gap": float(best["score"] - second["score"]),
        "scores": scores,
    }


def _make_bottom_light_mask(zone: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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


def _find_black_panel_box(bottom_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if bottom_zone is None or bottom_zone.size == 0:
        return None
    gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    image_area = float(max(w * h, 1))
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        area_ratio = area / image_area
        if area_ratio < 0.18:
            continue
        if bw < w * 0.45 or bh < h * 0.45 or x > w * 0.20:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 1.2 or ratio > 4.8:
            continue
        cx = x + (bw / 2.0)
        left_score = 1.0 - min(cx / float(max(w, 1)), 1.0)
        size_score = min(area_ratio / 0.45, 1.0)
        score = (size_score * 3.0) + left_score
        candidates.append((score, x, y, bw, bh))
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, x, y, bw, bh = candidates[0]
        return _clip_box_tuple(x, y, bw, bh, w, h)
    return _clip_box_tuple(int(w * 0.02), int(h * 0.08), int(w * 0.50), int(h * 0.84), w, h)


def _find_special_white_panel_box(bottom_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if bottom_zone is None or bottom_zone.size == 0:
        return None
    mask, gray = _make_bottom_light_mask(bottom_zone)
    if mask is None or gray is None:
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
        if area_ratio < 0.12 or x < w * 0.10 or bw < w * 0.35 or bh < h * 0.42 or y > h * 0.35:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 0.6 or ratio > 1.8:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_x_score = 1.0 - min(abs(cx - (w * 0.55)) / float(max(w * 0.35, 1)), 1.0)
        center_y_score = 1.0 - min(abs(cy - (h * 0.52)) / float(max(h * 0.35, 1)), 1.0)
        size_score = min(area_ratio / 0.28, 1.0)
        score = (center_x_score * 2.0) + (center_y_score * 2.0) + (size_score * 2.0)
        candidates.append((score, x, y, bw, bh))
    if not candidates or dark_ratio > 0.22:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return _clip_box_tuple(x, y, bw, bh, w, h)


def _score_points_badge_candidates(panel_zone: np.ndarray) -> List[Dict[str, Any]]:
    if panel_zone is None or panel_zone.size == 0:
        return []
    ph, pw = panel_zone.shape[:2]
    sx, sy, sw, sh = _clip_box_tuple(int(pw * 0.00), int(ph * 0.04), int(pw * 0.58), int(ph * 0.90), pw, ph)
    search = panel_zone[sy:sy + sh, sx:sx + sw]
    if search is None or search.size == 0:
        return []
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask_hsv = cv2.inRange(hsv, (0, 0, 115), (180, 130, 255))
    _, mask_gray = cv2.threshold(blur, 138, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_hsv, mask_gray)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    zh, zw = search.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        area_ratio = area / float(max(zw * zh, 1))
        if area_ratio < 0.01 or bw < 6 or bh < 10:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 0.20 or ratio > 1.60:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        left_score = 1.0 - min(cx / float(max(zw * 0.60, 1)), 1.0)
        center_y_score = 1.0 - min(abs(cy - (zh * 0.52)) / float(max(zh * 0.40, 1)), 1.0)
        size_score = min(area_ratio / 0.12, 1.0)
        edge_penalty = 0.0
        if x <= 1:
            edge_penalty += 0.15
        if x + bw >= zw - 1:
            edge_penalty += 0.10
        if x > zw * 0.38:
            edge_penalty += 0.35
        score = (left_score * 2.4) + (center_y_score * 1.7) + (size_score * 1.8) - edge_penalty
        pad_x = max(2, int(bw * 0.12))
        pad_y = max(2, int(bh * 0.12))
        rx = max(0, x - pad_x)
        ry = max(0, y - pad_y)
        rw = min(zw - rx, bw + 2 * pad_x)
        rh = min(zh - ry, bh + 2 * pad_y)
        candidates.append({
            "score": float(score),
            "box": (int(rx + sx), int(ry + sy), int(rw), int(rh)),
            "raw_box": (int(x + sx), int(y + sy), int(bw), int(bh)),
            "area_ratio": float(area_ratio),
            "ratio": float(ratio),
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:5]


def _find_points_badge_in_black_panel(panel_zone: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    if panel_zone is None or panel_zone.size == 0:
        return None, None, []
    candidates = _score_points_badge_candidates(panel_zone)
    if candidates:
        x, y, w, h = candidates[0]["box"]
        crop = panel_zone[y:y + h, x:x + w]
        if crop is not None and crop.size > 0:
            return crop, (x, y, w, h), candidates
    ph, pw = panel_zone.shape[:2]
    fx, fy, fw, fh = _clip_box_tuple(int(pw * 0.03), int(ph * 0.08), int(pw * 0.50), int(ph * 0.82), pw, ph)
    crop = panel_zone[fy:fy + fh, fx:fx + fw]
    if crop is None or crop.size == 0:
        return None, None, candidates
    return crop, (fx, fy, fw, fh), candidates


def _find_slash_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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
        if area_ratio < 0.01 or bh < zh * 0.28 or bw > zw * 0.45:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio > 0.80:
            continue
        cx = x + bw / 2.0
        center_score = 1.0 - min(abs(cx - (zw * 0.50)) / float(max(zw * 0.35, 1)), 1.0)
        tall_score = min(bh / float(max(zh * 0.65, 1)), 1.0)
        score = (center_score * 2.0) + (tall_score * 2.0) + (area_ratio * 3.0)
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return x + x1, y, bw, bh


def _find_right_icon_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    ph, pw = panel_zone.shape[:2]
    x1, x2 = int(pw * 0.64), pw
    y1, y2 = int(ph * 0.08), int(ph * 0.72)
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
        if area_ratio < 0.08 or bw < zw * 0.22 or bh < zh * 0.28:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 0.55 or ratio > 1.50 or (x + bw) >= (zw - 1):
            continue
        score = (area_ratio * 4.0) + (bh / float(max(zh, 1)))
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return x + x1, y + y1, bw, bh


def _find_bottom_line_box(panel_zone: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if panel_zone is None or panel_zone.size == 0:
        return None
    ph, pw = panel_zone.shape[:2]
    x1, x2 = int(pw * 0.52), pw
    y1, y2 = int(ph * 0.68), ph
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
        if area_ratio < 0.015 or bw < zw * 0.24 or bh > zh * 0.32:
            continue
        ratio = bw / float(max(bh, 1))
        if ratio < 2.4:
            continue
        score = (ratio * 1.5) + (area_ratio * 6.0)
        candidates.append((score, x, y, bw, bh))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return x + x1, y + y1, bw, bh


def analyze_bottom(bottom_zone: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {
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
        "digit_scores": [],
        "points_candidates": [],
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
    badge_crop, badge_box_local, candidates = _find_points_badge_in_black_panel(panel_zone)
    result["points_candidates"] = [{
        "score": float(c["score"]),
        "box": _offset_box(c["box"], px, py),
        "raw_box": _offset_box(c["raw_box"], px, py),
        "area_ratio": float(c["area_ratio"]),
        "ratio": float(c["ratio"]),
    } for c in candidates]
    if badge_crop is not None and badge_box_local is not None:
        digit_res = detect_digit(badge_crop)
        result["raw_points"] = digit_res.get("digit")
        result["points_score"] = float(digit_res.get("score", 0.0))
        result["points_gap"] = float(digit_res.get("gap", 0.0))
        if digit_res.get("digit") is not None and (digit_res.get("score", 0.0) >= 0.62 or (digit_res.get("score", 0.0) >= 0.54 and digit_res.get("gap", 0.0) >= 0.02)):
            result["points"] = int(digit_res["digit"])
        bx, by, bw, bh = badge_box_local
        result["points_box"] = _offset_box((bx, by, bw, bh), px, py)
        result["digit_scores"] = digit_res.get("scores", [])
    slash_box_local = _find_slash_box(panel_zone)
    if slash_box_local is not None:
        result["has_slash"] = True
        result["slash_box"] = _offset_box(slash_box_local, px, py)
    if result["has_slash"]:
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


# -----------------------------------------------------
# CARD SIGNATURE + MATCHING
# -----------------------------------------------------

def _tuple_box_to_dict(box: Optional[Tuple[int, int, int, int]], kind: str) -> Optional[Dict[str, Any]]:
    if box is None:
        return None
    x, y, w, h = box
    return {"type": kind, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def compute_card_signature(warped: np.ndarray) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    try:
        card = normalize_card_image(warped)
        rois: List[Dict[str, Any]] = []
        debug: Dict[str, Any] = {}
        save_debug_image("card_normalized.jpg", card)

        global_crop, global_box = crop_rel(card, 0.00, 0.00, 1.00, 1.00)
        color_crop, color_box = crop_rel(card, 0.00, 0.00, 0.38, 0.18)
        bottom_crop, bottom_box = crop_rel(card, 0.00, 0.82, 0.55, 1.00)
        rois.extend([
            {"type": "COLOR", **color_box},
            {"type": "GLOBAL", **global_box},
            {"type": "BOTTOM", **bottom_box},
        ])

        color_info = detect_dominant_color_name(color_crop)
        symbol_info, symbol_box, symbol_mask = detect_symbol_name(card)
        rois.append({"type": "SYMBOL", **symbol_box})

        bottom_layout = analyze_bottom(bottom_crop)
        if bottom_layout.get("special_box") is not None:
            rois.append({"type": "BOTTOM_SPECIAL", **_tuple_box_to_dict(bottom_layout["special_box"], "BOTTOM_SPECIAL")})
        if bottom_layout.get("panel_box") is not None:
            rois.append({"type": "BOTTOM_PANEL", **_tuple_box_to_dict(bottom_layout["panel_box"], "BOTTOM_PANEL")})
        if bottom_layout.get("points_box") is not None:
            rois.append({"type": "POINTS_BADGE", **_tuple_box_to_dict(bottom_layout["points_box"], "POINTS_BADGE")})
        if bottom_layout.get("slash_box") is not None:
            rois.append({"type": "BOTTOM_SLASH", **_tuple_box_to_dict(bottom_layout["slash_box"], "BOTTOM_SLASH")})
        if bottom_layout.get("right_icon_box") is not None:
            rois.append({"type": "BOTTOM_RIGHT_ICON", **_tuple_box_to_dict(bottom_layout["right_icon_box"], "BOTTOM_RIGHT_ICON")})
        if bottom_layout.get("bottom_line_box") is not None:
            rois.append({"type": "BOTTOM_LINE", **_tuple_box_to_dict(bottom_layout["bottom_line_box"], "BOTTOM_LINE")})

        bottom_sig = compute_basic_signature(bottom_crop)
        global_sig = compute_basic_signature(global_crop)
        color_sig = compute_basic_signature(color_crop)

        points_info = {
            "digit": int(bottom_layout["points"]) if bottom_layout.get("points") is not None else int(bottom_layout.get("raw_points") or 0),
            "raw_digit": int(bottom_layout.get("raw_points") or 0),
            "score": float(bottom_layout.get("points_score", 0.0)),
            "gap": float(bottom_layout.get("points_gap", 0.0)),
            "found": bool(bottom_layout.get("points") is not None),
            "scores": bottom_layout.get("digit_scores", []),
            "mean": safe_float(bottom_sig.get("mean", 0.0)) if bottom_sig else 0.0,
            "std": safe_float(bottom_sig.get("std", 0.0)) if bottom_sig else 0.0,
        }

        signature: Dict[str, Any] = {
            "normalized_size": {"w": CARD_W, "h": CARD_H},
            "global": global_sig,
            "bottom": bottom_sig,
            "color": {
                **color_info,
                **({"mean": color_sig.get("mean"), "std": color_sig.get("std"), "color": color_sig.get("color")} if color_sig else {}),
            },
            "symbol": symbol_info,
            "points": points_info,
            "bottom_layout": {
                "layout": bottom_layout.get("layout", "UNKNOWN"),
                "points": int(bottom_layout["points"]) if bottom_layout.get("points") is not None else int(bottom_layout.get("raw_points") or 0),
                "raw_points": int(bottom_layout.get("raw_points") or 0),
                "points_score": float(bottom_layout.get("points_score", 0.0)),
                "points_gap": float(bottom_layout.get("points_gap", 0.0)),
                "range": "",
                "target": "",
                "ref_card_id": None,
                "ref_score": float(bottom_layout.get("points_score", 0.0)),
                "ref_gap": float(bottom_layout.get("points_gap", 0.0)),
                "has_slash": bool(bottom_layout.get("has_slash")),
                "has_right_icon": bool(bottom_layout.get("has_right_icon")),
                "has_bottom_line": bool(bottom_layout.get("has_bottom_line")),
                "has_special_white_panel": bool(bottom_layout.get("has_special_white_panel")),
            },
        }

        debug["saved"] = {
            "warp": save_debug_image("card_warp.jpg", warped),
            "normalized": save_debug_image("card_normalized.jpg", card),
            "symbol_mask": save_debug_image("card_symbol_mask.jpg", symbol_mask),
            "bottom": save_debug_image("card_bottom.jpg", bottom_crop),
        }
        debug["symbol_box"] = symbol_box
        debug["bottom_analysis"] = bottom_layout
        return signature, rois, debug
    except Exception as e:
        print("compute_card_signature ERROR", e)
        return None, [], {"error": str(e)}


def get_scan_part(signature: Optional[Dict[str, Any]], part: str) -> Optional[Dict[str, Any]]:
    if not signature:
        return None
    if isinstance(signature.get("scan"), dict) and signature["scan"].get(part) is not None:
        return signature["scan"].get(part)
    return signature.get(part)


def clamp01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def get_reliable_detected_points(query_sig: Dict[str, Any]) -> Optional[int]:
    points = get_scan_part(query_sig, "points") or {}
    digit = points.get("digit")
    if digit is None:
        return None
    score = float(points.get("score", points.get("points_score", 0.0)) or 0.0)
    gap = float(points.get("gap", points.get("points_gap", 0.0)) or 0.0)
    if score < 0.70 or gap < 0.02:
        return None
    return int(digit)


def get_detected_symbol_info(query_sig: Dict[str, Any]) -> Dict[str, Any]:
    symbol = get_scan_part(query_sig, "symbol") or {}
    raw_name = str(symbol.get("raw_name") or symbol.get("name") or "").upper().strip() or None
    score = float(symbol.get("score", 0.0) or 0.0)
    gap = float(symbol.get("gap", 0.0) or 0.0)
    top_candidates = symbol.get("top_candidates") if isinstance(symbol.get("top_candidates"), list) else []
    score_conf = clamp01((score - 0.58) / 0.16)
    gap_conf = clamp01(gap / 0.05)
    confidence = clamp01((score_conf * 0.75) + (gap_conf * 0.25))
    reliable = bool(raw_name and score >= 0.68 and gap >= 0.03)
    weak_reliable = bool(raw_name and score >= 0.62)
    return {
        "rawName": raw_name,
        "name": raw_name if reliable else None,
        "score": score,
        "gap": gap,
        "reliable": reliable,
        "weakReliable": weak_reliable,
        "confidence": confidence,
        "topCandidates": top_candidates,
    }


def points_match_score(query_sig: Dict[str, Any], card: Dict[str, Any]) -> float:
    detected_points = get_reliable_detected_points(query_sig)
    if detected_points is None:
        return 0.50
    try:
        card_points = int(card.get("points"))
    except Exception:
        return 0.50
    if card_points == detected_points:
        return 1.0
    if abs(card_points - detected_points) == 1:
        return 0.15
    return 0.0


def symbol_match_score(symbol_info: Dict[str, Any], card: Dict[str, Any]) -> float:
    if not symbol_info or not symbol_info.get("rawName"):
        return 0.50
    card_symbol = str(card.get("symbol") or "").upper().strip()
    detected_symbol = str(symbol_info.get("rawName") or "").upper().strip()
    confidence = clamp01(float(symbol_info.get("confidence", 0.0)))
    if not card_symbol or not detected_symbol:
        return 0.50
    if card_symbol == detected_symbol:
        return 0.50 + (confidence * 0.50)
    return 0.50 - (confidence * 0.45)


def enrich_candidate(query_sig: Dict[str, Any], card: Dict[str, Any], symbol_info: Dict[str, Any]) -> Dict[str, Any]:
    card_sig = card.get("signature") or card
    q_color = (get_scan_part(query_sig, "color") or {}).get("color")
    c_color = (get_scan_part(card_sig, "color") or {}).get("color")
    color_score = 0.0
    if isinstance(q_color, list) and isinstance(c_color, list) and len(q_color) == 3 and len(c_color) == 3:
        d = abs(q_color[0] - c_color[0]) + abs(q_color[1] - c_color[1]) + abs(q_color[2] - c_color[2])
        color_score = 1.0 / (1.0 + (d / 50.0))
    symbol_score = symbol_match_score(symbol_info, card)
    points_score = points_match_score(query_sig, card)
    bottom_score = patch_score(get_scan_part(query_sig, "bottom"), get_scan_part(card_sig, "bottom"))
    global_score = patch_score(get_scan_part(query_sig, "global"), get_scan_part(card_sig, "global"))
    card_symbol = str(card.get("symbol") or "").upper().strip()
    symbol_exact = bool(symbol_info.get("rawName") and card_symbol == str(symbol_info.get("rawName")).upper().strip())
    return {
        "card": card,
        "colorScore": color_score,
        "symbolScore": symbol_score,
        "pointsScore": points_score,
        "bottomScore": bottom_score,
        "globalScore": global_score,
        "symbolExactMatch": symbol_exact,
    }


def keep_best_by(candidates: List[Dict[str, Any]], key: str, keep_top: int, ratio: float, min_keep: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    sorted_items = sorted(candidates, key=lambda c: float(c.get(key, 0.0)), reverse=True)
    best = float(sorted_items[0].get(key, 0.0))
    kept = [c for c in sorted_items if float(c.get(key, 0.0)) >= best * ratio]
    if len(kept) < min_keep:
        kept = sorted_items[: min(keep_top, len(sorted_items))]
    else:
        kept = kept[:keep_top]
    return kept


def compute_final_score(candidate: Dict[str, Any], symbol_mode: str) -> float:
    if symbol_mode == "strong":
        return (
            candidate["colorScore"] * 0.18
            + candidate["symbolScore"] * 0.30
            + candidate["pointsScore"] * 0.05
            + candidate["bottomScore"] * 0.30
            + candidate["globalScore"] * 0.17
        )
    if symbol_mode == "weak":
        return (
            candidate["colorScore"] * 0.22
            + candidate["symbolScore"] * 0.15
            + candidate["pointsScore"] * 0.05
            + candidate["bottomScore"] * 0.36
            + candidate["globalScore"] * 0.22
        )
    return (
        candidate["colorScore"] * 0.26
        + candidate["pointsScore"] * 0.05
        + candidate["bottomScore"] * 0.44
        + candidate["globalScore"] * 0.25
    )


def summarize_candidates(candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    for c in candidates[:limit]:
        card = c.get("card") or {}
        out.append({
            "id": card.get("id"),
            "couleur": card.get("couleur"),
            "symbol": card.get("symbol"),
            "points": card.get("points"),
            "colorScore": float(c.get("colorScore", 0.0)),
            "symbolScore": float(c.get("symbolScore", 0.0)),
            "pointsScore": float(c.get("pointsScore", 0.0)),
            "bottomScore": float(c.get("bottomScore", 0.0)),
            "globalScore": float(c.get("globalScore", 0.0)),
            "boostedBottom": float(c.get("boostedBottom", 0.0)),
            "finalScore": float(c.get("finalScore", 0.0)),
            "symbolExactMatch": bool(c.get("symbolExactMatch", False)),
        })
    return out


def match_front_signature(query_sig: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not query_sig or not CARD_DB:
        return None
    symbol_info = get_detected_symbol_info(query_sig)
    candidates = [enrich_candidate(query_sig, card, symbol_info) for card in CARD_DB]
    step_color = keep_best_by(candidates, "colorScore", keep_top=12, ratio=0.90, min_keep=4)
    step_symbol = step_color
    symbol_filter_applied = False
    if symbol_info.get("reliable") and symbol_info.get("rawName"):
        filtered = [c for c in step_color if c.get("symbolExactMatch")]
        if filtered:
            step_symbol = filtered
            symbol_filter_applied = True
    step_points = []
    for c in step_symbol:
        c2 = dict(c)
        c2["boostedBottom"] = (c["bottomScore"] * 0.88) + (c["pointsScore"] * 0.12)
        step_points.append(c2)
    step_bottom = keep_best_by(step_points, "boostedBottom", keep_top=6, ratio=0.93, min_keep=2)
    step_global = keep_best_by(step_bottom, "globalScore", keep_top=4, ratio=0.92, min_keep=2)
    symbol_mode = "strong" if symbol_filter_applied else ("weak" if symbol_info.get("weakReliable") else "none")
    scored_final = []
    for c in step_global:
        c2 = dict(c)
        c2["finalScore"] = compute_final_score(c2, symbol_mode)
        scored_final.append(c2)
    scored_final.sort(key=lambda c: c["finalScore"], reverse=True)
    best = scored_final[0] if scored_final else None
    if best is None:
        fallback = []
        for c in candidates:
            c2 = dict(c)
            c2["finalScore"] = compute_final_score(c2, symbol_mode)
            fallback.append(c2)
        fallback.sort(key=lambda c: c["finalScore"], reverse=True)
        best = fallback[0] if fallback else None
        scored_final = fallback
    if best is None or not best.get("card"):
        return None
    debug = {
        "detectedSymbol": symbol_info,
        "symbolFilterApplied": symbol_filter_applied,
        "finalOptions": {"symbolMode": symbol_mode},
        "steps": {
            "afterColor": summarize_candidates(step_color, 5),
            "afterSymbol": summarize_candidates(step_symbol, 5),
            "afterBottom": summarize_candidates(step_bottom, 5),
            "afterGlobal": summarize_candidates(step_global, 5),
            "final": summarize_candidates(scored_final, 5),
        },
        "best": {
            "id": best["card"].get("id"),
            "colorScore": best.get("colorScore", 0.0),
            "symbolScore": best.get("symbolScore", 0.0),
            "pointsScore": best.get("pointsScore", 0.0),
            "bottomScore": best.get("bottomScore", 0.0),
            "globalScore": best.get("globalScore", 0.0),
            "finalScore": best.get("finalScore", 0.0),
        },
    }
    return {"card": best["card"], "score": float(best.get("finalScore", 0.0)), "debug": debug}


def resolve_final_card(signature: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if signature is None:
        return None
    front = match_front_signature(signature)
    color_name = ((signature.get("color") or {}).get("detected") or "INCONNU")
    symbol_name = ((signature.get("symbol") or {}).get("name") or (signature.get("symbol") or {}).get("raw_name") or "INCONNU")
    bottom_layout = ((signature.get("bottom_layout") or {}).get("layout") or "UNKNOWN")
    points = int(((signature.get("bottom_layout") or {}).get("points") or 0))
    if front is None:
        return {
            "candidate_cards": [],
            "color_name": color_name,
            "symbol_name": symbol_name,
            "symbol_source": (signature.get("symbol") or {}).get("threshold_mode"),
            "bottom_layout": bottom_layout,
            "points": points,
            "final_card_id": None,
            "final_status": "unknown",
            "final_score": 0.0,
            "final_gap": 0.0,
            "reason": "no_card_db",
        }
    card = front.get("card") or {}
    final_score = float(front.get("score", 0.0))
    debug_final = ((front.get("debug") or {}).get("steps") or {}).get("final") or []
    second_score = float(debug_final[1].get("finalScore", 0.0)) if len(debug_final) > 1 else 0.0
    gap = final_score - second_score
    if final_score >= 0.89 and gap >= 0.008:
        status = "accepted"
        reason = "strong_front_match"
    elif final_score >= 0.84:
        status = "candidate"
        reason = "usable_front_match"
    else:
        status = "unknown"
        reason = "weak_front_match"
    candidate_cards = []
    for row in debug_final[:5]:
        candidate_cards.append({
            "card_id": row.get("id"),
            "score": int(round(float(row.get("finalScore", 0.0)) * 100)),
            "details": [],
            "global_visual": float(row.get("globalScore", 0.0)),
            "bottom_visual": float(row.get("bottomScore", 0.0)),
            "expected_bottom": {
                "layout": None,
                "points": row.get("points"),
                "target": row.get("symbol"),
                "range": "",
                "has_special_white_panel": False,
                "has_slash": False,
                "has_right_icon": False,
                "has_bottom_line": False,
            },
        })
    return {
        "candidate_cards": candidate_cards,
        "color_name": color_name,
        "symbol_name": symbol_name,
        "symbol_source": (signature.get("symbol") or {}).get("threshold_mode"),
        "bottom_layout": bottom_layout,
        "points": points,
        "final_card_id": card.get("id"),
        "final_status": status,
        "final_score": int(round(final_score * 100)),
        "final_gap": int(round(gap * 100)),
        "reason": reason,
    }


# -----------------------------------------------------
# STABLE PAYLOAD
# -----------------------------------------------------

def empty_upload_payload(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "rects": [],
        "signature": None,
        "rois": [],
        "card_match": None,
        "front_match": None,
        "final_card_id": None,
        "final_status": None,
        "final_score": 0.0,
        "final_gap": 0.0,
        "color_name": None,
        "symbol_name": None,
        "bottom_layout": None,
        "points": None,
        "board_analysis": None,
        "board_matches": [],
        "debug": None,
        "error": error,
    }


# -----------------------------------------------------
# BOARD (kept from current base)
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
        return _box_from_margins(box, left=0.12, top=0.00, right=0.12, bottom=0.02)
    if slot_id.startswith("bottom_"):
        return _box_from_margins(box, left=0.08, top=0.00, right=0.08, bottom=0.02)
    if slot_id in ("left_outer", "right_outer"):
        return _box_from_margins(box, left=0.03, top=0.00, right=0.03, bottom=0.02)
    return _box_from_margins(box, left=0.08, top=0.00, right=0.08, bottom=0.02)


def _projection_bounds(values: np.ndarray, threshold_ratio: float = 0.28, pad: int = 8) -> Optional[Tuple[int, int]]:
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
    return max(0, int(idx[0]) - pad), min(arr.size - 1, int(idx[-1]) + pad)


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
        if fill_ratio < 0.18 or area_ratio < 0.08 or area_ratio > 0.96 or aspect < 0.30 or aspect > 1.05:
            continue
        score = area * (0.45 + fill_ratio)
        if score > best_score:
            best_score = score
            best = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    return best


def _union_local_boxes(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    x1 = min(int(a["x"]), int(b["x"]))
    y1 = min(int(a["y"]), int(b["y"]))
    x2 = max(int(a["x"] + a["w"]), int(b["x"] + b["w"]))
    y2 = max(int(a["y"] + a["h"]), int(b["y"] + b["h"]))
    return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}


def _expand_local_box(local_box: Dict[str, int], crop_shape: Tuple[int, int, int], pad_x: int, pad_y: int) -> Dict[str, int]:
    h, w = crop_shape[:2]
    x1 = max(0, int(local_box["x"]) - int(pad_x))
    y1 = max(0, int(local_box["y"]) - int(pad_y))
    x2 = min(w, int(local_box["x"] + local_box["w"]) + int(pad_x))
    y2 = min(h, int(local_box["y"] + local_box["h"]) + int(pad_y))
    return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}


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
    edges = cv2.Canny(blur, 50, 150)
    pad_x = max(4, int(base_box["w"] * 0.02))
    pad_y = max(4, int(base_box["h"] * 0.02))
    inner = edges[pad_y:max(pad_y + 1, edges.shape[0] - pad_y), pad_x:max(pad_x + 1, edges.shape[1] - pad_x)]
    projection_box = None
    if inner.size > 0:
        xproj = np.mean(inner > 0, axis=0).astype(np.float32)
        yproj = np.mean(inner > 0, axis=1).astype(np.float32)
        xb = _projection_bounds(xproj, threshold_ratio=0.28, pad=max(6, int(base_box["w"] * 0.03)))
        yb = _projection_bounds(yproj, threshold_ratio=0.28, pad=max(6, int(base_box["h"] * 0.03)))
        if xb and yb:
            projection_box = {"x": int(xb[0] + pad_x), "y": int(yb[0] + pad_y), "w": int(xb[1] - xb[0] + 1), "h": int(yb[1] - yb[0] + 1)}
            projection_box = _clip_local_box(projection_box, crop.shape)
    foreground_mask = np.zeros_like(gray)
    foreground_mask[(gray < 228) | (hsv[:, :, 1] > 28)] = 255
    k = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, k)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, k)
    contour_box = _largest_contour_box(foreground_mask)
    candidate = default_box
    if projection_box and contour_box:
        candidate = _union_local_boxes(projection_box, contour_box)
    elif projection_box:
        candidate = _union_local_boxes(projection_box, default_box)
    elif contour_box:
        candidate = _union_local_boxes(contour_box, default_box)
    candidate = _clip_local_box(candidate, crop.shape)
    extra_x = max(6, int(candidate["w"] * 0.06))
    extra_y = max(6, int(candidate["h"] * 0.04))
    local_box = _expand_local_box(candidate, crop.shape, pad_x=extra_x, pad_y=extra_y)
    local_box = _union_local_boxes(local_box, default_box)
    local_box = _clip_local_box(local_box, crop.shape)
    abs_box = _local_box_to_abs(base_box, local_box)
    card_crop = crop[local_box["y"]:local_box["y"] + local_box["h"], local_box["x"]:local_box["x"] + local_box["w"]].copy()
    return {"local_box": local_box, "abs_box": abs_box, "tight_crop": card_crop}


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
    occupied = sat_ratio >= 0.05 or edge_ratio >= 0.025 or dark_ratio >= 0.18
    return occupied, {"sat_ratio": sat_ratio, "edge_ratio": edge_ratio, "dark_ratio": dark_ratio}


def _analyze_single_slot(slot_id: str, crop: np.ndarray, box: Dict[str, int]) -> Dict[str, Any]:
    occupied, metrics = _board_slot_occupied(crop)
    refined = _refine_board_card_box(slot_id, crop, box)
    tight_box = refined["abs_box"]
    slot: Dict[str, Any] = {
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
        "debug": None,
    }
    if not occupied:
        return slot
    tight_crop = refined.get("tight_crop") or crop.copy()
    rect = detect_main_card(tight_crop)
    if rect is None:
        h, w = tight_crop.shape[:2]
        rect = {"x": 0, "y": 0, "w": w, "h": h, "quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]}
    quad = np.array(rect["quad"], dtype=np.float32)
    warped = warp_quad(tight_crop, quad)
    if warped is None or warped.size == 0:
        warped = tight_crop.copy()
    sig, rois, debug = compute_card_signature(warped)
    slot["signature"] = sig
    slot["rois"] = rois
    slot["debug"] = debug
    if sig is not None:
        slot["match"] = resolve_final_card(sig)
    return slot


def analyze_board(img: np.ndarray) -> Dict[str, Any]:
    slots = []
    rects = []
    board_matches = []
    for slot_id, xr1, yr1, xr2, yr2 in BOARD_SLOT_LAYOUT:
        crop, box = _board_crop_box(img, xr1, yr1, xr2, yr2)
        slot = _analyze_single_slot(slot_id, crop, box)
        slots.append(slot)
        rects.append({"x": slot["x"], "y": slot["y"], "w": slot["w"], "h": slot["h"], "slot_id": slot_id})
        if slot.get("occupied") and slot.get("match"):
            m = slot["match"]
            board_matches.append({
                "slot_id": slot_id,
                "final_card_id": m.get("final_card_id"),
                "final_status": m.get("final_status"),
                "final_score": m.get("final_score", 0.0),
                "final_gap": m.get("final_gap", 0.0),
                "signature": slot.get("signature"),
                "candidates": m.get("candidate_cards", []),
            })
    return {
        "board_analysis": {"slots": slots, "slots_count": len(slots), "occupied_count": sum(1 for s in slots if s.get("occupied"))},
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
    return "<h1>SpaceLab backend OK</h1><p>Utilise <code>POST /upload</code> avec un champ fichier <code>image</code>.</p>"


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "spacelab",
        "card_db_loaded": bool(CARD_DB),
        "card_db_size": len(CARD_DB),
        "symbols_templates": {k: len(v) for k, v in SYMBOL_TEMPLATES.items()},
        "digits_dir": str(DIGITS_DIR) if DIGITS_DIR else None,
        "cards_dir": str(CARDS_DIR) if CARDS_DIR else None,
    })


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

        if mode == "BOARD":
            board_result = analyze_board(img)
            payload = empty_upload_payload(None)
            payload.update({
                "rects": board_result["rects"],
                "board_analysis": board_result["board_analysis"],
                "board_matches": board_result["board_matches"],
                "error": None,
            })
            return jsonify(payload)

        h, w = img.shape[:2]
        rect = detect_main_card(img)
        if rect is None:
            rect = {
                "x": 0,
                "y": 0,
                "w": int(w),
                "h": int(h),
                "quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                "metrics": {"fallback": True},
            }
        quad = np.array(rect["quad"], dtype=np.float32)
        warped = warp_quad(img, quad)
        if warped is None or warped.size == 0:
            warped = img.copy()
        try:
            cv2.imwrite(WARP_PATH, warped)
        except Exception:
            pass

        signature, rois, debug = compute_card_signature(warped)
        front_match = match_front_signature(signature) if signature else None
        card_match = resolve_final_card(signature) if signature else None

        payload = empty_upload_payload(None)
        payload.update({
            "rects": [rect],
            "signature": signature,
            "rois": rois,
            "card_match": card_match,
            "front_match": front_match,
            "final_card_id": (card_match or {}).get("final_card_id") if card_match else None,
            "final_status": (card_match or {}).get("final_status") if card_match else None,
            "final_score": (card_match or {}).get("final_score", 0.0) if card_match else 0.0,
            "final_gap": (card_match or {}).get("final_gap", 0.0) if card_match else 0.0,
            "color_name": (card_match or {}).get("color_name") if card_match else None,
            "symbol_name": (card_match or {}).get("symbol_name") if card_match else None,
            "bottom_layout": (card_match or {}).get("bottom_layout") if card_match else None,
            "points": (card_match or {}).get("points") if card_match else None,
            "debug": {
                "main_card_box": rect,
                "warp_path": WARP_PATH,
                **(debug or {}),
            },
            "error": None,
        })
        return jsonify(payload)
    except Exception as e:
        print("UPLOAD ERROR:", e)
        payload = empty_upload_payload(f"upload_exception: {e}")
        return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
