import os
import io
import json
import uuid
import zipfile
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, redirect, url_for, Response


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
CARD_W = 2000
CARD_H = 900

app = Flask(__name__)


# =========================
# Helpers fichiers / HTML
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def read_image_from_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(str(path), image)


def np_to_builtin(obj):
    if isinstance(obj, dict):
        return {k: np_to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [np_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# =========================
# Géométrie / perspective
# =========================
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def four_point_transform(image, pts, out_w=CARD_W, out_h=CARD_H):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def angle_cos(p0, p1, p2):
    d1 = p0 - p1
    d2 = p2 - p1
    denom = np.sqrt((d1 * d1).sum() * (d2 * d2).sum()) + 1e-10
    return abs(np.dot(d1, d2) / denom)


# =========================
# Détection plateau / multi-cartes
# =========================
def is_valid_card_quad(quad, image_shape):
    if quad is None or len(quad) != 4:
        return False

    h, w = image_shape[:2]
    area = cv2.contourArea(quad.astype(np.float32))
    img_area = w * h

    if area < img_area * 0.01:
        return False
    if area > img_area * 0.95:
        return False

    rect = order_points(quad.reshape(4, 2))
    widths = [
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[3]),
    ]
    heights = [
        np.linalg.norm(rect[2] - rect[1]),
        np.linalg.norm(rect[3] - rect[0]),
    ]

    width = max(np.mean(widths), 1)
    height = max(np.mean(heights), 1)
    ratio = width / height

    # Une carte proche de 2000x900 -> ratio ≈ 2.22
    if ratio < 1.45 or ratio > 2.9:
        return False

    pts = rect.astype(np.float32)
    max_cos = 0
    for i in range(4):
        c = angle_cos(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
        max_cos = max(max_cos, c)

    if max_cos > 0.40:
        return False

    return True


def non_max_suppression_quads(quads, overlap_thresh=0.25):
    if not quads:
        return []

    boxes = []
    for q in quads:
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        boxes.append([x, y, x + w, y + h, q])

    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []

    while boxes:
        current = boxes.pop(0)
        kept.append(current[4])

        remaining = []
        x1, y1, x2, y2, _ = current
        area1 = max(1, (x2 - x1) * (y2 - y1))

        for b in boxes:
            xx1 = max(x1, b[0])
            yy1 = max(y1, b[1])
            xx2 = min(x2, b[2])
            yy2 = min(y2, b[3])

            iw = max(0, xx2 - xx1)
            ih = max(0, yy2 - yy1)
            inter = iw * ih

            area2 = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            iou = inter / float(area1 + area2 - inter + 1e-9)

            if iou < overlap_thresh:
                remaining.append(b)

        boxes = remaining

    return kept


def detect_cards_on_board(image, debug=False):
    debug_data = {}

    h, w = image.shape[:2]
    scale = 1400.0 / max(h, w) if max(h, w) > 1400 else 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale != 1.0 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges1 = cv2.Canny(blur, 60, 160)
    edges2 = cv2.Canny(blur, 30, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel, iterations=2)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 8
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidate_maps = [edges1, edges2, thresh]
    candidate_quads = []

    for cmap in candidate_maps:
        contours, _ = cv2.findContours(cmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                if is_valid_card_quad(quad, small.shape):
                    candidate_quads.append(quad)
            else:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(np.float32)
                if is_valid_card_quad(box, small.shape):
                    candidate_quads.append(box)

    candidate_quads = non_max_suppression_quads(candidate_quads, overlap_thresh=0.25)

    results = []
    for quad in candidate_quads:
        quad_full = quad / scale
        warp = four_point_transform(image, quad_full, out_w=CARD_W, out_h=CARD_H)
        x, y, ww, hh = cv2.boundingRect(quad_full.astype(np.int32))

        results.append({
            "quad": quad_full.astype(int).tolist(),
            "rect": [int(x), int(y), int(ww), int(hh)],
            "warp": warp,
        })

    results = sorted(results, key=lambda r: (r["rect"][1], r["rect"][0]))

    if debug:
        dbg = image.copy()
        for i, r in enumerate(results, start=1):
            quad = np.array(r["quad"], dtype=np.int32)
            cv2.polylines(dbg, [quad], True, (0, 255, 0), 3)
            x, y, ww, hh = r["rect"]
            cv2.putText(
                dbg,
                f"CARTE {i}",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        debug_data["board_detected"] = dbg
        debug_data["edges1"] = edges1
        debug_data["edges2"] = edges2
        debug_data["threshold"] = thresh

    return results, debug_data


# =========================
# ROIs d'une carte redressée
# =========================
def get_card_rois(card_img):
    h, w = card_img.shape[:2]

    color_roi = {
        "type": "COLOR",
        "x": 0,
        "y": 0,
        "w": int(w * 0.038),
        "h": int(h * 0.060),
    }
    symbol_roi = {
        "type": "SYMBOL",
        "x": int(w * 0.002),
        "y": int(h * 0.053),
        "w": int(w * 0.022),
        "h": int(h * 0.060),
    }
    bottom_roi = {
        "type": "BOTTOM",
        "x": 0,
        "y": int(h * 0.272),
        "w": int(w * 0.055),
        "h": int(h * 0.061),
    }
    global_roi = {
        "type": "GLOBAL",
        "x": 0,
        "y": 0,
        "w": int(w * 0.10),
        "h": int(h * 0.333),
    }

    return [color_roi, symbol_roi, bottom_roi, global_roi]


def crop_roi(image, roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    return image[y:y + h, x:x + w].copy()


# =========================
# Heuristiques d'analyse carte
# =========================
def mean_std_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def signature_vector(gray_image, size=(16, 16)):
    small = cv2.resize(gray_image, size, interpolation=cv2.INTER_AREA)
    return small.flatten().astype(int).tolist()


def detect_color_name(color_crop):
    if color_crop.size == 0:
        return {
            "detected": "INCONNU",
            "color": [0, 0, 0],
            "debug": {"BLEU": 0, "JAUNE": 0, "ROUGE": 0, "VERT": 0},
            "mean": 0.0,
            "std": 0.0,
        }

    hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    sat_mask = s > 40
    if sat_mask.sum() == 0:
        mean_bgr = color_crop.reshape(-1, 3).mean(axis=0)
    else:
        mean_bgr = color_crop[sat_mask].mean(axis=0)

    scores = {
        "BLEU": int(((h >= 90) & (h <= 135) & (s > 40)).sum()),
        "VERT": int(((h >= 36) & (h <= 89) & (s > 40)).sum()),
        "JAUNE": int(((h >= 16) & (h <= 35) & (s > 40)).sum()),
        "ROUGE": int((((h <= 15) | (h >= 165)) & (s > 40)).sum()),
    }

    detected = max(scores, key=scores.get) if max(scores.values()) > 0 else "INCONNU"
    return {
        "detected": detected,
        "color": [float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])],
        "debug": scores,
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def detect_points_and_bottom_layout(bottom_crop):
    gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape[:2]

    left = th[:, : max(1, int(w * 0.55))]
    right = th[:, int(w * 0.70):] if w > 5 else th
    center_band = th[:, int(w * 0.45): int(w * 0.75)] if w > 10 else th

    left_ratio = float((left > 0).mean()) if left.size else 0.0
    right_ratio = float((right > 0).mean()) if right.size else 0.0
    center_ratio = float((center_band > 0).mean()) if center_band.size else 0.0

    has_right_icon = right_ratio > 0.08
    has_slash = center_ratio > 0.10
    has_bottom_line = float((th[int(h * 0.82):, :] > 0).mean()) > 0.20 if h > 5 else False
    has_special_white_panel = mean_val > 200 and std_val < 25

    # OCR robuste sans dépendances = non, on fait une estimation heuristique simple.
    # Valeurs possibles minimales pour que le JSON reste exploitable.
    points_guess = 3 if has_right_icon or has_slash else 10

    if has_right_icon:
        layout = "NUMBER_ICON"
    else:
        layout = "NUMBER_ONLY"

    target = "MEDECIN" if has_right_icon else ""
    range_name = "GLOBAL" if has_slash else ""

    return {
        "layout": layout,
        "points": int(points_guess),
        "raw_points": int(points_guess),
        "target": target,
        "range": range_name,
        "has_bottom_line": bool(has_bottom_line),
        "has_right_icon": bool(has_right_icon),
        "has_slash": bool(has_slash),
        "has_special_white_panel": bool(has_special_white_panel),
        "points_gap": 0.0,
        "points_score": 0.0,
        "ref_card_id": None,
        "ref_gap": 0.0,
        "ref_score": 0.0,
    }


def detect_symbol_name(symbol_crop):
    gray = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    score = min(1.0, max(0.0, std_val / 40.0))

    # Placeholder stable en attendant tes vraies références de symboles.
    name = "MEDECIN" if score >= 0.25 else "INCONNU"

    return {
        "name": name,
        "raw_name": name,
        "score": float(score),
        "gap": float(1.0 - score),
        "mode": "heuristic",
        "threshold_mode": "accepted" if name != "INCONNU" else "rejected",
        "mean": mean_val,
        "std": std_val,
        "runner_up": None,
        "top_candidates": [{
            "name": name,
            "score": float(score),
            "best_kind": "heuristic",
            "best_source": "local",
            "support": 1,
        }],
        "winner_references": [],
    }


def analyze_single_card(card_img):
    rois = get_card_rois(card_img)
    roi_map = {roi["type"]: roi for roi in rois}

    color_crop = crop_roi(card_img, roi_map["COLOR"])
    symbol_crop = crop_roi(card_img, roi_map["SYMBOL"])
    bottom_crop = crop_roi(card_img, roi_map["BOTTOM"])
    global_crop = crop_roi(card_img, roi_map["GLOBAL"])

    color_info = detect_color_name(color_crop)
    symbol_info = detect_symbol_name(symbol_crop)
    bottom_layout = detect_points_and_bottom_layout(bottom_crop)

    global_gray = cv2.cvtColor(global_crop, cv2.COLOR_BGR2GRAY)
    bottom_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)

    global_mean, global_std = mean_std_gray(global_crop)
    bottom_mean, bottom_std = mean_std_gray(bottom_crop)

    final_card_id = f"{color_info['detected']}_{bottom_layout['points']}"
    final_score = 50.0 + (symbol_info["score"] * 20.0) + (10.0 if bottom_layout["has_right_icon"] else 0.0)
    final_gap = max(0.0, 100.0 - final_score)
    final_status = "accepted" if final_score >= 55 else "rejected"

    card_match = {
        "bottom_layout": bottom_layout["layout"],
        "candidate_cards": [
            {
                "card_id": final_card_id,
                "score": float(final_score),
                "bottom_visual": 0.0,
                "global_visual": 0.0,
                "details": [
                    "heuristic_color",
                    "heuristic_symbol",
                    "heuristic_bottom_layout",
                ],
                "expected_bottom": {
                    "layout": bottom_layout["layout"],
                    "points": bottom_layout["points"],
                    "target": bottom_layout["target"],
                    "range": bottom_layout["range"],
                    "has_special_white_panel": bottom_layout["has_special_white_panel"],
                    "has_slash": bottom_layout["has_slash"],
                    "has_right_icon": bottom_layout["has_right_icon"],
                    "has_bottom_line": bottom_layout["has_bottom_line"],
                },
            }
        ],
        "color_name": color_info["detected"],
        "symbol_name": symbol_info["name"],
        "symbol_source": symbol_info["mode"],
        "points": bottom_layout["points"],
        "final_card_id": final_card_id,
        "final_score": float(final_score),
        "final_gap": float(final_gap),
        "final_status": final_status,
        "reason": "heuristic_match",
    }

    result = {
        "color_name": color_info["detected"],
        "symbol_name": symbol_info["name"],
        "points": bottom_layout["points"],
        "bottom_layout": bottom_layout["layout"],
        "final_card_id": final_card_id,
        "final_score": float(final_score),
        "final_gap": float(final_gap),
        "final_status": final_status,
        "rects": [{
            "x": 0,
            "y": 0,
            "w": int(card_img.shape[1]),
            "h": int(card_img.shape[0]),
            "quad": [[0, 0], [int(card_img.shape[1]) - 1, 0], [int(card_img.shape[1]) - 1, int(card_img.shape[0]) - 1], [0, int(card_img.shape[0]) - 1]],
        }],
        "rois": rois,
        "card_match": card_match,
        "signature": {
            "color": color_info,
            "symbol": symbol_info,
            "points": {
                "digit": int(bottom_layout["points"]),
                "raw_digit": int(bottom_layout["points"]),
                "found": True,
                "gap": 0.0,
                "score": 0.0,
                "mean": 0.0,
                "std": 0.0,
            },
            "bottom_layout": bottom_layout,
            "bottom": {
                "mean": bottom_mean,
                "std": bottom_std,
                "vector": signature_vector(bottom_gray),
            },
            "global": {
                "mean": global_mean,
                "std": global_std,
                "vector": signature_vector(global_gray),
            },
            "card_match": card_match,
        },
    }

    debug_images = {
        "card": card_img,
        "color_crop": color_crop,
        "symbol_crop": symbol_crop,
        "bottom_crop": bottom_crop,
        "global_crop": global_crop,
    }

    return np_to_builtin(result), debug_images


# =========================
# Export HTML / ZIP
# =========================
def build_result_html(run_id: str, analysis: dict) -> str:
    cards = analysis.get("cards", [])
    summary = analysis.get("summary", {})

    parts = []
    parts.append("<!doctype html>")
    parts.append("<html lang='fr'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>SpaceLab - Résultat</title>")
    parts.append("<style>")
    parts.append("body{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;}")
    parts.append(".wrap{max-width:1100px;margin:0 auto;}")
    parts.append(".card{background:#111827;border:1px solid #334155;border-radius:16px;padding:18px;margin:16px 0;}")
    parts.append(".muted{color:#94a3b8;}")
    parts.append(".pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#1e293b;margin-right:8px;margin-bottom:8px;}")
    parts.append("a.button{display:inline-block;padding:12px 16px;border-radius:12px;background:#2563eb;color:white;text-decoration:none;font-weight:700;}")
    parts.append("pre{white-space:pre-wrap;word-wrap:break-word;background:#020617;padding:16px;border-radius:12px;overflow:auto;}")
    parts.append("img{max-width:100%;border-radius:12px;border:1px solid #334155;}")
    parts.append("</style></head><body><div class='wrap'>")

    parts.append("<h1>Résultat SpaceLab</h1>")
    parts.append(f"<p class='muted'>Run ID: {html_escape(run_id)}<br>Créé le: {html_escape(analysis.get('created_at', ''))}</p>")

    status = summary.get("status", "unknown")
    count = summary.get("detected_cards", 0)
    parts.append("<div class='card'>")
    parts.append("<h2>Résumé</h2>")
    parts.append(f"<div class='pill'>Statut: {html_escape(str(status))}</div>")
    parts.append(f"<div class='pill'>Cartes détectées: {html_escape(str(count))}</div>")
    parts.append(f"<div style='margin-top:12px'><a class='button' href='/export/{html_escape(run_id)}.zip'>Exporter image + résultat + JSON + HTML</a></div>")
    parts.append("</div>")

    if analysis.get("files", {}).get("board_detected"):
        parts.append("<div class='card'>")
        parts.append("<h2>Détection plateau</h2>")
        parts.append(f"<img src='/runs/{html_escape(run_id)}/board_detected.png' alt='Plateau détecté'>")
        parts.append("</div>")

    for card in cards:
        idx = card.get("index")
        result = card.get("result", {})
        parts.append("<div class='card'>")
        parts.append(f"<h2>Carte {idx}</h2>")
        parts.append(f"<div class='pill'>ID: {html_escape(str(result.get('final_card_id')))}</div>")
        parts.append(f"<div class='pill'>Couleur: {html_escape(str(result.get('color_name')))}</div>")
        parts.append(f"<div class='pill'>Symbole: {html_escape(str(result.get('symbol_name')))}</div>")
        parts.append(f"<div class='pill'>Points: {html_escape(str(result.get('points')))}</div>")
        parts.append(f"<div class='pill'>Layout: {html_escape(str(result.get('bottom_layout')))}</div>")
        parts.append(f"<div class='pill'>Score: {html_escape(str(result.get('final_score')))}</div>")
        parts.append(f"<div class='pill'>Statut: {html_escape(str(result.get('final_status')))}</div>")
        parts.append(f"<p><img src='/runs/{html_escape(run_id)}/card_{idx}.png' alt='Carte {idx}'></p>")
        parts.append("</div>")

    parts.append("<div class='card'>")
    parts.append("<h2>JSON complet</h2>")
    parts.append(f"<pre>{html_escape(json.dumps(analysis, ensure_ascii=False, indent=2))}</pre>")
    parts.append("</div>")

    parts.append("</div></body></html>")
    return "".join(parts)


# =========================
# Pipeline complète
# =========================
def run_analysis(image: np.ndarray, original_filename: str):
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    original_path = run_dir / "original.png"
    save_image(original_path, image)

    cards, board_debug = detect_cards_on_board(image, debug=True)

    files = {
        "original": "original.png",
    }

    if "board_detected" in board_debug:
        save_image(run_dir / "board_detected.png", board_debug["board_detected"])
        files["board_detected"] = "board_detected.png"
    if "edges1" in board_debug:
        save_image(run_dir / "edges1.png", board_debug["edges1"])
        files["edges1"] = "edges1.png"
    if "edges2" in board_debug:
        save_image(run_dir / "edges2.png", board_debug["edges2"])
        files["edges2"] = "edges2.png"
    if "threshold" in board_debug:
        save_image(run_dir / "threshold.png", board_debug["threshold"])
        files["threshold"] = "threshold.png"

    card_results = []
    for idx, card in enumerate(cards, start=1):
        card_img = card["warp"]
        save_image(run_dir / f"card_{idx}.png", card_img)

        result, debug_images = analyze_single_card(card_img)
        for debug_name, debug_img in debug_images.items():
            save_image(run_dir / f"card_{idx}_{debug_name}.png", debug_img)

        card_results.append({
            "index": idx,
            "rect": card["rect"],
            "quad": card["quad"],
            "result": result,
        })

    status = "accepted" if len(card_results) > 0 else "no_card_detected"
    message = "Analyse terminée" if len(card_results) > 0 else "Aucune carte détectée sur le plateau"

    analysis = {
        "run_id": run_id,
        "created_at": now_iso(),
        "source_filename": original_filename,
        "summary": {
            "status": status,
            "message": message,
            "detected_cards": len(card_results),
        },
        "files": files,
        "cards": card_results,
    }

    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(np_to_builtin(analysis), ensure_ascii=False, indent=2), encoding="utf-8")

    html_content = build_result_html(run_id, analysis)
    html_path = run_dir / "result.html"
    html_path.write_text(html_content, encoding="utf-8")

    return np_to_builtin(analysis)


# =========================
# Routes Flask
# =========================
@app.route("/", methods=["GET"])
def home():
    return """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpaceLab</title>
  <style>
    body{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;}
    .wrap{max-width:980px;margin:0 auto;}
    .card{background:#111827;border:1px solid #334155;border-radius:18px;padding:20px;margin:18px 0;}
    h1,h2{margin-top:0;}
    input[type=file]{display:block;margin:14px 0;padding:12px;background:#0b1220;color:#e2e8f0;border:1px solid #334155;border-radius:12px;width:100%;}
    button{padding:14px 18px;border:none;border-radius:12px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer;}
    .muted{color:#94a3b8;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    @media (max-width:800px){.grid{grid-template-columns:1fr;}}
    a{color:#93c5fd;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>SpaceLab</h1>
      <p class="muted">Upload une image de plateau. L'app détecte plusieurs cartes, redresse chaque carte, produit le résultat, le JSON et une page HTML exportable.</p>
      <form action="/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Analyser l'image</button>
      </form>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Ce que fait ce fichier</h2>
        <ul>
          <li>pas de Tkinter</li>
          <li>compatible Railway / serveur</li>
          <li>détection plateau multi-cartes</li>
          <li>crop de chaque carte</li>
          <li>résultat JSON</li>
          <li>page HTML</li>
          <li>export ZIP en un clic</li>
        </ul>
      </div>
      <div class="card">
        <h2>Export complet</h2>
        <p class="muted">Après analyse, tu auras un bouton pour exporter en une seule archive :</p>
        <ul>
          <li>l'image envoyée</li>
          <li>le résultat JSON</li>
          <li>la page HTML</li>
          <li>les crops et images debug</li>
        </ul>
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "image" not in request.files:
        return Response("Aucun fichier reçu", status=400)

    file = request.files["image"]
    if not file or not file.filename:
        return Response("Aucun fichier sélectionné", status=400)

    if not allowed_file(file.filename):
        return Response("Format de fichier non supporté", status=400)

    data = file.read()
    image = read_image_from_bytes(data)
    if image is None:
        return Response("Impossible de lire l'image", status=400)

    try:
        analysis = run_analysis(image, file.filename)
        run_id = analysis["run_id"]
        return redirect(url_for("view_result", run_id=run_id))
    except Exception:
        tb = traceback.format_exc()
        return Response(f"Erreur pendant l'analyse\n\n{tb}", status=500, mimetype="text/plain")


@app.route("/result/<run_id>", methods=["GET"])
def view_result(run_id):
    run_dir = RUNS_DIR / run_id
    html_path = run_dir / "result.html"
    if not html_path.exists():
        return Response("Résultat introuvable", status=404)
    return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/json/<run_id>", methods=["GET"])
def view_json(run_id):
    run_dir = RUNS_DIR / run_id
    json_path = run_dir / "result.json"
    if not json_path.exists():
        return Response("JSON introuvable", status=404)
    return Response(json_path.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/runs/<run_id>/<filename>", methods=["GET"])
def serve_run_file(run_id, filename):
    run_dir = RUNS_DIR / run_id
    path = run_dir / filename
    if not path.exists() or not path.is_file():
        return Response("Fichier introuvable", status=404)
    return send_file(path)


@app.route("/export/<run_id>.zip", methods=["GET"])
def export_zip(run_id):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return Response("Run introuvable", status=404)

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.glob("**/*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(run_dir))

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"spacelab_{run_id}.zip",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "time": now_iso()})


# =========================
# Main Railway / local
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
