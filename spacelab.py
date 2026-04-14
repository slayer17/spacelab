import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import base64
import shutil
from datetime import datetime

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

from bottom import (
    extract_bottom_roi_from_full_card,
    analyze_bottom,
    _normalize_badge,
    _extract_digit_mask,
    build_overlay,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, "cards")
DIGITS_DIR = os.path.join(BASE_DIR, "digits")
CARDS_JS_PATH = os.path.join(BASE_DIR, "cards.js")
WARP_PATH = os.path.join(BASE_DIR, "warp.jpg")


def _decode_image_bytes_with_orientation(raw_bytes):
    if not raw_bytes:
        return None
    if Image is not None and ImageOps is not None:
        try:
            import io
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _read_image_file_with_orientation(path):
    with open(path, "rb") as f:
        return _decode_image_bytes_with_orientation(f.read())


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
    return f'''
    <div style="margin-bottom:20px;">
      <h3 style="margin:0 0 8px 0;">{title}</h3>
      <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ccc;" />
    </div>
    '''


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
    base = str(card_id).lower()
    for ext in [".jpeg", ".jpg", ".png"]:
        path = os.path.join(CARDS_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


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
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))


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
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * 0.15 or area > image_area * 0.98:
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
    return {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "quad": quad.astype(int).tolist()}


def compute_patch_signature(zone, size=(16, 16)):
    if zone is None or zone.size == 0:
        return {"mean": 0.0, "std": 0.0, "vector": []}
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return {"mean": float(np.mean(gray)), "std": float(np.std(gray)), "vector": small.flatten().astype(float).tolist()}


def detect_card_color(zone):
    if zone is None or zone.size == 0:
        return "ROUGE", {"reason": "empty"}, [0.0, 0.0, 0.0]
    mean_bgr = zone.mean(axis=(0, 1)).tolist()
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
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
    return max(counts, key=counts.get), counts, mean_bgr


def _extract_symbol_zone_from_card(img):
    img = cv2.resize(img, (200, 300))
    h, w = img.shape[:2]
    return img[int(h * 0.16):int(h * 0.34), int(w * 0.02):int(w * 0.24)]


def compute_signature(img):
    img = cv2.resize(img, (200, 300))
    rois = []
    h, w = img.shape[:2]

    color_zone = img[0:int(h * 0.18), 0:int(w * 0.38)]
    rois.append({"type": "COLOR", "x": 0, "y": 0, "w": int(w * 0.38), "h": int(h * 0.18)})
    gray = cv2.cvtColor(color_zone, cv2.COLOR_BGR2GRAY)
    detected_color, color_debug, mean_bgr = detect_card_color(color_zone)

    symbol_zone = _extract_symbol_zone_from_card(img)
    rois.append({"type": "SYMBOL", "x": int(w * 0.02), "y": int(h * 0.16), "w": int(w * 0.22), "h": int(h * 0.18)})
    symbol_gray = cv2.cvtColor(symbol_zone, cv2.COLOR_BGR2GRAY)

    full_img, bottom_zone, bottom_box = extract_bottom_roi_from_full_card(img)
    bx, by, bw2, bh2 = bottom_box
    rois.append({"type": "BOTTOM", "x": bx, "y": by, "w": bw2, "h": bh2})

    bottom_layout = analyze_bottom(bottom_zone, DIGITS_DIR)
    rois.append({"type": "GLOBAL", "x": 0, "y": 0, "w": w, "h": h})

    return {
        "color": {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "color": mean_bgr,
            "detected": detected_color,
            "debug": color_debug,
        },
        "symbol": {
            "mean": float(np.mean(symbol_gray)),
            "std": float(np.std(symbol_gray)),
            "name": None,
            "raw_name": None,
            "score": 0.0,
            "gap": 0.0,
            "threshold_mode": "no_match",
            "top_candidates": [],
            "winner_references": [],
            "runner_up": None,
            "mode": "icon_card_refs",
        },
        "points": {
            "mean": 0.0,
            "std": 0.0,
            "digit": bottom_layout.get("points"),
            "raw_digit": bottom_layout.get("raw_points"),
            "score": float(bottom_layout.get("points_score", 0.0)),
            "gap": float(bottom_layout.get("points_gap", 0.0)),
            "found": bool(bottom_layout.get("points_box") is not None),
        },
        "bottom": compute_patch_signature(bottom_zone, size=(16, 16)),
        "bottom_layout": {
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
        },
        "global": compute_patch_signature(img, size=(16, 16)),
    }, rois


def resolve_final_card(scan_sig, cards=None):
    if cards is None:
        try:
            cards = load_cards_js()
        except Exception:
            cards = []
    if not cards:
        return {"color_name": None, "symbol_name": None, "symbol_source": "none", "bottom_layout": None, "points": None, "candidate_cards": [], "final_card_id": None, "final_score": 0.0, "final_gap": 0.0, "final_status": "rejected", "reason": "no_cards_reference"}

    color_name = (scan_sig.get("color") or {}).get("detected")
    bottom_layout = (scan_sig.get("bottom_layout") or {}).get("layout")
    points = (scan_sig.get("bottom_layout") or {}).get("points")

    candidates = []
    for card in cards:
        score = 0.0
        if color_name and str(card.get("couleur", "")).upper() == str(color_name).upper():
            score += 5.0
        if points is not None and str(card.get("points", "")) == str(points):
            score += 3.0
        candidates.append({"card_id": str(card.get("id") or ""), "score": float(score), "details": []})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0] if candidates else None
    runner = candidates[1] if len(candidates) > 1 else None
    gap = float((best["score"] - runner["score"]) if best and runner else (best["score"] if best else 0.0))

    if best and best["score"] >= 5.0:
        status = "accepted" if gap >= 1.0 else "fragile"
        final_card_id = best["card_id"]
    else:
        status = "rejected"
        final_card_id = None

    return {
        "color_name": color_name,
        "symbol_name": None,
        "symbol_source": "none",
        "bottom_layout": bottom_layout,
        "points": points,
        "candidate_cards": candidates[:8],
        "final_card_id": final_card_id,
        "final_score": float(best["score"]) if best else 0.0,
        "final_gap": gap,
        "final_status": status,
        "reason": "simplified_match",
    }


def _recognize_card_in_crop(crop):
    if crop is None or crop.size == 0:
        return {"ok": False, "reason": "empty_crop", "rect": None, "signature": None, "rois": [], "card_match": None, "final_card_id": None, "final_status": None, "final_score": 0.0, "final_gap": 0.0}
    try:
        rect = detect_main_card(crop)
    except Exception:
        rect = None
    if rect is None:
        h, w = crop.shape[:2]
        rect = {"x": 0, "y": 0, "w": int(w), "h": int(h), "quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]}
    quad = np.array(rect["quad"], dtype="float32")
    warped = warp_quad(crop, quad)
    if warped is None or warped.size == 0:
        warped = crop.copy()
    sig, rois = compute_signature(warped)
    card_match = resolve_final_card(sig)
    sig["card_match"] = card_match
    return {
        "ok": True,
        "reason": None,
        "rect": rect,
        "signature": sig,
        "rois": rois,
        "card_match": card_match,
        "final_card_id": card_match.get("final_card_id"),
        "final_status": card_match.get("final_status"),
        "final_score": float(card_match.get("final_score", 0.0)),
        "final_gap": float(card_match.get("final_gap", 0.0)),
        "color_name": card_match.get("color_name"),
        "symbol_name": card_match.get("symbol_name"),
        "bottom_layout": card_match.get("bottom_layout"),
        "points": card_match.get("points"),
    }


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({"mode": "single", "rects": [], "signature": None, "rois": [], "card_match": None, "final_card_id": None, "final_status": None, "final_score": 0.0, "final_gap": 0.0})
        raw = request.files['image'].read()
        img = _decode_image_bytes_with_orientation(raw)
        if img is None:
            return jsonify({"mode": "single", "rects": [], "signature": None, "rois": [], "card_match": None, "final_card_id": None, "final_status": None, "final_score": 0.0, "final_gap": 0.0})
        recognized = _recognize_card_in_crop(img)
        return jsonify({
            "mode": "single",
            "rects": [recognized.get("rect")] if recognized.get("rect") else [],
            "signature": recognized.get("signature"),
            "rois": recognized.get("rois") or [],
            "card_match": recognized.get("card_match"),
            "final_card_id": recognized.get("final_card_id"),
            "final_status": recognized.get("final_status"),
            "final_score": float(recognized.get("final_score", 0.0)),
            "final_gap": float(recognized.get("final_gap", 0.0)),
            "color_name": recognized.get("color_name"),
            "symbol_name": recognized.get("symbol_name"),
            "bottom_layout": recognized.get("bottom_layout"),
            "points": recognized.get("points"),
        })
    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"mode": "single", "rects": [], "signature": None, "rois": [], "card_match": None, "final_card_id": None, "final_status": None, "final_score": 0.0, "final_gap": 0.0})


@app.route('/bottom-test', methods=['GET', 'POST'])
def bottom_test():
    if request.method == 'GET':
        return '''
        <html><head><meta charset="utf-8" /><title>Bottom Test</title></head>
        <body style="font-family:Arial,sans-serif; padding:20px;">
          <h1>Test du bas de carte</h1>
          <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required />
            <button type="submit">Analyser</button>
          </form>
        </body></html>
        '''

    file = request.files.get('image')
    if not file:
        return 'Aucun fichier envoyé', 400
    img = _decode_image_bytes_with_orientation(file.read())
    if img is None:
        return 'Image invalide', 400

    full_img, bottom_roi, bottom_box = extract_bottom_roi_from_full_card(img)
    result = analyze_bottom(bottom_roi, DIGITS_DIR)
    overlay = build_overlay(full_img, bottom_box, result)
    points_crop = None
    badge_norm = None
    digit_mask = None
    points_box = result.get('points_box')
    if points_box is not None:
        x, y, w, h = points_box
        points_crop = bottom_roi[y:y + h, x:x + w]
        if points_crop is not None and points_crop.size != 0:
            badge_norm = _normalize_badge(points_crop)
            digit_mask = _extract_digit_mask(points_crop)

    pretty_json = json.dumps(result, indent=2, ensure_ascii=False)
    return f'''
    <html><head><meta charset="utf-8" /><title>Bottom Test</title></head>
    <body style="font-family:Arial,sans-serif; padding:20px;">
      <h1>Résultat du test du bas</h1>
      <p><a href="/bottom-test">← Revenir au formulaire</a></p>
      <pre style="background:#f5f5f5; padding:12px; border:1px solid #ddd; overflow:auto;">{pretty_json}</pre>
      {_html_img_block('Image complète', _img_to_base64(full_img))}
      {_html_img_block('ROI du bas', _img_to_base64(bottom_roi))}
      {_html_img_block('Overlay debug', _img_to_base64(overlay))}
      {_html_img_block('Crop points', _img_to_base64(points_crop) if points_crop is not None else None)}
      {_html_img_block('Badge normalisé', _img_to_base64(badge_norm) if badge_norm is not None else None)}
      {_html_img_block('Masque du chiffre', _img_to_base64(digit_mask) if digit_mask is not None else None)}
    </body></html>
    '''


@app.route('/build_signatures')
def build_signatures():
    try:
        cards = load_cards_js()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{CARDS_JS_PATH}.bak_{timestamp}"
        shutil.copyfile(CARDS_JS_PATH, backup_path)
        updated = 0
        skipped = []
        errors = []
        for c in cards:
            card_id = c.get('id')
            if not card_id:
                skipped.append({'id': None, 'reason': 'missing_id'})
                continue
            path = find_card_image(card_id)
            if path is None:
                skipped.append({'id': card_id, 'reason': 'image_not_found'})
                continue
            img = _read_image_file_with_orientation(path)
            if img is None or img.size == 0:
                skipped.append({'id': card_id, 'reason': 'image_unreadable'})
                continue
            h, w = img.shape[:2]
            quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
            warped = warp_quad(img, quad)
            if warped is None or warped.size == 0:
                skipped.append({'id': card_id, 'reason': 'warp_failed'})
                continue
            try:
                sig, _ = compute_signature(warped)
            except Exception as e:
                errors.append({'id': card_id, 'reason': f'compute_signature_failed: {str(e)}'})
                continue
            c['signature'] = {'scan': sig}
            updated += 1
        save_cards_js(cards)
        return jsonify({'ok': True, 'updated': updated, 'total': len(cards), 'skipped_count': len(skipped), 'errors_count': len(errors), 'backup': os.path.basename(backup_path), 'skipped': skipped[:20], 'errors': errors[:20]})
    except Exception as e:
        print('BUILD SIGNATURES ERROR:', e)
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/test')
def test():
    return 'OK TEST'


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/warp')
def warp():
    if not os.path.exists(WARP_PATH):
        return 'warp not found', 404
    return send_from_directory(BASE_DIR, 'warp.jpg')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(BASE_DIR, path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
