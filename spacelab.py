import os
import io
import json
import math
import uuid
import base64
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    jsonify,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename

# ============================================================
# SpaceLab - version Flask simple et cohérente
# ============================================================
# Objectif:
# - Démarrer proprement sur Railway / serveur Linux
# - Upload d'image
# - Détection plateau multi-cartes
# - Découpage de cartes candidates
# - Matching basique par références si présentes
# - Affichage HTML du résultat
# - Export JSON et HTML
#
# Cette version évite Tkinter, fonctionne en web, et reste
# modulaire pour pouvoir améliorer l'algorithme ensuite.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
REF_DIR = BASE_DIR / "refs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_IMAGE_SIDE = 2200
MIN_CARD_AREA_RATIO = 0.003
MAX_CARD_AREA_RATIO = 0.35
CARD_ASPECT_MIN = 0.52
CARD_ASPECT_MAX = 0.86

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# Utilitaires
# ============================================================


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image vide")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image



def resize_for_processing(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(side)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale



def image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")



def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)



def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered



def warp_quad(image: np.ndarray, quad: np.ndarray, out_w: int = 300, out_h: int = 420) -> np.ndarray:
    quad = order_quad_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))



def contour_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32)



def clip_rect(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h



def safe_crop(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    x, y, w, h = clip_rect(x, y, w, h, image.shape[1], image.shape[0])
    return image[y:y + h, x:x + w].copy()



def compute_gray_signature(image: np.ndarray, size: Tuple[int, int] = (16, 16)) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    vector = resized.flatten().astype(np.float32)
    return {
        "mean": float(vector.mean()),
        "std": float(vector.std()),
        "vector": [int(v) for v in vector.tolist()],
    }



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))



def normalize_vector(v: List[int]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    if arr.std() > 1e-6:
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr


# ============================================================
# Structures de données
# ============================================================

@dataclass
class CardCandidate:
    index: int
    rect: Dict[str, int]
    quad: List[List[int]]
    area: float
    aspect_ratio: float
    card_image_b64: str
    match: Dict[str, Any]


@dataclass
class AnalysisResult:
    analysis_id: str
    created_at: str
    image_filename: str
    image_width: int
    image_height: int
    cards_found: int
    cards: List[Dict[str, Any]]
    debug: Dict[str, Any]


# ============================================================
# Chargement des références
# ============================================================


def load_reference_library() -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not REF_DIR.exists():
        return refs

    for path in sorted(REF_DIR.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        image = ensure_bgr(image)
        warped = cv2.resize(image, (300, 420), interpolation=cv2.INTER_AREA)

        bottom = extract_bottom_roi(warped)
        global_sig = compute_gray_signature(warped, (16, 16))
        bottom_sig = compute_gray_signature(bottom, (16, 8))

        refs.append(
            {
                "card_id": path.stem,
                "path": str(path.relative_to(BASE_DIR)),
                "global_sig": global_sig,
                "bottom_sig": bottom_sig,
            }
        )
    return refs


REFERENCE_LIBRARY = load_reference_library()


# ============================================================
# Détection des cartes
# ============================================================


def extract_bottom_roi(card_image: np.ndarray) -> np.ndarray:
    h, w = card_image.shape[:2]
    y0 = int(h * 0.76)
    return card_image[y0:h, 0:w].copy()



def detect_card_candidates(image: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    work, scale = resize_for_processing(image, MAX_IMAGE_SIDE)
    h, w = work.shape[:2]
    img_area = float(w * h)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_image = work.copy()
    candidates: List[Dict[str, Any]] = []

    min_area = img_area * MIN_CARD_AREA_RATIO
    max_area = img_area * MAX_CARD_AREA_RATIO

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        quad = contour_to_quad(contour)
        if quad is None:
            continue

        x, y, ww, hh = cv2.boundingRect(quad.astype(np.int32))
        aspect = ww / float(hh + 1e-6)
        if not (CARD_ASPECT_MIN <= aspect <= CARD_ASPECT_MAX):
            continue

        warped = warp_quad(work, quad, 300, 420)
        mean_brightness = float(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).mean())
        if mean_brightness < 35:
            continue

        quad_int = quad.astype(int)
        cv2.polylines(debug_image, [quad_int], True, (0, 255, 0), 3)
        cv2.putText(
            debug_image,
            f"{len(candidates)+1}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        candidates.append(
            {
                "rect": {"x": int(x / scale), "y": int(y / scale), "w": int(ww / scale), "h": int(hh / scale)},
                "quad": [[int(px / scale), int(py / scale)] for px, py in quad.tolist()],
                "area": float(area / (scale * scale)),
                "aspect_ratio": float(aspect),
                "warped": warped,
            }
        )

    candidates = deduplicate_candidates(candidates)
    candidates.sort(key=lambda c: (c["rect"]["y"], c["rect"]["x"]))

    debug = {
        "scale": scale,
        "image_size_processed": {"w": w, "h": h},
        "contours_total": len(contours),
        "candidates_after_filter": len(candidates),
        "thresholds": {
            "min_area": min_area,
            "max_area": max_area,
            "aspect_min": CARD_ASPECT_MIN,
            "aspect_max": CARD_ASPECT_MAX,
        },
    }
    return candidates, debug, debug_image



def rect_iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / float(union + 1e-6)



def deduplicate_candidates(candidates: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c["area"], reverse=True):
        overlaps = [rect_iou(cand["rect"], k["rect"]) for k in kept]
        if overlaps and max(overlaps) > iou_threshold:
            continue
        kept.append(cand)
    return kept


# ============================================================
# Matching simple avec références
# ============================================================


def match_card(card_image: np.ndarray) -> Dict[str, Any]:
    bottom = extract_bottom_roi(card_image)
    card_sig = compute_gray_signature(card_image, (16, 16))
    bottom_sig = compute_gray_signature(bottom, (16, 8))

    if not REFERENCE_LIBRARY:
        return {
            "status": "no_reference_library",
            "final_card_id": None,
            "final_score": 0.0,
            "candidates": [],
            "signature": {
                "global": card_sig,
                "bottom": bottom_sig,
            },
        }

    gv = normalize_vector(card_sig["vector"])
    bv = normalize_vector(bottom_sig["vector"])

    ranked = []
    for ref in REFERENCE_LIBRARY:
        ref_gv = normalize_vector(ref["global_sig"]["vector"])
        ref_bv = normalize_vector(ref["bottom_sig"]["vector"])
        global_score = cosine_similarity(gv, ref_gv)
        bottom_score = cosine_similarity(bv, ref_bv)
        score = 0.45 * global_score + 0.55 * bottom_score
        ranked.append(
            {
                "card_id": ref["card_id"],
                "global_visual": float(global_score),
                "bottom_visual": float(bottom_score),
                "score": float(score * 100.0),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    gap = best["score"] - second["score"] if second else best["score"]

    if best["score"] >= 82:
        status = "accepted"
        reason = "strong_match"
    elif best["score"] >= 72 and gap >= 6:
        status = "accepted"
        reason = "good_unique_match"
    elif best["score"] >= 60:
        status = "uncertain"
        reason = "weak_match"
    else:
        status = "rejected"
        reason = "too_low"

    return {
        "status": status,
        "reason": reason,
        "final_card_id": best["card_id"],
        "final_score": float(best["score"]),
        "final_gap": float(gap),
        "candidates": ranked[:5],
        "signature": {
            "global": card_sig,
            "bottom": bottom_sig,
        },
    }


# ============================================================
# Analyse complète
# ============================================================


def analyze_image(image: np.ndarray, filename: str) -> AnalysisResult:
    image = ensure_bgr(image)
    analysis_id = uuid.uuid4().hex[:12]

    candidates, detect_debug, debug_annotated = detect_card_candidates(image)

    cards: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        card_image = cand.pop("warped")
        match = match_card(card_image)
        cards.append(
            asdict(
                CardCandidate(
                    index=idx,
                    rect=cand["rect"],
                    quad=cand["quad"],
                    area=float(cand["area"]),
                    aspect_ratio=float(cand["aspect_ratio"]),
                    card_image_b64=image_to_base64(card_image),
                    match=match,
                )
            )
        )

    result = AnalysisResult(
        analysis_id=analysis_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        image_filename=filename,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        cards_found=len(cards),
        cards=cards,
        debug={
            "reference_count": len(REFERENCE_LIBRARY),
            "detection": detect_debug,
            "annotated_image_b64": image_to_base64(debug_annotated),
        },
    )

    save_analysis(result)
    return result



def save_analysis(result: AnalysisResult) -> None:
    path = RESULT_DIR / f"{result.analysis_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)



def load_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    path = RESULT_DIR / f"{analysis_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Templates HTML
# ============================================================

INDEX_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpaceLab</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    h1, h2, h3 { margin-top: 0; }
    .btn { display: inline-block; background: #2563eb; color: white; text-decoration: none; padding: 12px 16px; border-radius: 10px; border: none; cursor: pointer; }
    .btn.secondary { background: #475569; }
    .btn.green { background: #059669; }
    input[type=file] { padding: 10px; background: #0b1220; color: #fff; border: 1px solid #334155; border-radius: 10px; width: 100%; }
    .muted { color: #94a3b8; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
    .accepted { background: #14532d; color: #dcfce7; }
    .uncertain { background: #78350f; color: #fef3c7; }
    .rejected { background: #7f1d1d; color: #fee2e2; }
    .small { font-size: 13px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SpaceLab</h1>
    <div class="card">
      <h2>Analyser un plateau</h2>
      <p class="muted">Envoie une image du plateau. L'application cherche plusieurs cartes, les découpe, puis tente un matching avec les références présentes dans <code>refs/</code>.</p>
      <form action="{{ url_for('analyze_route') }}" method="post" enctype="multipart/form-data">
        <p><input type="file" name="image" accept="image/*" required></p>
        <p>
          <button class="btn" type="submit">Analyser l'image</button>
        </p>
      </form>
    </div>

    <div class="card">
      <h3>État</h3>
      <p class="small">Références chargées: <strong>{{ reference_count }}</strong></p>
      <p class="small">Dossiers attendus: <code>refs/</code>, <code>uploads/</code>, <code>results/</code></p>
      <p class="small muted">Si aucune référence n'est présente, la détection de cartes fonctionne quand même, mais le nom exact de carte ne pourra pas être confirmé.</p>
    </div>
  </div>
</body>
</html>
"""


RESULT_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Résultat SpaceLab</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    .btn { display: inline-block; background: #2563eb; color: white; text-decoration: none; padding: 10px 14px; border-radius: 10px; margin-right: 10px; }
    .btn.secondary { background: #475569; }
    .btn.green { background: #059669; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #334155; }
    pre { white-space: pre-wrap; word-break: break-word; background: #020617; border: 1px solid #334155; padding: 12px; border-radius: 12px; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
    .accepted { background: #14532d; color: #dcfce7; }
    .uncertain { background: #78350f; color: #fef3c7; }
    .rejected { background: #7f1d1d; color: #fee2e2; }
    .small { font-size: 13px; }
    .muted { color: #94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Résultat d'analyse</h1>
      <p class="small muted">ID: {{ result.analysis_id }} · Fichier: {{ result.image_filename }}</p>
      <p>
        <a class="btn" href="{{ url_for('index') }}">Nouvelle analyse</a>
        <a class="btn secondary" href="{{ url_for('export_json_route', analysis_id=result.analysis_id) }}">Exporter JSON</a>
        <a class="btn green" href="{{ url_for('export_html_route', analysis_id=result.analysis_id) }}">Exporter HTML</a>
        <a class="btn secondary" href="{{ url_for('exploration_route', analysis_id=result.analysis_id) }}">Exploration JSON + HTML</a>
      </p>
    </div>

    <div class="card">
      <h2>Résumé</h2>
      <p>Cartes trouvées: <strong>{{ result.cards_found }}</strong></p>
      {% if result.cards_found == 0 %}
        <p>Aucune carte détectée sur le plateau.</p>
      {% endif %}
    </div>

    <div class="card">
      <h2>Détection annotée</h2>
      <img src="data:image/jpeg;base64,{{ result.debug.annotated_image_b64 }}" alt="annotated">
    </div>

    <div class="grid">
      {% for card in result.cards %}
      <div class="card">
        <h3>Carte {{ card.index }}</h3>
        <p>
          <span class="pill {{ card.match.status }}">{{ card.match.status }}</span>
          {% if card.match.final_card_id %}
            <strong>{{ card.match.final_card_id }}</strong>
          {% else %}
            <strong>Non reconnue</strong>
          {% endif %}
        </p>
        <img src="data:image/jpeg;base64,{{ card.card_image_b64 }}" alt="card-{{ card.index }}">
        <p class="small">Score: {{ '%.2f'|format(card.match.final_score or 0) }} · Gap: {{ '%.2f'|format(card.match.final_gap or 0) }}</p>
        <p class="small">Rect: x={{ card.rect.x }}, y={{ card.rect.y }}, w={{ card.rect.w }}, h={{ card.rect.h }}</p>
        <details>
          <summary>Top candidats</summary>
          <pre>{{ card.match.candidates | tojson(indent=2) }}</pre>
        </details>
      </div>
      {% endfor %}
    </div>

    <div class="card">
      <h2>Debug</h2>
      <pre>{{ result.debug | tojson(indent=2) }}</pre>
    </div>
  </div>
</body>
</html>
"""


EXPLORATION_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exploration SpaceLab</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #020617; color: #e2e8f0; }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #334155; }
    pre { white-space: pre-wrap; word-break: break-word; background: #000; padding: 12px; border-radius: 12px; border: 1px solid #334155; }
    .btn { display: inline-block; background: #2563eb; color: white; text-decoration: none; padding: 10px 14px; border-radius: 10px; margin-right: 10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Exploration complète</h1>
      <p>
        <a class="btn" href="{{ url_for('result_route', analysis_id=result.analysis_id) }}">Voir résultat</a>
        <a class="btn" href="{{ url_for('export_json_route', analysis_id=result.analysis_id) }}">Télécharger JSON</a>
        <a class="btn" href="{{ url_for('export_html_route', analysis_id=result.analysis_id) }}">Télécharger HTML</a>
      </p>
    </div>

    <div class="row">
      <div class="card">
        <h2>Image annotée</h2>
        <img src="data:image/jpeg;base64,{{ result.debug.annotated_image_b64 }}" alt="annotated">
      </div>
      <div class="card">
        <h2>JSON</h2>
        <pre>{{ result | tojson(indent=2) }}</pre>
      </div>
    </div>
  </div>
</body>
</html>
"""


# ============================================================
# Routes Flask
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, reference_count=len(REFERENCE_LIBRARY))


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "service": "spacelab",
            "references": len(REFERENCE_LIBRARY),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "image" not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files["image"]
    if not file or file.filename == "":
        return "Fichier vide", 400

    if not allowed_file(file.filename):
        return "Format non supporté", 400

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    upload_path = UPLOAD_DIR / unique_name
    file.save(str(upload_path))

    image = cv2.imread(str(upload_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return "Impossible de lire l'image", 400

    try:
        result = analyze_image(image, original_name)
    except Exception as exc:
        return f"Erreur pendant l'analyse: {exc}", 500

    return redirect(url_for("result_route", analysis_id=result.analysis_id))


@app.route("/result/<analysis_id>", methods=["GET"])
def result_route(analysis_id: str):
    result = load_analysis(analysis_id)
    if not result:
        abort(404)
    return render_template_string(RESULT_HTML, result=result)


@app.route("/exploration/<analysis_id>", methods=["GET"])
def exploration_route(analysis_id: str):
    result = load_analysis(analysis_id)
    if not result:
        abort(404)
    return render_template_string(EXPLORATION_HTML, result=result)


@app.route("/api/result/<analysis_id>", methods=["GET"])
def api_result_route(analysis_id: str):
    result = load_analysis(analysis_id)
    if not result:
        return jsonify({"error": "not_found"}), 404
    return jsonify(result)


@app.route("/export/json/<analysis_id>", methods=["GET"])
def export_json_route(analysis_id: str):
    path = RESULT_DIR / f"{analysis_id}.json"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/json", as_attachment=True, download_name=f"spacelab_{analysis_id}.json")


@app.route("/export/html/<analysis_id>", methods=["GET"])
def export_html_route(analysis_id: str):
    result = load_analysis(analysis_id)
    if not result:
        abort(404)
    html = render_template_string(RESULT_HTML, result=result)
    return send_file(
        io.BytesIO(html.encode("utf-8")),
        mimetype="text/html",
        as_attachment=True,
        download_name=f"spacelab_{analysis_id}.html",
    )


@app.route("/reload_refs", methods=["POST", "GET"])
def reload_refs_route():
    global REFERENCE_LIBRARY
    REFERENCE_LIBRARY = load_reference_library()
    return jsonify({"ok": True, "references": len(REFERENCE_LIBRARY)})


# ============================================================
# Main Railway / local
# ============================================================

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
