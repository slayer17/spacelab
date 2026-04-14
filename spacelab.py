import os
import io
import cv2
import json
import math
import base64
import traceback
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

DEBUG_DIR = "debug_exports"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Taille attendue d'une carte en ratio h / w
CARD_RATIO_MIN = 1.25
CARD_RATIO_MAX = 2.20

# Taille minimale d'une carte détectée
MIN_CARD_W = 60
MIN_CARD_H = 100

# Taille max relative
MAX_CARD_W_RATIO = 0.35
MAX_CARD_H_RATIO = 0.60

# Marge de sécurité pour les crops
PAD_X = 10
PAD_Y = 10

# Fusion des boîtes proches
MERGE_IOU_THRESHOLD = 0.20
MERGE_CENTER_DIST = 70

# Seuils de debug / validation
MIN_REFINED_AREA = 5000
MIN_SCORE_ACCEPT = 0.10


# ============================================================
# DATA
# ============================================================

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

    def x2(self) -> int:
        return self.x + self.w

    def y2(self) -> int:
        return self.y + self.h

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]

    def clip(self, W: int, H: int) -> "Rect":
        x = max(0, min(self.x, W - 1))
        y = max(0, min(self.y, H - 1))
        x2 = max(x + 1, min(self.x + self.w, W))
        y2 = max(y + 1, min(self.y + self.h, H))
        return Rect(x, y, x2 - x, y2 - y)

    def expand(self, px: int, py: int, W: int, H: int) -> "Rect":
        return Rect(
            self.x - px,
            self.y - py,
            self.w + 2 * px,
            self.h + 2 * py
        ).clip(W, H)


@dataclass
class CardDetection:
    search_rect: Rect
    refined_rect: Optional[Rect]
    accepted: bool
    reason: str
    index: int
    analysis: Optional[Dict[str, Any]] = None


# ============================================================
# UTILS
# ============================================================

def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image vide")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {path}")
    return img


def write_image(path: str, image: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext if ext else ".png", image)
    if not ok:
        raise ValueError(f"Impossible d'écrire l'image: {path}")
    buf.tofile(path)


def rect_iou(a: Rect, b: Rect) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2(), b.x2())
    y2 = min(a.y2(), b.y2())
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def rect_center(r: Rect) -> Tuple[float, float]:
    return (r.x + r.w / 2.0, r.y + r.h / 2.0)


def center_distance(a: Rect, b: Rect) -> float:
    ax, ay = rect_center(a)
    bx, by = rect_center(b)
    return math.hypot(ax - bx, ay - by)


def merge_rects(rects: List[Rect]) -> List[Rect]:
    changed = True
    out = rects[:]
    while changed:
        changed = False
        new_rects = []
        used = [False] * len(out)

        for i in range(len(out)):
            if used[i]:
                continue
            r = out[i]
            group = [r]
            used[i] = True

            for j in range(i + 1, len(out)):
                if used[j]:
                    continue
                if rect_iou(r, out[j]) > MERGE_IOU_THRESHOLD or center_distance(r, out[j]) < MERGE_CENTER_DIST:
                    group.append(out[j])
                    used[j] = True
                    changed = True

            x1 = min(g.x for g in group)
            y1 = min(g.y for g in group)
            x2 = max(g.x2() for g in group)
            y2 = max(g.y2() for g in group)
            new_rects.append(Rect(x1, y1, x2 - x1, y2 - y1))

        out = new_rects

    return out


def crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    return image[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w].copy()


def is_card_like_rect(r: Rect, W: int, H: int) -> bool:
    if r.w < MIN_CARD_W or r.h < MIN_CARD_H:
        return False
    if r.w > W * MAX_CARD_W_RATIO or r.h > H * MAX_CARD_H_RATIO:
        return False
    ratio = r.h / max(r.w, 1)
    if not (CARD_RATIO_MIN <= ratio <= CARD_RATIO_MAX):
        return False
    return True


def np_to_base64_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ============================================================
# DETECTION PLATEAU
# ============================================================

def detect_search_zones(image: np.ndarray) -> List[Rect]:
    """
    Détection large : on récupère des zones candidates qui contiennent
    potentiellement une carte.
    Ces zones NE SONT PAS le résultat final.
    """
    img = ensure_bgr(image)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Le fond est très clair => on segmente tout ce qui est "non blanc"
    mask = cv2.inRange(gray, 0, 235)

    # Nettoyage
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[Rect] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        r = Rect(x, y, w, h)

        # On reste permissif ici : zone de recherche
        if w < 40 or h < 60:
            continue
        if w > W * 0.60 or h > H * 0.80:
            continue

        rr = r.expand(20, 20, W, H)
        rects.append(rr)

    rects = merge_rects(rects)

    # Filtrage grossier
    final_rects: List[Rect] = []
    for r in rects:
        if r.area() < 7000:
            continue
        final_rects.append(r)

    # Tri visuel
    final_rects.sort(key=lambda rr: (rr.y, rr.x))
    return final_rects


# ============================================================
# RAFFINEMENT LOCAL : TROUVER LA VRAIE CARTE DANS CHAQUE ZONE
# ============================================================

def refine_card_inside_zone(image: np.ndarray, zone: Rect) -> Tuple[Optional[Rect], str]:
    img = ensure_bgr(image)
    H, W = img.shape[:2]
    zone = zone.clip(W, H)
    roi = crop(img, zone)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Approche 1 : non blanc
    mask1 = cv2.inRange(gray, 0, 235)

    # Approche 2 : contours / gradients
    edges = cv2.Canny(gray, 40, 120)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask2 = cv2.dilate(edges, k, iterations=1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k, iterations=2)

    # Fusion
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[float, Rect]] = []
    roi_h, roi_w = roi.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        r = Rect(x, y, w, h)

        if w < 40 or h < 70:
            continue

        ratio = h / max(w, 1)
        if ratio < 1.10 or ratio > 2.40:
            continue

        area = w * h
        if area < 3000:
            continue

        # Heuristique :
        # on favorise les rectangles verticaux, assez grands,
        # et moins "énormes" qu'une capsule
        area_score = min(area / 25000.0, 1.5)
        ratio_score = 1.0 - min(abs(ratio - 1.60) / 0.8, 1.0)

        # pénalité si trop centré sur un gros support gris
        # on regarde la densité de pixels sombres
        patch = gray[y:y+h, x:x+w]
        dark_ratio = float(np.mean(patch < 180))
        dark_score = 1.0 - abs(dark_ratio - 0.35)

        score = area_score * 1.6 + ratio_score * 1.8 + dark_score * 0.9
        candidates.append((score, r))

    if not candidates:
        return None, "aucun_contour_local_valide"

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_local = candidates[0]

    refined = Rect(
        zone.x + best_local.x,
        zone.y + best_local.y,
        best_local.w,
        best_local.h
    ).expand(PAD_X, PAD_Y, W, H)

    if refined.area() < MIN_REFINED_AREA:
        return None, "zone_trop_petite"

    if not is_card_like_rect(refined, W, H):
        return None, "ratio_taille_non_valide"

    return refined, f"ok(score={best_score:.3f})"


# ============================================================
# IDENTIFICATION CARTE
# ============================================================

def identify_card_fallback(card_image: np.ndarray, index: int) -> Dict[str, Any]:
    """
    Fallback simple si tu n'as pas encore rebranché ton vrai moteur.
    Ici on ne prétend PAS reconnaître la vraie carte.
    On fournit juste des infos utiles de debug.
    """
    img = ensure_bgr(card_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    return {
        "final_status": "debug_only",
        "final_card_id": f"CARD_{index:02d}",
        "final_score": 0.0,
        "reason": "identification_non_branchee",
        "signature": {
            "gray_mean": mean,
            "gray_std": std,
            "hsv_mean": {
                "h": h_mean,
                "s": s_mean,
                "v": v_mean
            }
        }
    }


def identify_card(card_image: np.ndarray, index: int) -> Dict[str, Any]:
    """
    Branche ici ton vrai moteur si tu l'as déjà.
    Exemple :
        return analyze_single_card(card_image)
    Sinon fallback debug.
    """
    try:
        # --------------------------------------------------------
        # TODO : branche ton vrai moteur ici
        # Exemples possibles selon ton projet :
        # return analyze_single_card(card_image)
        # return match_card_to_references(card_image)
        # return detect_card_signature(card_image)
        # --------------------------------------------------------
        return identify_card_fallback(card_image, index)
    except Exception as e:
        return {
            "final_status": "error",
            "final_card_id": None,
            "final_score": 0.0,
            "reason": f"exception_identification: {e}",
            "traceback": traceback.format_exc()
        }


# ============================================================
# PIPELINE COMPLET
# ============================================================

def analyze_board(image: np.ndarray) -> Dict[str, Any]:
    img = ensure_bgr(image)
    H, W = img.shape[:2]

    search_zones = detect_search_zones(img)

    detections: List[CardDetection] = []
    accepted_count = 0

    for idx, zone in enumerate(search_zones, start=1):
        refined, reason = refine_card_inside_zone(img, zone)

        if refined is None:
            detections.append(CardDetection(
                search_rect=zone,
                refined_rect=None,
                accepted=False,
                reason=reason,
                index=idx,
                analysis=None
            ))
            continue

        card_img = crop(img, refined)
        analysis = identify_card(card_img, idx)

        accepted = True
        detections.append(CardDetection(
            search_rect=zone,
            refined_rect=refined,
            accepted=accepted,
            reason=reason,
            index=idx,
            analysis=analysis
        ))
        accepted_count += 1

    result = {
        "image_size": {"w": W, "h": H},
        "search_zone_count": len(search_zones),
        "accepted_card_count": accepted_count,
        "cards": []
    }

    for det in detections:
        result["cards"].append({
            "index": det.index,
            "accepted": det.accepted,
            "reason": det.reason,
            "search_rect": asdict(det.search_rect),
            "refined_rect": asdict(det.refined_rect) if det.refined_rect else None,
            "analysis": det.analysis
        })

    return result


# ============================================================
# DEBUG VISUEL
# ============================================================

def draw_debug(image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    img = ensure_bgr(image).copy()

    for card in result.get("cards", []):
        sr = card["search_rect"]
        rr = card["refined_rect"]
        accepted = card["accepted"]
        index = card["index"]
        analysis = card.get("analysis") or {}

        # zone de recherche = jaune
        cv2.rectangle(
            img,
            (sr["x"], sr["y"]),
            (sr["x"] + sr["w"], sr["y"] + sr["h"]),
            (0, 255, 255),
            3
        )

        label1 = f"zone {index}"
        cv2.putText(
            img,
            label1,
            (sr["x"], max(20, sr["y"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 180),
            2,
            cv2.LINE_AA
        )

        # rectangle raffiné = vert / rouge
        if rr is not None:
            color = (0, 200, 0) if accepted else (0, 0, 255)
            cv2.rectangle(
                img,
                (rr["x"], rr["y"]),
                (rr["x"] + rr["w"], rr["y"] + rr["h"]),
                color,
                3
            )

            card_id = analysis.get("final_card_id", "UNKNOWN")
            score = analysis.get("final_score", 0.0)
            txt = f"{index}: {card_id} ({score:.2f})"
            cv2.putText(
                img,
                txt,
                (rr["x"], min(img.shape[0] - 10, rr["y"] + rr["h"] + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )
        else:
            txt = f"{index}: refine KO"
            cv2.putText(
                img,
                txt,
                (sr["x"], min(img.shape[0] - 10, sr["y"] + sr["h"] + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    return img


# ============================================================
# EXPORTS
# ============================================================

def build_html_report(image: np.ndarray, debug_image: np.ndarray, result: Dict[str, Any]) -> str:
    img64 = np_to_base64_png(image)
    dbg64 = np_to_base64_png(debug_image)
    json_pretty = json.dumps(result, ensure_ascii=False, indent=2)

    rows = []
    for card in result.get("cards", []):
        analysis = card.get("analysis") or {}
        card_id = analysis.get("final_card_id", "")
        status = analysis.get("final_status", "")
        score = analysis.get("final_score", 0.0)
        reason = card.get("reason", "")
        rows.append(f"""
        <tr>
            <td>{card.get("index", "")}</td>
            <td>{'oui' if card.get("accepted") else 'non'}</td>
            <td>{card_id}</td>
            <td>{status}</td>
            <td>{score:.3f}</td>
            <td>{reason}</td>
        </tr>
        """)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Export SpaceLab</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f7f7f7;
    color: #222;
}}
h1, h2 {{
    margin-top: 28px;
}}
img {{
    max-width: 100%;
    border: 1px solid #ccc;
    background: white;
}}
pre {{
    background: #111;
    color: #eee;
    padding: 16px;
    overflow-x: auto;
    border-radius: 8px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 8px 10px;
    text-align: left;
}}
th {{
    background: #eaeaea;
}}
.card {{
    background: white;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
</style>
</head>
<body>

<h1>Export SpaceLab</h1>

<div class="card">
    <h2>Résumé</h2>
    <p><b>Zones détectées :</b> {result.get("search_zone_count", 0)}</p>
    <p><b>Cartes retenues :</b> {result.get("accepted_card_count", 0)}</p>
</div>

<div class="card">
    <h2>Image source</h2>
    <img src="data:image/png;base64,{img64}" alt="image source">
</div>

<div class="card">
    <h2>Image debug</h2>
    <img src="data:image/png;base64,{dbg64}" alt="image debug">
</div>

<div class="card">
    <h2>Tableau</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Acceptée</th>
                <th>Carte</th>
                <th>Statut</th>
                <th>Score</th>
                <th>Raison</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</div>

<div class="card">
    <h2>JSON</h2>
    <pre>{json_pretty}</pre>
</div>

</body>
</html>
"""


def export_all(image: np.ndarray, result: Dict[str, Any], base_name: str) -> Dict[str, str]:
    debug_img = draw_debug(image, result)

    png_path = os.path.join(DEBUG_DIR, f"{base_name}_debug.png")
    json_path = os.path.join(DEBUG_DIR, f"{base_name}_result.json")
    html_path = os.path.join(DEBUG_DIR, f"{base_name}_report.html")

    write_image(png_path, debug_img)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    html = build_html_report(image, debug_img, result)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "png": png_path,
        "json": json_path,
        "html": html_path
    }


# ============================================================
# TKINTER UI
# ============================================================

class SpaceLabApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SpaceLab - Vérification avant production")

        self.image_path: Optional[str] = None
        self.image: Optional[np.ndarray] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_exports: Optional[Dict[str, str]] = None

        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        self.btn_open = tk.Button(frame, text="Ouvrir image", command=self.open_image, width=24)
        self.btn_open.grid(row=0, column=0, padx=5, pady=5)

        self.btn_analyze = tk.Button(frame, text="Analyser plateau", command=self.analyze, width=24)
        self.btn_analyze.grid(row=0, column=1, padx=5, pady=5)

        self.btn_export = tk.Button(frame, text="Exporter JSON + HTML + image", command=self.export_report, width=28)
        self.btn_export.grid(row=0, column=2, padx=5, pady=5)

        self.text = tk.Text(frame, height=28, width=140)
        self.text.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

    def log(self, msg: str):
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.root.update_idletasks()

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return

        try:
            self.image_path = path
            self.image = read_image(path)
            self.last_result = None
            self.last_exports = None
            self.text.delete("1.0", "end")
            self.log(f"Image chargée : {path}")
            self.log(f"Taille : {self.image.shape[1]} x {self.image.shape[0]}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def analyze(self):
        if self.image is None:
            messagebox.showwarning("Attention", "Charge une image d'abord.")
            return

        try:
            self.log("Analyse du plateau en cours...")
            result = analyze_board(self.image)
            self.last_result = result

            self.log("")
            self.log("=== RÉSULTAT ===")
            self.log(f"Zones détectées : {result.get('search_zone_count', 0)}")
            self.log(f"Cartes retenues : {result.get('accepted_card_count', 0)}")
            self.log("")

            for card in result.get("cards", []):
                analysis = card.get("analysis") or {}
                self.log(
                    f"[{card['index']}] "
                    f"accepted={card['accepted']} | "
                    f"reason={card['reason']} | "
                    f"card={analysis.get('final_card_id')} | "
                    f"status={analysis.get('final_status')} | "
                    f"score={analysis.get('final_score', 0.0):.3f}"
                )

            self.log("")
            self.log(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            messagebox.showerror("Erreur analyse", f"{e}\n\n{traceback.format_exc()}")

    def export_report(self):
        if self.image is None or self.last_result is None:
            messagebox.showwarning("Attention", "Fais une analyse avant l'export.")
            return

        try:
            base_name = "spacelab_export"
            if self.image_path:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

            exports = export_all(self.image, self.last_result, base_name)
            self.last_exports = exports

            self.log("")
            self.log("=== EXPORTS ===")
            self.log(f"Image debug : {exports['png']}")
            self.log(f"JSON       : {exports['json']}")
            self.log(f"HTML       : {exports['html']}")

            messagebox.showinfo(
                "Export terminé",
                "Les fichiers ont été exportés dans le dossier debug_exports."
            )
        except Exception as e:
            messagebox.showerror("Erreur export", f"{e}\n\n{traceback.format_exc()}")


# ============================================================
# CLI
# ============================================================

def run_cli(image_path: str):
    image = read_image(image_path)
    result = analyze_board(image)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    exports = export_all(image, result, base_name)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nExports :")
    print(" -", exports["png"])
    print(" -", exports["json"])
    print(" -", exports["html"])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os
    import sys

    railway = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PORT")

    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    elif railway:
        print("Environnement Railway détecté : interface Tkinter désactivée.")
        print("Ce fichier spacelab.py est une version locale desktop, pas une app web.")
        print("Déploie une version Gradio/Flask pour Railway.")
    else:
        if not TK_AVAILABLE:
            print("Tkinter indisponible. Lance avec un chemin image en argument.")
        else:
            root = tk.Tk()
            app = SpaceLabApp(root)
            root.mainloop()
