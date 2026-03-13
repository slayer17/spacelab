import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# ORDER 4 POINTS
# =====================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# =====================================================
# WARP PERSPECTIVE
# =====================================================

def warp_quad(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    if max_w < 1 or max_h < 1:
        return None

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))

    return warped


# =====================================================
# DETECT CARD-LIKE QUADS
# =====================================================

def detect_cards(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img

    rects = []

    for c in contours:
        area = cv2.contourArea(c)

        # filtre bruit
        if area < 3000:
            continue

        # filtre trop grand
        if area > img_area * 0.8:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # on veut un quadrilatère
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")
        warped = warp_quad(img, pts)

        if warped is None:
            continue

        h, w = warped.shape[:2]

        if w < 40 or h < 40:
            continue

        ratio = h / float(w)

        # carte ~ 9/7 = 1.28, on laisse large au début
        if ratio < 1.15 or ratio > 1.65:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)

        rects.append({
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "type": "CARD",
            "ratio": float(ratio),
            "area": float(area)
        })

    # supprime les doublons / recouvrements forts
    rects = non_max_suppression(rects)

    print("CARDS DETECTED:", len(rects))

    return rects


# =====================================================
# SIMPLE NON-MAX SUPPRESSION
# =====================================================

def iou(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    return inter / float(area_a + area_b - inter)


def non_max_suppression(rects):
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: r["area"], reverse=True)
    kept = []

    for r in rects:
        keep = True

        for k in kept:
            if iou(r, k) > 0.4:
                keep = False
                break

        if keep:
            kept.append(r)

    kept = sorted(kept, key=lambda r: (r["y"], r["x"]))
    return kept


# =====================================================
# UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, 1)

    if img is None:
        return jsonify({"rects": []})

    rects = detect_cards(img)

    # on renvoie seulement les rectangles pour STEP 1
    return jsonify({"rects": rects})


# =====================================================
# STATIC
# =====================================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
