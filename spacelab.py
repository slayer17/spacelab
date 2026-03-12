import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

CARDS = []


# =====================================================
# LOAD CARDS
# =====================================================

def load_cards():

    global CARDS

    folder = "cards"

    for name in os.listdir(folder):

        path = os.path.join(folder, name)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (200, 300))

        CARDS.append({
            "name": name,
            "img": img
        })

    print("CARDS:", len(CARDS))


# =====================================================
# RECOGNIZE CARD
# =====================================================

def recognize_card(img):

    img = cv2.resize(img, (200, 300))

    best = None
    best_score = 999999999

    for c in CARDS:

        diff = cv2.absdiff(img, c["img"])

        score = diff.mean()

        if score < best_score:
            best_score = score
            best = c["name"]

    return best


# =====================================================
# DETECT STATIONS
# =====================================================

def detect_stations(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 8000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        ratio = h / float(w)

        if ratio < 1.2:
            continue

        if h < 120:
            continue

        objects.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })

    objects = sorted(
        objects,
        key=lambda o: o["h"],
        reverse=True
    )

    stations = objects[:3]

    stations = sorted(
        stations,
        key=lambda s: s["x"]
    )

    return stations


# =====================================================
# BUILD GRID (ROBUST VERSION)
# =====================================================

def build_grid(stations):

    left, mid, right = stations

    p_left = np.array([
        left["x"] + left["w"] / 2,
        left["y"] + left["h"] / 2
    ], dtype=np.float32)

    p_mid = np.array([
        mid["x"] + mid["w"] / 2,
        mid["y"] + mid["h"] / 2
    ], dtype=np.float32)

    p_right = np.array([
        right["x"] + right["w"] / 2,
        right["y"] + right["h"] / 2
    ], dtype=np.float32)

    vx = p_right - p_left

    dist = np.linalg.norm(vx)

    if dist < 1:
        return [], 0, 0

    ux = vx / dist

    uy = np.array([-ux[1], ux[0]], dtype=np.float32)

    step_x = dist / 2

    avg_h = (
        left["h"] +
        mid["h"] +
        right["h"]
    ) / 3

    avg_w = (
        left["w"] +
        mid["w"] +
        right["w"]
    ) / 3

    step_y = avg_h * 0.95

    card_w = int(avg_w * 0.6)

    card_h = int(avg_h * 0.8)

    offsets = [

        (0, -1), (1, -1), (2, -1),

        (-1, 0), (1, 0),

        (0, 0), (2, 0),

        (1, 0), (3, 0),

        (0, 1), (1, 1), (2, 1)

    ]

    positions = []

    for col, row in offsets:

        center = (
            p_left +
            ux * (col * step_x) +
            uy * (row * step_y)
        )

        x = int(center[0] - card_w / 2)
        y = int(center[1] - card_h / 2)

        positions.append((x, y))

    return positions, card_w, card_h


# =====================================================
# UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    img = cv2.imdecode(data, 1)

    stations = detect_stations(img)

    if len(stations) != 3:
        return jsonify({"rects": []})

    grid, cw, ch = build_grid(stations)

    rects = []

    for x, y in grid:

        x = max(0, x)
        y = max(0, y)

        crop = img[
            y:y+ch,
            x:x+cw
        ]

        if crop.size == 0:
            continue

        name = recognize_card(crop)

        rects.append({
            "x": x,
            "y": y,
            "w": cw,
            "h": ch,
            "name": name,
            "type": "CARTE"
        })

        return jsonify({"rects": rects})


from flask import send_from_directory


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

    load_cards()

    app.run(
        host="0.0.0.0",
        port=8080
    )
