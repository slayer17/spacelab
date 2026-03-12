import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# DETECT OBJECTS
# =====================================================

def detect_objects(img):

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

        if area < 6000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        ratio = h / float(w)

        objects.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "ratio": ratio
        })

    return objects


# =====================================================
# FIND STATIONS (max 3)
# =====================================================

def find_stations(objects):

    candidates = []

    for o in objects:

        w = o["w"]
        h = o["h"]
        area = o["area"]
        ratio = o["ratio"]

        # filtre taille minimum
        if area < 30000:
            continue

        # station = très verticale
        if ratio < 1.4:
            continue

        # station = large
        if w < 120:
            continue

        # station = grande hauteur
        if h < 200:
            continue

        candidates.append(o)

    # on garde les plus gros
    candidates = sorted(
        candidates,
        key=lambda o: o["area"],
        reverse=True
    )

    stations = candidates[:3]

    stations = sorted(
        stations,
        key=lambda s: s["x"]
    )

    print("STATIONS =", stations)

    return stations

# =====================================================
# BUILD LAYOUT FROM STATIONS
# =====================================================

def build_layout(stations):

    left, mid, right = stations

    cx = mid["x"] + mid["w"] / 2
    cy = mid["y"] + mid["h"] / 2

    dist = (right["x"] - left["x"]) / 2

    step_x = dist * 0.9
    step_y = mid["h"] * 1.1

    card_w = int(mid["w"] * 0.8)
    card_h = int(mid["h"] * 0.9)

    offsets = [

        (-1, -1),
        (0, -1),
        (1, -1),

        (-2, 0),
        (-1, 0),
        (1, 0),
        (2, 0),

        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    rects = []

    for ox, oy in offsets:

        x = int(cx + ox * step_x - card_w / 2)
        y = int(cy + oy * step_y - card_h / 2)

        rects.append((x, y, card_w, card_h))

    return rects


# =====================================================
# UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(data, 1)

    objects = detect_objects(img)

    stations = find_stations(objects)

    if len(stations) != 3:
        return jsonify({"rects": []})

    layout = build_layout(stations)

    rects = []

    for x, y, w, h in layout:

        rects.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "type": "GRID"
        })

    return jsonify({
        "rects": rects,
        "stations": len(stations)
    })


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

    app.run(
        host="0.0.0.0",
        port=8080
    )
