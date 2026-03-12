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

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(c)

        ratio = h / float(w)

        obj_type = "UNKNOWN"

        # ratio carte ~ 9/7 = 1.28
        if 1.1 < ratio < 1.6:
            obj_type = "CARD"

        # station = très grande
        if area > 30000 and ratio > 1.3:
            obj_type = "STATION"

        objects.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "ratio": ratio,
            "type": obj_type
        })

    print("OBJECTS:", len(objects))

    return objects

# =====================================================
# FIND STATIONS (max 3)
# =====================================================

def find_stations(objects):

    stations = [
        o for o in objects
        if o["type"] == "STATION"
    ]

    stations = sorted(
        stations,
        key=lambda o: o["area"],
        reverse=True
    )

    stations = stations[:3]

    stations = sorted(
        stations,
        key=lambda o: o["x"]
    )

    print("STATIONS:", stations)

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
