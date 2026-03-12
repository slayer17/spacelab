import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# DETECT OBJECTS (cards + stations)
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

        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        ratio = h / float(w)

        obj_type = "UNKNOWN"

        # stations = très grandes + verticales
        if area > 40000 and ratio > 1.3:
            obj_type = "STATION"

        # cartes = moyennes + verticales
        elif area > 10000 and ratio > 1.2:
            obj_type = "CARD"

        objects.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "type": obj_type,
            "area": area
        })

    return objects


# =====================================================
# UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(data, 1)

    objects = detect_objects(img)

    cards = 0
    stations = 0

    rects = []

    for o in objects:

        if o["type"] == "CARD":
            cards += 1

        if o["type"] == "STATION":
            stations += 1

        rects.append({
            "x": o["x"],
            "y": o["y"],
            "w": o["w"],
            "h": o["h"],
            "type": o["type"]
        })

    print("CARDS:", cards)
    print("STATIONS:", stations)

    return jsonify({
        "rects": rects,
        "cards": cards,
        "stations": stations
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
