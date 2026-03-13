import cv2
import numpy as np
import os
import glob
import json

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# =====================================================
# UTILS
# =====================================================

def compute_signature(img):

    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))

    b = float(np.mean(small[:, :, 0]))
    g = float(np.mean(small[:, :, 1]))
    r = float(np.mean(small[:, :, 2]))

    return {
        "mean": mean,
        "std": std,
        "color": [b, g, r]
    }


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

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warp = cv2.warpPerspective(
        img,
        M,
        (maxW, maxH)
    )

    return warp


# =====================================================
# DETECT MAIN CARD
# =====================================================

def detect_main_card(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    quad = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "quad": quad.tolist()
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    data = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    rect = detect_main_card(img)

    if rect is None:
        return jsonify({"rects": []})

    quad = np.array(rect["quad"], dtype="float32")

    warp = warp_quad(img, quad)

    sig = None

    if warp is not None:

        cv2.imwrite("warp.jpg", warp)

        sig = compute_signature(warp)

    return jsonify({
        "rects": [rect],
        "signature": sig
    })


@app.route("/warp")
def warp():
    return send_from_directory(".", "warp.jpg")


# =====================================================
# BUILD SIGNATURES
# =====================================================

@app.route("/build_signatures")
def build_signatures():

    files = glob.glob("cards/*.jpg")
    files += glob.glob("cards/*.jpeg")
    files += glob.glob("cards/*.png")

    cards = []

    for f in files:

        img = cv2.imread(f)

        if img is None:
            continue

        h, w = img.shape[:2]

        quad = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")

        warp = warp_quad(img, quad)

        if warp is None:
            continue

        sig = compute_signature(warp)

        name = os.path.basename(f)

        cards.append({
            "id": name,
            "signature": {
                "scan": {
                    "globalSignature": sig
                }
            }
        })

    with open("cards.js", "w", encoding="utf-8") as f:

        f.write("window.CARDS = ")
        json.dump(cards, f, indent=2)

    return "OK"


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
