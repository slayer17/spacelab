import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


# ===============================
# ORDER POINTS
# ===============================

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ===============================
# WARP
# ===============================

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

    warp = cv2.warpPerspective(img, M, (maxW, maxH))

    return warp


# ===============================
# DETECT
# ===============================

def detect_cards(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rects = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 3000:
            continue

        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.02 * peri,
            True
        )

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")

        warp = warp_quad(img, pts)

        if warp is None:
            continue

        h, w = warp.shape[:2]

        if w == 0 or h == 0:
            continue

        ratio = h / float(w)

        if ratio < 1.1 or ratio > 1.7:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)

        rects.append({
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "type": "CARD"
        })

    print("RECTS:", rects)

    return rects


# ===============================
# UPLOAD
# ===============================

@app.route("/upload", methods=["POST"])
def upload():

    try:

        file = request.files["image"]

        data = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(data, 1)
        print("IMAGE SHAPE:", img.shape)
        if img is None:
            return jsonify({"rects": []})

        rects = detect_cards(img)

        return jsonify({"rects": rects})

    except Exception as e:

        print("ERROR:", e)

        return jsonify({"rects": []})


# ===============================
# STATIC
# ===============================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080
    )
