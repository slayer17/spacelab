from flask import Flask, request, send_from_directory
import os
import uuid
import cv2
import numpy as np
import json
import glob

app = Flask(__name__)

CARDS_FOLDER = "cards"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======================
# LOAD CARDS
# ======================

def load_cards():

    cards = []

    files = []
    files += glob.glob(CARDS_FOLDER + "/*.jpg")
    files += glob.glob(CARDS_FOLDER + "/*.jpeg")
    files += glob.glob(CARDS_FOLDER + "/*.png")

    for f in files:

        img = cv2.imread(f)

        if img is None:
            continue

        name = os.path.basename(f)

        cards.append({
            "name": name,
            "img": img
        })

    print("CARDS:", len(cards))

    return cards


CARD_DB = load_cards()


# ======================
# HASH
# ======================

def zone_hash(gray):

    small = cv2.resize(gray, (16, 16))
    avg = np.mean(small)

    return (small > avg).astype(np.uint8)


def hash_score(a, b):

    return np.sum(a != b)


# ======================
# RECOGNIZE
# ======================

def recognize_card(crop):

    if crop is None or crop.size == 0:
        return "None", 999999

    best_score = 999999999
    best_name = "None"

    try:
        test = cv2.resize(crop, (200, 300))
    except:
        return "None", 999999

    test_top = test[40:120, 20:110]
    test_bottom = test[220:295, 20:170]

    test_top_g = cv2.cvtColor(test_top, cv2.COLOR_BGR2GRAY)
    test_bottom_g = cv2.cvtColor(test_bottom, cv2.COLOR_BGR2GRAY)

    test_top_hash = zone_hash(test_top_g)
    test_bottom_hash = zone_hash(test_bottom_g)

    for card in CARD_DB:

        try:
            ref = cv2.resize(card["img"], (200, 300))
        except:
            continue

        ref_top = ref[40:120, 20:110]
        ref_bottom = ref[220:295, 20:170]

        ref_top_g = cv2.cvtColor(ref_top, cv2.COLOR_BGR2GRAY)
        ref_bottom_g = cv2.cvtColor(ref_bottom, cv2.COLOR_BGR2GRAY)

        ref_top_hash = zone_hash(ref_top_g)
        ref_bottom_hash = zone_hash(ref_bottom_g)

        diff1 = cv2.absdiff(ref_top_g, test_top_g)
        diff2 = cv2.absdiff(ref_bottom_g, test_bottom_g)

        pixel = np.mean(diff1) + np.mean(diff2)

        hashv = (
            hash_score(ref_top_hash, test_top_hash)
            + hash_score(ref_bottom_hash, test_bottom_hash)
        )

        score = pixel * 5 + hashv * 20

        if score < best_score:

            best_score = score
            best_name = card["name"]

    return best_name, best_score


# ======================
# DETECT STATIONS
# ======================

def detect_stations(img):


# ======================
# GRID
# ======================

def build_grid(stations):

    left, mid, right = stations

    dx = int((mid["x"] - left["x"]) * 0.9)
    dy = int(left["h"] * 0.95)

    card_w = int(left["w"] * 0.55)
    card_h = int(left["h"] * 0.8)

    positions = [

        (left["x"], left["y"] - dy),
        (mid["x"], mid["y"] - dy),
        (right["x"], right["y"] - dy),

        (left["x"] - dx, left["y"]),
        (left["x"] + dx, left["y"]),

        (mid["x"] - dx, mid["y"]),
        (mid["x"] + dx, mid["y"]),

        (right["x"] - dx, right["y"]),
        (right["x"] + dx, right["y"]),

        (left["x"], left["y"] + dy),
        (mid["x"], mid["y"] + dy),
        (right["x"], right["y"] + dy),
    ]

    return positions, card_w, card_h


# ======================
# ROUTES
# ======================

@app.route("/")
def home():

    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("image")

    if not file:
        return json.dumps({"rects": []})

    uid = uuid.uuid4().hex

    path = os.path.join(
        UPLOAD_FOLDER,
        uid + ".jpg"
    )

    file.save(path)

    img = cv2.imread(path)

    if img is None:
        return json.dumps({"rects": []})

    rects = []

    stations = detect_stations(img)

    if len(stations) < 3:
        return json.dumps({"rects": []})

    for s in stations:

        rects.append({
            "x": s["x"],
            "y": s["y"],
            "w": s["w"],
            "h": s["h"],
            "type": "STATION"
        })

    positions, cw, ch = build_grid(stations)

    h, w = img.shape[:2]

    for px, py in positions:

        x = int(px)
        y = int(py)

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(w, x + cw)
        y2 = min(h, y + ch)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]

        name, score = recognize_card(crop)

        rects.append({
            "x": x,
            "y": y,
            "w": cw,
            "h": ch,
            "type": "CARTE",
            "name": name,
            "score": float(score)
        })

    return json.dumps({"rects": rects})


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
