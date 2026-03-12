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

def load_card_images():

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

    print("CARDS LOADED:", len(cards))

    return cards

# Chargement de la base de données au démarrage
CARD_DB = load_card_images()

def compare_card(crop):

    if crop is None or crop.size == 0:
        return "Unknown", 999999

    best_score = float('inf')
    best_name = "None"

    test = cv2.resize(crop, (200, 300))

    # zone symbole (stable)
    test = test[20:120, 20:120]

    for card in CARD_DB:

        ref = cv2.resize(card["img"], (200, 300))

        ref = ref[20:120, 20:120]

        diff = cv2.absdiff(ref, test)

        score = np.sum(diff)

        if score < best_score:
            best_score = score
            best_name = card["name"]

    return best_name, best_score

@app.route("/")
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Fichier index.html non trouvé.", 404

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return json.dumps({"rects": []})

    uid = uuid.uuid4().hex
    path = os.path.join(UPLOAD_FOLDER, uid + ".jpg")
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return json.dumps({"rects": []})

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        objects.append({"x": x, "y": y, "w": w, "h": h, "area": area})

    # Tri par surface et récupération des 3 plus grandes (stations)
    objects = sorted(objects, key=lambda o: o["area"], reverse=True)
    stations = sorted(objects[:3], key=lambda s: s["x"])

    if len(stations) < 3:
        return json.dumps({"error": "Pas assez de stations détectées", "rects": []})

    rects = []
    for s in stations:
        rects.append({
            "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"],
            "type": "STATION"
        })

    # Calcul de la grille
    left, middle, right = stations[0], stations[1], stations[2]
    dx = middle["x"] - left["x"]
    dy = left["h"]
    card_w = int(left["w"] * 0.7)
    card_h = int(left["h"] * 0.9)

    positions = [
        (left["x"], left["y"] - dy), (middle["x"], middle["y"] - dy), (right["x"], right["y"] - dy),
        (left["x"] - dx, left["y"]), (left["x"] + dx, left["y"]),
        (middle["x"] - dx, middle["y"]), (middle["x"] + dx, middle["y"]),
        (right["x"] - dx, right["y"]), (right["x"] + dx, right["y"]),
        (left["x"], left["y"] + dy), (middle["x"], middle["y"] + dy), (right["x"], right["y"] + dy)
    ]

    for px, py in positions:
        x, y = int(px), int(py)
        
        # Vérification des limites de l'image pour le crop
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + card_w), min(height, y + card_h)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        crop = img[y1:y2, x1:x2]
        name, score = compare_card(crop)

        rects.append({
            "x": x, "y": y, "w": card_w, "h": card_h,
            "type": "CARTE", "name": name, "score": int(score)
        })

    return json.dumps({"rects": rects})

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
