from flask import Flask, request, url_for, send_from_directory
import os
import uuid
import cv2
import numpy as np
import json



app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def home():
    # On vérifie si le fichier existe avant de l'ouvrir
    if not os.path.exists("index.html"):
        # Si le fichier manque, on affiche la liste des fichiers présents
        # Cela nous aidera à comprendre où Railway a mis tes fichiers
        files = os.listdir('.')
        return f"Erreur : index.html non trouvé. Fichiers présents : {files}", 500
    
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file: return "Pas d'image", 400

    uid = uuid.uuid4().hex
    name = f"{uid}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, name)
    file.save(save_path)

    image = cv2.imread(save_path)
    if image is None: return "Erreur lecture image", 500

    # 1. Prétraitement classique
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. SEPARATION DES CARTES (La partie magique)
    # On nettoie un peu le bruit
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # On calcule la distance au bord noir le plus proche
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # On ne garde que les "pics" de distance (le centre des cartes)
    # Cela sépare automatiquement les cartes qui se touchent
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 3. On trouve les contours sur ces centres séparés
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects_for_js = []
    for c in contours:
        # Comme on travaille sur le "centre" érodé, on récupère le rectangle
        x, y, w, h = cv2.boundingRect(c)
        
        # On agrandit le rectangle pour compenser l'érosion et bien englober la carte
        # On rajoute environ 20% de marge
        margin_w = int(w * 0.35)
        margin_h = int(h * 0.35)
        
        new_x = max(0, x - margin_w)
        new_y = max(0, y - margin_h)
        new_w = w + (margin_w * 2)
        new_h = h + (margin_h * 2)

        if new_w > 30 and new_h > 30: # Filtre anti-bruit
            rects_for_js.append({
                "x": int(new_x),
                "y": int(new_y),
                "w": int(new_w),
                "h": int(new_h)
            })

    return json.dumps({
        "status": "success",
        "rects": rects_for_js
    })

# Pour que Railway trouve tes fichiers JS (app.js, cards.js...)
@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

@app.route('/processed/<filename>')
def send_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
