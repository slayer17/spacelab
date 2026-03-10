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
    result = f"{uid}_result.png"
    save_path = os.path.join(UPLOAD_FOLDER, name)
    result_path = os.path.join(PROCESSED_FOLDER, result)
    file.save(save_path)

    image = cv2.imread(save_path)
    # --- DETECTION AMELIOREE ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # On sépare les cartes collées
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(thresh, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 rects_for_js = []
    for c in contours:
        # On simplifie le contour pour obtenir un rectangle plus propre
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        area = cv2.contourArea(c)
        if area < 5000: continue # Augmenté pour éviter les petits bruits
        
        x, y, w, h = cv2.boundingRect(approx)
        
        # Filtre de proportion : une carte est plus haute que large
        # Une station est plus carrée ou large
        rects_for_js.append({"x": x, "y": y, "w": w, "h": h})

    # On renvoie les données à ton JavaScript
    return json.dumps({
        "status": "success",
        "rects": rects_for_js,
        "image_url": f"/processed/{result}"
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
