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
    return """
    <html>
    <body style="font-family:Arial;padding:30px;">
    <h1>SpaceLab - Détecteur</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <br><br>
        <button type="submit">Envoyer l'image</button>
    </form>
    </body>
    </html>
    """

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == '':
        return "Aucun fichier sélectionné", 400

    uid = uuid.uuid4().hex
    name = uid + "_" + file.filename
    result = uid + "_result.png"

    save_path = os.path.join(UPLOAD_FOLDER, name)
    result_path = os.path.join(PROCESSED_FOLDER, result)
    file.save(save_path)

    image = cv2.imread(save_path)
    if image is None:
        return "Erreur lors de la lecture de l'image", 500
        
    draw = image.copy()
    
    # --- AMÉLIORATION DE LA DÉTECTION ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # On floute pour enlever les petits détails inutiles des cartes
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Seuil adaptatif : très efficace pour séparer les objets collés
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Érosion : on "grignote" les bords pour détacher les cartes qui se touchent
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(thresh, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for c in contours:
        area = cv2.contourArea(c)
        # On ignore ce qui est trop petit (poussière) ou trop grand (fond)
        if area < 3000 or area > 500000:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        ratio = h / float(w)
        
        detected_objects.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": area, "ratio": ratio
        })

    # On trie pour trouver les stations (souvent plus larges/carrées que les cartes)
    # Les stations ont souvent un ratio proche de 1.0 à 1.4
    rectangles = []
    for obj in detected_objects:
        # Logique simplifiée : une station est massive
        if 0.8 < obj["ratio"] < 1.5 and obj["area"] > 8000:
            obj_type = "STATION"
            color = (0, 0, 255) # ROUGE
        else:
            obj_type = "CARTE"
            color = (0, 255, 0) # VERT

        rectangles.append({
            "x": int(obj["x"]),
            "y": int(obj["y"]),
            "width": int(obj["w"]),
            "height": int(obj["h"]),
            "type": obj_type
        })

        # On dessine le rectangle
        cv2.rectangle(draw, (obj["x"], obj["y"]), 
                      (obj["x"] + obj["w"], obj["y"] + obj["h"]), color, 3)
        
        # On écrit le type au dessus
        cv2.putText(draw, obj_type, (obj["x"], obj["y"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(result_path, draw)
    rectangles_json = json.dumps(rectangles)
    img_url = url_for("uploaded_file", filename=name)
    res_url = url_for("processed_file", filename=result)

    return f"""
    <html>
    <body style="font-family:Arial;padding:30px;">
    <h2>Résultat de l'analyse</h2>
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div>
            <p>Originale</p>
            <img src="{img_url}" width="600">
        </div>
        <div>
            <p>Traitée (Détection)</p>
            <img src="{res_url}" width="600">
        </div>
    </div>
    <script>
        window.PY_RECTS = {rectangles_json};
        console.log("Objets détectés :", window.PY_RECTS);
    </script>
    <br><br>
    <a href="/" style="font-size: 20px;">Envoyer une autre image</a>
    </body>
    </html>
    """

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
