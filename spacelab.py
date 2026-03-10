from flask import Flask, request, url_for, send_from_directory
import os
import uuid
import cv2
import numpy as np
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# Création des dossiers si absent
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return """
    <html>
    <body style="font-family:Arial;padding:30px;background-color:#f4f4f9;">
    <h1>SpaceLab - Analyse de Plateau</h1>
    <p>Envoyez une photo pour détecter les cartes et les stations.</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <br><br>
        <button type="submit" style="padding:10px 20px;cursor:pointer;">Analyser l'image</button>
    </form>
    </body>
    </html>
    """

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == '':
        return "Erreur : Aucun fichier sélectionné", 400

    uid = uuid.uuid4().hex
    name = f"{uid}_{file.filename}"
    result = f"{uid}_result.png"

    save_path = os.path.join(UPLOAD_FOLDER, name)
    result_path = os.path.join(PROCESSED_FOLDER, result)
    file.save(save_path)

    image = cv2.imread(save_path)
    if image is None:
        return "Erreur : Impossible de lire l'image", 500
        
    draw = image.copy()
    
    # --- PRÉ-TRAITEMENT POUR SÉPARER LES OBJETS COLLÉS ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flou pour lisser les textures internes des cartes
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Seuil adaptatif pour mieux gérer les ombres du plateau
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Érosion : on réduit les formes pour détacher ce qui se touche
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(thresh, kernel, iterations=1)
    # Dilatation : on redonne un peu de volume après avoir séparé
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for c in contours:
        area = cv2.contourArea(c)
        # On ignore les petits bruits (aire < 3000)
        if area < 3000:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        ratio = h / float(w)
        
        # Filtrage par ratio (évite de détecter des lignes ou des bords de table)
        if 0.4 < ratio < 2.8:
            detected_objects.append({
                "x": x, "y": y, "w": w, "h": h,
                "area": area, "ratio": ratio
            })

    rectangles = []
    for obj in detected_objects:
        # Une STATION est plus "carrée" et souvent plus grande qu'une carte
        if 0.8 < obj["ratio"] < 1.4 and obj["area"] > 7000:
            obj_type = "STATION"
            color = (0, 0, 255) # Rouge
        else:
            obj_type = "CARTE"
            color = (0, 255, 0) # Vert

        rectangles.append({
            "x": int(obj["x"]),
            "y": int(obj["y"]),
            "width": int(obj["w"]),
            "height": int(obj["h"]),
            "type": obj_type
        })

        # Dessin des résultats
        cv2.rectangle(draw, (obj["x"], obj["y"]), 
                      (obj["x"] + obj["w"], obj["y"] + obj["h"]), color, 4)
        cv2.putText(draw, obj_type, (obj["x"], obj["y"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imwrite(result_path, draw)
    rectangles_json = json.dumps(rectangles)
    
    img_url = url_for("uploaded_file", filename=name)
    res_url = url_for("processed_file", filename=result)

    return f"""
    <html>
    <body style="font-family:Arial;padding:20px;">
    <h2>Analyse terminée</h2>
    <div style="display: flex; gap: 10px;">
        <img src="{img_url}" width="45%">
        <img src="{res_url}" width="45%">
    </div>
    <script>
        window.PY_RECTS = {rectangles_json};
        console.log("Objets détectés :", window.PY_RECTS);
    </script>
    <br><br>
    <a href="/">Retour au menu</a>
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
