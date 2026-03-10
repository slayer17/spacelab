from flask import Flask, request, url_for, send_from_directory
import os
import uuid
import cv2
import numpy as np

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
    <h1>SpaceLab</h1>

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

    file = request.files["image"]

    uid = uuid.uuid4().hex
    name = uid + "_" + file.filename
    result = uid + "_result.png"

    save_path = os.path.join(UPLOAD_FOLDER, name)
    result_path = os.path.join(PROCESSED_FOLDER, result)

    file.save(save_path)

image = cv2.imread(save_path)
draw = image.copy()

height, width = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((9,9), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Diagnostic : dessiner tous les blobs
for c in contours:

    x, y, w, h = cv2.boundingRect(c)

    if w < 40 or h < 40:
        continue

    cv2.rectangle(draw, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite(result_path,draw)

    img_url = url_for("uploaded_file",filename=name)
    res_url = url_for("processed_file",filename=result)

    return f"""
    <html>
    <body style="font-family:Arial;padding:30px;">

    <h2>Image originale</h2>
    <img src="{img_url}" width="900">

    <h2>Image traitée</h2>
    <img src="{res_url}" width="900">

    <br><br>
    <a href="/">Retour</a>

    </body>
    </html>
    """


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER,filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
