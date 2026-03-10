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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

objects = []

for c in contours:

    area = cv2.contourArea(c)

    if area < 1500:
        continue

    x, y, w, h = cv2.boundingRect(c)

    ratio = h / float(w)

    objects.append({
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "area": area,
        "ratio": ratio
    })
# ==========================
# DETECTION DES STATIONS
# ==========================

stations = sorted(
    [o for o in objects if o["ratio"] < 1.9],
    key=lambda o: o["area"],
    reverse=True
)[:3]

rectangles = []

for obj in objects:

    obj_type = "CARTE"

    for s in stations:
        if obj["x"] == s["x"] and obj["y"] == s["y"]:
            obj_type = "STATION"

    rectangles.append({
        "x": int(obj["x"]),
        "y": int(obj["y"]),
        "width": int(obj["width"]),
        "height": int(obj["height"]),
        "type": obj_type
    })

    color = (0,255,0)

    if obj_type == "STATION":
        color = (0,0,255)

    cv2.rectangle(
        draw,
        (obj["x"], obj["y"]),
        (obj["x"]+obj["width"], obj["y"]+obj["height"]),
        color,
        2
    )
        cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imwrite(result_path, draw)

    rectangles_json = json.dumps(rectangles)

    img_url = url_for("uploaded_file", filename=name)
    res_url = url_for("processed_file", filename=result)

    return f"""
    <html>
    <body style="font-family:Arial;padding:30px;">

    <h2>Image originale</h2>
    <img src="{img_url}" width="900">

    <h2>Image traitée</h2>
    <img src="{res_url}" width="900">

    <script>
    window.PY_RECTS = {rectangles_json};
    console.log("Rectangles Python:", window.PY_RECTS);
    </script>

    <br><br>
    <a href="/">Retour</a>

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
    app.run(host="0.0.0.0", port=port)
