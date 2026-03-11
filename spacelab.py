from flask import Flask, request, send_from_directory
import os
import uuid
import cv2
import numpy as np
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    with open("index.html","r",encoding="utf-8") as f:
        return f.read()

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("image")
    if not file:
        return json.dumps({"rects":[]})

    uid = uuid.uuid4().hex
    path = os.path.join(UPLOAD_FOLDER, uid+".jpg")
    file.save(path)

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(7,7),0)

    edges = cv2.Canny(blur,40,120)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(edges,kernel,iterations=2)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    objects=[]

    for c in contours:

        area=cv2.contourArea(c)

        if area < 3000:
            continue

        x,y,w,h=cv2.boundingRect(c)

        objects.append({
            "x":int(x),
            "y":int(y),
            "w":int(w),
            "h":int(h),
            "area":area
        })

    # tri par taille
    objects=sorted(objects,key=lambda o:o["area"],reverse=True)

    # les 3 plus gros = stations
    stations=objects[:3]

    rects=[]

    for o in objects:

        t="CARTE"

        for s in stations:
            if o["x"]==s["x"] and o["y"]==s["y"]:
                t="STATION"

        rects.append({
            "x":o["x"],
            "y":o["y"],
            "w":o["w"],
            "h":o["h"],
            "type":t
        })

    return json.dumps({"rects":rects})


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
