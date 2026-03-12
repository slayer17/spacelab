from flask import Flask, request, send_from_directory
import os
import uuid
import cv2
import numpy as np
import json
import glob
CARDS_FOLDER = "cards"

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

        if area < 5000:
            continue

        x,y,w,h=cv2.boundingRect(c)

        objects.append({
            "x":x,
            "y":y,
            "w":w,
            "h":h,
            "area":area
        })

    # tri par surface
    objects=sorted(objects,key=lambda o:o["area"],reverse=True)

    # stations = 3 plus grandes
    stations=objects[:3]

    stations=sorted(stations,key=lambda s:s["x"])

    rects=[]

    for s in stations:

        rects.append({
            "x":s["x"],
            "y":s["y"],
            "w":s["w"],
            "h":s["h"],
            "type":"STATION"
        })

    # calcul grille
    left=stations[0]
    middle=stations[1]
    right=stations[2]

    dx=middle["x"]-left["x"]
    dy=left["h"]

    card_w=int(left["w"]*0.7)
    card_h=int(left["h"]*0.9)

    positions=[

        (left["x"],left["y"]-dy),
        (middle["x"],middle["y"]-dy),
        (right["x"],right["y"]-dy),

        (left["x"]-dx,left["y"]),
        (left["x"]+dx,left["y"]),

        (middle["x"]-dx,middle["y"]),
        (middle["x"]+dx,middle["y"]),

        (right["x"]-dx,right["y"]),
        (right["x"]+dx,right["y"]),

        (left["x"],left["y"]+dy),
        (middle["x"],middle["y"]+dy),
        (right["x"],right["y"]+dy)
    ]

    for x,y in positions:

        rects.append({
            "x":int(x),
            "y":int(y),
            "w":card_w,
            "h":card_h,
            "type":"CARTE"
        })

    return json.dumps({"rects":rects})


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
