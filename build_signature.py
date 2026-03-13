import cv2
import numpy as np
import os
import json
import glob


CARDS_FOLDER = "cards"


# =========================
# SAME FUNCTIONS AS SCANNER
# =========================

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_quad(img, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    if maxW < 10 or maxH < 10:
        return None

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warp = cv2.warpPerspective(
        img,
        M,
        (maxW, maxH)
    )

    return warp


def compute_signature(img):

    small = cv2.resize(img, (32, 32))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))

    b = float(np.mean(small[:, :, 0]))
    g = float(np.mean(small[:, :, 1]))
    r = float(np.mean(small[:, :, 2]))

    return {
        "mean": mean,
        "std": std,
        "color": [b, g, r]
    }


# =========================
# MAIN
# =========================

cards = []

files = glob.glob("cards/*.jpg")
files += glob.glob("cards/*.jpeg")
files += glob.glob("cards/*.png")


for f in files:

    print("Processing", f)

    img = cv2.imread(f)

    if img is None:
        continue

    h, w = img.shape[:2]

    quad = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    warp = warp_quad(img, quad)

    if warp is None:
        continue

    sig = compute_signature(warp)

    name = os.path.basename(f)

    cards.append({
        "id": name,
        "signature": {
            "scan": {
                "globalSignature": sig
            }
        }
    })


with open("cards.js", "w", encoding="utf-8") as f:

    f.write("window.CARDS = ")
    json.dump(cards, f, indent=2)

print("DONE")