def detect_main_card(img):

    original = img.copy()

    # ----------------------------
    # resize pour stabilité
    # ----------------------------
    max_dim = 1400

    h, w = img.shape[:2]
    scale = 1.0

    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape[:2]
    image_area = h * w

    # ----------------------------
    # preprocess
    # ----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 200)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    mask = cv2.bitwise_or(edges, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ----------------------------
    # contours
    # ----------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:

        area = cv2.contourArea(c)

        # ignore petits objets
        if area < image_area * 0.15:
            continue

        # approx
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(c)
            quad = cv2.boxPoints(rect).astype("float32")

        # warp test
        warp = warp_quad(img, quad)
        if warp is None:
            continue

        wh, ww = warp.shape[:2]

        if ww == 0 or wh == 0:
            continue

        ratio = wh / float(ww)

        # ratio carte portrait
        if ratio < 1.3 or ratio > 1.65:
            continue

        # bounding rect
        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))

        rect_area = bw * bh

        fill_ratio = area / float(rect_area)

        if fill_ratio < 0.75:
            continue

        candidates.append({
            "area": area,
            "rect_area": rect_area,
            "ratio": ratio,
            "fill": fill_ratio,
            "quad": quad,
            "bbox": [x, y, bw, bh]
        })

    if not candidates:
        return None

    # ----------------------------
    # meilleur candidat = plus grand
    # ----------------------------
    candidates.sort(
        key=lambda c: c["rect_area"],
        reverse=True
    )

    best = candidates[0]

    quad = best["quad"]

    if scale != 1.0:
        quad = quad / scale

    quad = order_points(quad)

    x, y, bw, bh = cv2.boundingRect(
        quad.astype(np.int32)
    )

    return {
        "x": int(x),
        "y": int(y),
        "w": int(bw),
        "h": int(bh),
        "type": "MAIN_CARD",
        "quad": quad.astype(int).tolist(),
        "debug": {
            "ratio": float(best["ratio"]),
            "fill": float(best["fill"]),
            "area": int(best["area"])
        }
    }
