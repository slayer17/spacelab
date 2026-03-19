def _find_bottom_line_box(panel_zone):
    """
    Cherche la ligne / double flèche en bas à droite.

    Nouvelle version :
    - plus stricte
    - évite de prendre les bords du panneau
    """
    if panel_zone is None or panel_zone.size == 0:
        return None

    ph, pw = panel_zone.shape[:2]
    x1 = int(pw * 0.52)
    x2 = pw
    y1 = int(ph * 0.68)
    y2 = ph

    zone = panel_zone[y1:y2, x1:x2]
    if zone is None or zone.size == 0:
        return None

    mask, _ = _make_bottom_light_mask(zone)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    zh, zw = zone.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0:
            continue

        area_ratio = area / float(max(zw * zh, 1))

        if area_ratio < 0.015:
            continue

        if bw < zw * 0.24:
            continue

        if bh > zh * 0.32:
            continue

        ratio = bw / float(max(bh, 1))
        if ratio < 2.4:
            continue

        # On évite les traits collés au bas ou au bord
        if (y + bh) >= (zh - 1):
            continue

        score = (ratio * 1.5) + (area_ratio * 6.0)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]

    return (x + x1, y + y1, bw, bh)
