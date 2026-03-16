import cv2
import numpy as np
import os
import json

CARDS_JS = 'cards.js'
CARDS_FOLDER = 'cards'


def crop_percent(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    xa = max(0, min(w, int(w * x1)))
    xb = max(0, min(w, int(w * x2)))
    ya = max(0, min(h, int(h * y1)))
    yb = max(0, min(h, int(h * y2)))

    if xb <= xa or yb <= ya:
        return None

    return img[ya:yb, xa:xb]


def compute_basic_signature(img):
    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    return {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'color': [
            float(np.mean(small[:, :, 0])),
            float(np.mean(small[:, :, 1])),
            float(np.mean(small[:, :, 2]))
        ]
    }


def compute_signature_safe(img):
    if img is None or img.size == 0:
        return None
    return compute_basic_signature(img)


def normalize_card(img):
    return cv2.resize(img, (240, 360))


def compute_signature(img):

    card = normalize_card(img)

    h, w = card.shape[:2]

    # GLOBAL
    global_sig = compute_basic_signature(card)

    # BOTTOM
    bottom = crop_percent(card, 0.00, 0.82, 0.55, 1.00)
    bottom_sig = compute_signature_safe(bottom)

    # COLOR
    color_band = crop_percent(card, 0.00, 0.00, 0.25, 0.20)
    color_sig = compute_signature_safe(color_band)

    # SYMBOL
    symbol_zone = crop_percent(card, 0.05, 0.20, 0.20, 0.31)

    symbol_sig = compute_signature_safe(symbol_zone)

    return {
        'global': global_sig,
        'bottom': bottom_sig,
        'color': color_sig,
        'symbol': symbol_sig
    }

def load_cards():
    with open(CARDS_JS, 'r', encoding='utf-8') as f:
        txt = f.read().replace('window.CARDS =', '', 1).strip()

    if txt.endswith(';'):
        txt = txt[:-1]

    return json.loads(txt)


def save_cards(cards):
    with open(CARDS_JS, 'w', encoding='utf-8') as f:
        f.write('window.CARDS = ')
        json.dump(cards, f, indent=2, ensure_ascii=False)


def main():
    cards = load_cards()
    count = 0

    for c in cards:
        path = os.path.join(CARDS_FOLDER, c['id'].lower() + '.jpeg')

        if not os.path.exists(path):
            alt_jpg = os.path.join(CARDS_FOLDER, c['id'].lower() + '.jpg')
            alt_png = os.path.join(CARDS_FOLDER, c['id'].lower() + '.png')
            if os.path.exists(alt_jpg):
                path = alt_jpg
            elif os.path.exists(alt_png):
                path = alt_png
            else:
                print('Image introuvable pour', c['id'])
                continue

        print('Processing', path)

        img = cv2.imread(path)
        if img is None:
            print('Impossible de lire', path)
            continue

        c['signature'] = {
            'scan': compute_signature(img)
        }
        count += 1

    save_cards(cards)
    print('DONE', count, 'cards updated')


if __name__ == '__main__':
    main()
    
