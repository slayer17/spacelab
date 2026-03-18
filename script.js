console.log("CARDS =", CARDS);

let mode = "BOARD";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");

const loadBtn = document.getElementById("loadBtn");
const fileInput = document.getElementById("file");

const boardBtn = document.getElementById("boardBtn");
const cardsBtn = document.getElementById("cardsBtn");

const resultEl = document.getElementById("result");

let currentStream = null;


// =========================
// MODE
// =========================

boardBtn.onclick = () => {
    mode = "BOARD";
    resultEl.textContent = "Mode BOARD";
};

cardsBtn.onclick = () => {
    mode = "CARDS_ONLY";
    resultEl.textContent = "Mode CARDS_ONLY";
};


// =========================
// CAMERA
// =========================

startBtn.onclick = async () => {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        });

        video.srcObject = currentStream;

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Erreur caméra";
    }
};


// =========================
// CAPTURE
// =========================

captureBtn.onclick = () => {
    if (!currentStream) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    sendToPython();
};


// =========================
// LOAD IMAGE
// =========================

loadBtn.onclick = () => {
    fileInput.click();
};

fileInput.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        sendToPython();
    };

    img.src = URL.createObjectURL(file);
};


// =========================
// COLOR
// =========================

// function detectColorFromBGR(b, g, r) {
    // const max = Math.max(r, g, b);
    // const min = Math.min(r, g, b);
    // const delta = max - min;

    // if (delta < 15) {
        // return "ROUGE";
    // }

    // let hue = 0;

    // if (max === r) {
        // hue = 60 * (((g - b) / delta) % 6);
    // } else if (max === g) {
        // hue = 60 * (((b - r) / delta) + 2);
    // } else {
        // hue = 60 * (((r - g) / delta) + 4);
    // }

    // if (hue < 0) hue += 360;

    // if (hue >= 340 || hue <= 20) return "ROUGE";
    // if (hue >= 25 && hue <= 75) return "JAUNE";
    // if (hue >= 80 && hue <= 170) return "VERT";
    // if (hue >= 190 && hue <= 260) return "BLEU";

    // return "ROUGE";
// }


// =========================
// SEND TO PYTHON
// =========================

function sendToPython() {
    canvas.toBlob(async blob => {
        const form = new FormData();

        form.append("image", blob, "capture.jpg");
        form.append("mode", mode);

        resultEl.textContent = "Envoi…";

        try {
            const res = await fetch("/upload", {
                method: "POST",
                body: form
            });

            const json = await res.json();

            console.log("SERVER SIG", json.signature);

            drawRects(json.rects || []);

            if (json.rois && json.rects && json.rects.length > 0) {
                const r = json.rects[0];
                drawRois(json.rois, r);
            }

                  // =========================
            // MATCH
            // =========================

            if (json.signature) {
                let detectedColor = null;

                if (json.signature?.color?.detected) {
                    detectedColor = json.signature.color.detected;

                    console.log(
                        "COLOR PY =",
                        detectedColor,
                        json.signature.color.debug,
                        json.signature.color.color
                    );
                }

                let cardsFiltered = CARDS;

                if (detectedColor) {
                    cardsFiltered = CARDS.filter(
                        c => c.couleur === detectedColor
                    );
                }

                const resultMatch = matchSignature(
                    json.signature,
                    cardsFiltered
                );

                console.log("MATCH =", resultMatch);

        if (resultMatch && resultMatch.card) {

    const detectedSymbol =
        json.signature?.symbol?.name || "??";

    const rawDetectedSymbol =
        json.signature?.symbol?.raw_name || "??";

    const detectedSymbolScore =
        json.signature?.symbol?.score ?? 0;

const detectedPoints =
    json.signature?.points?.digit ?? "??";

const rawDetectedPoints =
    json.signature?.points?.raw_digit ?? "??";

const detectedPointsScore =
    json.signature?.points?.score ?? 0;

    resultEl.textContent =
        "Carte : " + resultMatch.card.id + "\n" +
        "Couleur : " + resultMatch.card.couleur + "\n" +
        "Symbole carte : " + resultMatch.card.symbol + "\n" +
        "Symbole détecté : " + detectedSymbol + "\n" +
        "Symbole brut : " + rawDetectedSymbol + "\n" +
        "Score symbole : " + Number(detectedSymbolScore).toFixed(3) + "\n" +
        "Points carte : " + resultMatch.card.points + "\n" +
        "Points détectés : " + detectedPoints + "\n" +
        "Points bruts : " + rawDetectedPoints + "\n" +
        "Score points : " + Number(detectedPointsScore).toFixed(3) + "\n" +
        "Score : " + resultMatch.score.toFixed(3);
} else {
                    resultEl.textContent = "Pas trouvé";
                }

            } else {
                resultEl.textContent = "Pas de signature";
            }

        } catch (err) {
            console.error(err);
            resultEl.textContent = "Erreur serveur";
        }
    }, "image/jpeg");
}



// =========================
// DRAW RECT
// =========================

function drawRects(rects) {
    ctx.lineWidth = 3;
    ctx.font = "20px Arial";

    let text = "";

    rects.forEach(r => {
        if (r.type === "STATION") {
            ctx.strokeStyle = "red";
        } else {
            ctx.strokeStyle = "yellow";
        }

        ctx.strokeRect(r.x, r.y, r.w, r.h);

        if (r.name) {
            ctx.fillStyle = "lime";
            ctx.fillText(r.name, r.x, r.y - 5);
            text += r.name + "\n";
        }
    });

    if (text) resultEl.textContent = text;
}


// =========================
// DRAW ROI
// =========================

function drawRois(rois, rect) {
    const scaleX = rect.w / 200;
    const scaleY = rect.h / 300;

    rois.forEach(r => {
        if (r.type === "GLOBAL") return;

        if (r.type === "COLOR")
            ctx.strokeStyle = "red";
        else if (r.type === "SYMBOL")
            ctx.strokeStyle = "blue";
        else if (r.type === "BOTTOM")
            ctx.strokeStyle = "violet";
        else
            ctx.strokeStyle = "white";

        ctx.lineWidth = 2;

        ctx.strokeRect(
            rect.x + r.x * scaleX,
            rect.y + r.y * scaleY,
            r.w * scaleX,
            r.h * scaleY
        );
    });
}
