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

const result = document.getElementById("result");

let currentStream = null;



// =========================
// MODE
// =========================

boardBtn.onclick = () => {

    mode = "BOARD";
    result.textContent = "Mode BOARD";

};

cardsBtn.onclick = () => {

    mode = "CARDS_ONLY";
    result.textContent = "Mode CARDS_ONLY";

};



// =========================
// CAMERA
// =========================

startBtn.onclick = async () => {

    try {

        currentStream =
            await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" }
            });

        video.srcObject = currentStream;

    } catch (err) {

        console.error(err);
        result.textContent = "Erreur caméra";

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
// SEND TO PYTHON
// =========================

function sendToPython() {

    canvas.toBlob(async blob => {

        const form = new FormData();

        form.append("image", blob, "capture.jpg");
        form.append("mode", mode);

        result.textContent = "Envoi…";

        const res = await fetch("/upload", {
            method: "POST",
            body: form
        });

        const json = await res.json();

        console.log("SERVER SIG", json.signature);

        drawRects(json.rects);

        if (json.rois && json.rects && json.rects.length > 0) {

            const r = json.rects[0];
            drawRois(json.rois, r);

        }


        // =========================
        // MATCH
        // =========================

        if (json.signature && json.rois && json.rects.length > 0) {

            const rect = json.rects[0];

       let detectedColor = null;

if (json.signature &&
    json.signature.color &&
    json.signature.color.color) {

    const c =
        json.signature.color.color;

    const b = c[0];
    const g = c[1];
    const r = c[2];

    if (r > g && r > b)
        detectedColor = "ROUGE";

    else if (g > r && g > b)
        detectedColor = "VERT";

    else if (b > r && b > g)
        detectedColor = "BLEU";

    else
        detectedColor = "JAUNE";

    console.log(
        "COLOR PY =",
        detectedColor,
        c
    );
}

            let detectedColor = null;

            if (colorROI) {

                const scaleX = rect.w / 200;
                const scaleY = rect.h / 300;

                const x =
                    rect.x + colorROI.x * scaleX;

                const y =
                    rect.y + colorROI.y * scaleY;

                const w =
                    colorROI.w * scaleX;

                const h =
                    colorROI.h * scaleY;

                const imageData =
                    ctx.getImageData(
                        x,
                        y,
                        w,
                        h
                    );

                detectedColor =
                    detectColor(imageData);

                console.log(
                    "COLOR =",
                    detectedColor
                );
            }

            let cardsFiltered = CARDS;

            if (detectedColor) {

                cardsFiltered =
                    CARDS.filter(
                        c => c.couleur === detectedColor
                    );

            }

            const resultMatch =
                matchSignature(
                    json.signature,
                    cardsFiltered
                );

            console.log("MATCH =", resultMatch);

            if (resultMatch && resultMatch.card) {

                result.textContent =
                    "Carte : " +
                    resultMatch.card.id;

            } else {

                result.textContent =
                    "Pas trouvé";

            }

        } else {

            result.textContent =
                "Pas de signature";

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

            ctx.fillText(
                r.name,
                r.x,
                r.y - 5
            );

            text += r.name + "\n";

        }

    });

    if (text) result.textContent = text;

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
