console.log("CARDS =",CARDS);
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

if (json.rois) {
    drawRois(json.rois);
}

   if (json.signature) {

    const card = matchSignature(json.signature);

    if (card) {

        result.textContent =
            "Carte : " + card.id;

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
// DRAW
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

//==========================
// DRAW ROI
//==========================
function drawRois(rois) {

    ctx.lineWidth = 2;

    rois.forEach(r => {

        if (r.type === "COLOR")
            ctx.strokeStyle = "red";

        else if (r.type === "SYMBOL")
            ctx.strokeStyle = "blue";

        else if (r.type === "BOTTOM")
            ctx.strokeStyle = "violet";

        else if (r.type === "GLOBAL")
            ctx.strokeStyle = "green";

        else
            ctx.strokeStyle = "white";

        ctx.strokeRect(
            r.x,
            r.y,
            r.w,
            r.h
        );

    });

}

// =========================
// MATCH SIGNATURE
// =========================

function distance(a, b) {
    return Math.abs(a - b);
}

function colorVectorDistance(a, b) {
    if (!a || !b || a.length !== 3 || b.length !== 3) return 0;

    return Math.sqrt(
        Math.pow((a[0] || 0) - (b[0] || 0), 2) +
        Math.pow((a[1] || 0) - (b[1] || 0), 2) +
        Math.pow((a[2] || 0) - (b[2] || 0), 2)
    );
}

function zoneDistance(a, b) {

    if (!a || !b) return 999999;

    let d = 0;

    // mean
    if (a.mean !== undefined && b.mean !== undefined) {
        d += distance(a.mean, b.mean);
    }

    // std
    if (a.std !== undefined && b.std !== undefined) {
        d += distance(a.std, b.std);
    }

    // color vector
    if (a.color && b.color) {
        d += colorVectorDistance(a.color, b.color) * 0.2;
    }

    return d;
}
/*-------------------------------
fonction pour match des signatures
------------------------------------*/

function matchSignature(sig) {

    if (!sig) return null;

    let best = null;
    let bestScore = 999999;

    let detectedColor = null;

    if (sig.color && sig.color.color) {

        const c = sig.color.color; // [b,g,r]

        if (c[2] > c[1] && c[2] > c[0]) detectedColor = "ROUGE";
        else if (c[0] > c[1] && c[0] > c[2]) detectedColor = "BLEU";
        else if (c[1] > c[2] && c[1] > c[0]) detectedColor = "VERT";
        else detectedColor = "JAUNE";

    }

    for (let c of CARDS) {

        if (!c.signature || !c.signature.scan) continue;

        if (detectedColor && c.couleur !== detectedColor) continue;

        const s = c.signature.scan;

        if (!s.global) continue;

        const dGlobal = zoneDistance(sig.global, s.global);
        const dBottom = zoneDistance(sig.bottom, s.bottom);
        const dColor = zoneDistance(sig.color, s.color);

        const score =
            dBottom * 0.5 +
            dColor * 0.4 +
            dGlobal * 0.1;

        if (score < bestScore) {

            bestScore = score;
            best = c;

        }

    }

    return best;
}
