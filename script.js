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

        drawRects(json.rects);

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


console.log("SIG", sig);
console.log("CARD", c.id, s);




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



// =========================
// MATCH SIGNATURE
// =========================

function distance(a, b) {
    return Math.abs(a - b);
}

function matchSignature(sig) {

    if (!sig) return null;

    let best = null;
    let bestScore = 999999;

    for (let c of CARDS) {

        if (!c.signature) continue;
        if (!c.signature.scan) continue;

        const s = c.signature.scan.globalSignature;

        if (!s) continue;

        const score =
            distance(sig.mean, s.mean || 0) +
            distance(sig.std, s.std || 0);

        if (score < bestScore) {

            bestScore = score;
            best = c;

        }

    }

    return best;
}
