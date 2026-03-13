(() => {

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
let mode = "BOARD";
result.textContent = "Mode BOARD";


/* =========================
   CAMERA
========================= */

async function startCamera() {

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

}



/* =========================
   CAPTURE
========================= */

function takePhoto() {

    if (!currentStream) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    sendToPython();

}


if (boardBtn) {
    boardBtn.addEventListener("click", () => {

        mode = "BOARD";
        console.log("MODE =", mode);

        result.textContent = "Mode BOARD";

    });
}

if (cardsBtn) {
    cardsBtn.addEventListener("click", () => {

        mode = "CARDS_ONLY";
        console.log("MODE =", mode);

        result.textContent = "Mode CARDS_ONLY";

    });
}
/* =========================
   LOAD IMAGE
========================= */

function loadImage(file) {

    const img = new Image();

    img.onload = () => {

        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        sendToPython();
    };

    img.src = URL.createObjectURL(file);

}



/* =========================
   SEND TO PYTHON
========================= */

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

    }, "image/jpeg");

}

/* =========================
   DRAW
========================= */

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

    result.textContent = text;

}



/* =========================
   EVENTS
========================= */

startBtn.addEventListener("click", startCamera);

captureBtn.addEventListener("click", takePhoto);

loadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", e => {

    const file = e.target.files[0];

    if (file) loadImage(file);

});

})();
