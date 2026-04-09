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
// SEND TO PYTHON
// =========================

function sendToPython() {
    canvas.toBlob(async blob => {
        const form = new FormData();

        form.append("image", blob, "capture.jpg");
        form.append("mode", mode);

        resultEl.textContent = "Analyse en cours par le serveur...";

        try {
            const res = await fetch("/upload", {
                method: "POST",
                body: form
            });

            const json = await res.json();
            console.log("UPLOAD JSON =", json);

            if (json.error) {
                resultEl.textContent = "Erreur : " + json.error;
                return;
            }

            // Dessin des rectangles de détection
            drawRects(json.rects || []);

            // Dessin des zones d'intérêt (ROI) si une carte unique est détectée
            if (mode !== "BOARD" && json.rois && json.rects && json.rects.length > 0) {
                const r = json.rects[0];
                drawRois(json.rois, r);
            }

            // --- CAS 1 : MODE BOARD ---
            if (mode === "BOARD") {
                const matches = Array.isArray(json.board_matches) ? json.board_matches : [];

                if (!matches.length) {
                    resultEl.textContent = "Aucune carte détectée sur le plateau";
                    return;
                }

                const accepted = matches.filter(m => m.final_card_id && m.final_status === "accepted");
                const proposed = matches.filter(m => m.final_card_id && m.final_status && m.final_status !== "accepted");
                const failed = matches.filter(m => !m.final_card_id);

                const lines = [];
                lines.push("=== MODE : BOARD ===");
                lines.push(`Total : ${matches.length} | OK : ${accepted.length} | ?? : ${proposed.length}`);
                lines.push("");

                matches.forEach((m, idx) => {
                    const num = idx + 1;
                    if (m.final_card_id) {
                        const statusIcon = m.final_status === "accepted" ? "✅" : "⚠️";
                        lines.push(`${num}. ${statusIcon} ${m.final_card_id} (${m.final_status})`);
                        lines.push(`   - ${m.color_name || "?"} | ${m.symbol_name || "?"} | pts:${m.points ?? "?"}`);
                    } else {
                        lines.push(`${num}. ❌ Non reconnue`);
                    }
                });

                resultEl.textContent = lines.join("\n");
                return;
            }

            // --- CAS 2 : MODE CARDS_ONLY ---
            if (json.signature) {
                const finalCardId = json.final_card_id || "Inconnue";
                const finalStatus = json.final_status || "rejected";
                const finalScore = Number(json.final_score ?? 0);
                const finalGap = Number(json.final_gap ?? 0);

                const color = json.color_name || json.signature?.color?.detected || "??";
                const symbol = json.symbol_name || json.signature?.symbol?.name || "??";
                const points = json.points ?? json.signature?.points?.digit ?? "??";
                const layout = json.bottom_layout || json.signature?.bottom_layout?.layout || "??";

                let message = "";
                if (finalStatus === "accepted") {
                    message += `✅ CARTE CONFIRMÉE : ${finalCardId}\n`;
                } else {
                    message += `⚠️ CARTE PROPOSÉE : ${finalCardId}\n`;
                }

                message += `Statut : ${finalStatus}\n`;
                message += `---------------------------\n`;
                message += `Couleur  : ${color}\n`;
                message += `Symbole  : ${symbol}\n`;
                message += `Points   : ${points}\n`;
                message += `Layout   : ${layout}\n`;
                message += `Score    : ${finalScore.toFixed(3)}\n`;
                message += `Gap      : ${finalGap.toFixed(3)}`;

                resultEl.textContent = message;
            } else {
                resultEl.textContent = "Pas de signature détectée par le serveur.";
            }

        } catch (err) {
            console.error(err);
            resultEl.textContent = "Erreur lors de la communication avec le serveur.";
        }
    }, "image/jpeg");
}


// =========================
// DRAW RECT
// =========================

function drawRects(rects) {
    ctx.lineWidth = 3;
    ctx.font = "20px Arial";

    rects.forEach(r => {
        ctx.strokeStyle = (r.type === "STATION") ? "red" : "yellow";
        ctx.strokeRect(r.x, r.y, r.w, r.h);

        if (r.name) {
            ctx.fillStyle = "lime";
            ctx.fillText(r.name, r.x, r.y - 5);
        }
    });
}


// =========================
// DRAW ROI
// =========================

function drawRois(rois, rect) {
    const scaleX = rect.w / 200;
    const scaleY = rect.h / 300;

    rois.forEach(r => {
        if (r.type === "GLOBAL") return;

        switch(r.type) {
            case "COLOR": ctx.strokeStyle = "red"; break;
            case "SYMBOL": ctx.strokeStyle = "blue"; break;
            case "BOTTOM": ctx.strokeStyle = "violet"; break;
            case "POINTS_BADGE": ctx.strokeStyle = "white"; break;
            default: ctx.strokeStyle = "cyan";
        }

        ctx.lineWidth = 2;
        ctx.strokeRect(
            rect.x + r.x * scaleX,
            rect.y + r.y * scaleY,
            r.w * scaleX,
            r.h * scaleY
        );
    });
}
