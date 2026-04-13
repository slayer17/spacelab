console.log("CARDS =", CARDS);

let lastImageDataUrl = null;
let lastAnalysisJson = null;
let lastResultText = "";

let mode = "BOARD";
window.lastUploadJson = null;

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

const btnShowJson = document.getElementById("btnShowJson");
const btnCopyJson = document.getElementById("btnCopyJson");
const jsonOutput = document.getElementById("jsonOutput");

// bouton export
const exportReportBtn = document.getElementById("exportReportBtn");

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

    // mémorise l'image affichée
    lastImageDataUrl = canvas.toDataURL("image/jpeg", 0.92);

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

        // mémorise l'image affichée
        lastImageDataUrl = canvas.toDataURL("image/jpeg", 0.92);

        sendToPython();
    };

    img.src = URL.createObjectURL(file);
};


// =========================
// EXPORT HTML + JSON
// =========================

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportHtmlAndJson() {
    if (!lastAnalysisJson) {
        alert("Aucun JSON à exporter.");
        return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const jsonPretty = JSON.stringify(lastAnalysisJson, null, 2);
    const safeResult = escapeHtml(lastResultText || "Aucun résultat.");

    const imageBlock = lastImageDataUrl
        ? `<img src="${lastImageDataUrl}" alt="image analysée" style="max-width:100%; border:1px solid #ccc; border-radius:8px;">`
        : `<p style="color:#888;">Image non disponible.</p>`;

    const htmlContent = `<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport SpaceLab</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 24px;
            background: #f7f7f7;
            color: #222;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        h1, h2 {
            margin-top: 0;
        }
        pre {
            background: #111;
            color: #eee;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .result {
            white-space: pre-wrap;
            font-size: 16px;
            line-height: 1.5;
        }
        .meta {
            color: #666;
            font-size: 14px;
            margin-bottom: 12px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Rapport d'analyse SpaceLab</h1>
        <div class="meta">Généré le ${new Date().toLocaleString("fr-FR")}</div>
    </div>

    <div class="card">
        <h2>Image analysée</h2>
        ${imageBlock}
    </div>

    <div class="card">
        <h2>Résultat</h2>
        <div class="result">${safeResult}</div>
    </div>

    <div class="card">
        <h2>JSON complet</h2>
        <pre>${escapeHtml(jsonPretty)}</pre>
    </div>
</body>
</html>`;

    downloadFile(
        `spacelab_report_${timestamp}.html`,
        htmlContent,
        "text/html;charset=utf-8"
    );

    downloadFile(
        `spacelab_report_${timestamp}.json`,
        jsonPretty,
        "application/json;charset=utf-8"
    );
}

function buildReadableResult(json) {
    if (!json) return "Aucun résultat.";

    // MODE BOARD
    const boardMatches = Array.isArray(json.board_analysis?.board_matches)
        ? json.board_analysis.board_matches
        : Array.isArray(json.board_matches)
            ? json.board_matches
            : [];

    if (boardMatches.length > 0) {
        const lines = [];
        lines.push("=== MODE : BOARD ===");

        boardMatches.forEach((item, index) => {
            const slot = item.slot_id || item.slot_label || item.slot || `slot_${index + 1}`;
            const card = item.final_card_id || item.card_id || "inconnue";
            const status = item.final_status || "unknown";
            const score = item.final_score ?? item.score ?? "n/a";
            lines.push(`${index + 1}. ${slot} : ${card} (${status}) score=${score}`);
        });

        return lines.join("\n");
    }

    // cas où le backend renvoie bien une structure board mais vide
    if (json.board_analysis || json.board_matches) {
        return "Aucune carte détectée sur le plateau.";
    }

    // MODE carte unique
    if (json.final_card_id) {
        return `Carte détectée : ${json.final_card_id} (score: ${json.final_score ?? "n/a"}, statut: ${json.final_status ?? "unknown"})`;
    }

    if (json.signature) {
        const color = json.color_name || json.signature?.color?.detected || "??";
        const symbol = json.symbol_name || json.signature?.symbol?.name || "??";
        const points = json.points ?? json.signature?.points?.digit ?? "??";
        return `Carte détectée : couleur=${color}, symbole=${symbol}, points=${points}`;
    }

    return "Résultat disponible dans le JSON.";
}

// =========================
// JSON TOOLS
// =========================

function showLastJson() {
    if (!jsonOutput) return;

    jsonOutput.style.display = "block";

    if (!window.lastUploadJson) {
        jsonOutput.textContent = "Aucun JSON disponible pour le moment.";
        return;
    }

    jsonOutput.textContent = JSON.stringify(window.lastUploadJson, null, 2);
}

async function copyLastJson() {
    if (!window.lastUploadJson) {
        alert("Aucun JSON disponible pour le moment.");
        return;
    }

    try {
        await navigator.clipboard.writeText(
            JSON.stringify(window.lastUploadJson, null, 2)
        );
        alert("JSON copié.");
    } catch (err) {
        console.error(err);
        alert("Impossible de copier le JSON.");
    }
}

if (btnShowJson) {
    btnShowJson.addEventListener("click", showLastJson);
}

if (btnCopyJson) {
    btnCopyJson.addEventListener("click", copyLastJson);
}

if (exportReportBtn) {
    exportReportBtn.addEventListener("click", exportHtmlAndJson);
}


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
            window.lastUploadJson = json;

            // mémorise pour export
            lastAnalysisJson = json;
            lastResultText = buildReadableResult(json);

            // si l'image n'a pas encore été mémorisée
            if (!lastImageDataUrl) {
                lastImageDataUrl = canvas.toDataURL("image/jpeg", 0.92);
            }

            console.log("UPLOAD JSON =", json);

            if (json.error) {
                resultEl.textContent = "Erreur : " + json.error;
                return;
            }

            // redessine les rectangles de détection
            drawRects(json.rects || []);

            // zones ROI si carte unique
            if (mode !== "BOARD" && json.rois && json.rects && json.rects.length > 0) {
                const r = json.rects[0];
                drawRois(json.rois, r);
            }

            // --- CAS 1 : MODE BOARD ---
            if (mode === "BOARD") {
                const matches = Array.isArray(json.board_analysis?.board_matches)
    ? json.board_analysis.board_matches
    : [];

                if (!matches.length) {
                    resultEl.textContent = "Aucune carte détectée sur le plateau";
                    lastResultText = resultEl.textContent;
                    return;
                }

                const accepted = matches.filter(m => m.final_card_id && m.final_status === "accepted");
                const proposed = matches.filter(m => m.final_card_id && m.final_status && m.final_status !== "accepted");
                const failed = matches.filter(m => !m.final_card_id);

                const lines = [];
                lines.push("=== MODE : BOARD ===");
                lines.push(`Total : ${matches.length} | OK : ${accepted.length} | ?? : ${proposed.length} | KO : ${failed.length}`);
                lines.push("");

                matches.forEach((m, idx) => {
                    const num = idx + 1;
                    if (m.final_card_id) {
                        const statusIcon = m.final_status === "accepted" ? "✅" : "⚠️";
                        lines.push(`${num}. ${statusIcon} ${m.final_card_id} (${m.final_status})`);
                        lines.push(`   - slot: ${m.slot_id || "?"}`);
                        lines.push(`   - ${m.color_name || "?"} | ${m.symbol_name || "?"} | pts:${m.points ?? "?"}`);
                        lines.push(`   - score:${Number(m.final_score ?? 0).toFixed(3)} | gap:${Number(m.final_gap ?? 0).toFixed(3)}`);
                    } else {
                        lines.push(`${num}. ❌ Non reconnue`);
                        lines.push(`   - slot: ${m.slot_id || "?"}`);
                    }
                });

                resultEl.textContent = lines.join("\n");
                lastResultText = resultEl.textContent;
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
                lastResultText = resultEl.textContent;
            } else {
                resultEl.textContent = "Pas de signature détectée par le serveur.";
                lastResultText = resultEl.textContent;
            }

        } catch (err) {
            console.error(err);
            resultEl.textContent = "Erreur lors de la communication avec le serveur.";
            lastResultText = resultEl.textContent;
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

        switch (r.type) {
            case "COLOR":
                ctx.strokeStyle = "red";
                break;
            case "SYMBOL":
                ctx.strokeStyle = "blue";
                break;
            case "BOTTOM":
                ctx.strokeStyle = "violet";
                break;
            case "POINTS_BADGE":
                ctx.strokeStyle = "white";
                break;
            default:
                ctx.strokeStyle = "cyan";
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
