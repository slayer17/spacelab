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

    const boardMatches = Array.isArray(json.board_analysis?.board_matches)
        ? json.board_analysis.board_matches
        : Array.isArray(json.board_matches)
            ? json.board_matches
            : Array.isArray(json.board_analysis?.slots)
                ? json.board_analysis.slots
                : [];

    if (boardMatches.length > 0) {
        const lines = [];
        lines.push("=== MODE : BOARD ===");

        boardMatches.forEach((item, index) => {
            const slot =
                item.slot_id ||
                item.slot_label ||
                item.slot ||
                item.band ||
                `slot_${index + 1}`;

            const card =
                item.final_card_id ||
                item.card_id ||
                item.label ||
                "carte détectée";

            const status =
                item.final_status ||
                (item.final_card_id ? "accepted" : "detected");

            const score =
                item.final_score ??
                item.score ??
                item.match_score ??
                "n/a";

            lines.push(`${index + 1}. ${slot} : ${card} (${status}) score=${score}`);
        });

        return lines.join("\n");
    }

    if (json.board_analysis || json.board_matches) {
        return "Aucune carte détectée sur le plateau.";
    }

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
    const boardAnalysis = json.board_analysis || {};
    const matches = Array.isArray(boardAnalysis.board_matches)
        ? boardAnalysis.board_matches
        : Array.isArray(json.board_matches)
            ? json.board_matches
            : [];
    const slots = Array.isArray(boardAnalysis.slots) ? boardAnalysis.slots : [];

    if (matches.length > 0) {
        const lines = [];
        lines.push("=== MODE : BOARD ===");
        lines.push(`Cartes reconnues : ${matches.length}`);
        lines.push("");

        matches.forEach((m, idx) => {
            const slotName =
                m.slot_id ||
                m.slot ||
                m.band ||
                `slot_${idx + 1}`;

            const cardName =
                m.final_card_id ||
                m.card_id ||
                "inconnue";

            const status =
                m.final_status ||
                "detected";

            const score =
                m.final_score ?? m.score ?? "n/a";

            lines.push(`${idx + 1}. ${slotName} : ${cardName} (${status}) score=${score}`);
        });

        resultEl.textContent = lines.join("\n");
        lastResultText = resultEl.textContent;
        return;
    }

    if (slots.length > 0) {
        const lines = [];
        lines.push("=== MODE : BOARD ===");
        lines.push(`Zones détectées : ${slots.length}`);
        lines.push("Aucune carte reconnue pour le moment.");
        lines.push("");
        lines.push("Détail des zones :");

        slots.forEach((s, idx) => {
            const slotName =
                s.slot_id ||
                s.slot ||
                s.band ||
                `slot_${idx + 1}`;

            const status = s.status || "unknown";
            lines.push(`${idx + 1}. ${slotName} : ${status}`);
        });

        resultEl.textContent = lines.join("\n");
        lastResultText = resultEl.textContent;
        return;
    }

    resultEl.textContent = "Aucune zone détectée sur le plateau";
    lastResultText = resultEl.textContent;
    return;
}           
			
			
			
			
// --- CAS 2 : MODE CARDS_ONLY ---
if (json.signature) {
    let frontMatch = null;

    try {
        if (typeof matchSignature === "function" && Array.isArray(CARDS)) {
            frontMatch = matchSignature(json.signature, CARDS);
            console.log("FRONT MATCH =", frontMatch);
        }
    } catch (e) {
        console.error("matchSignature ERROR", e);
    }

    const color = json.signature?.color?.detected || "??";
    const symbol = json.signature?.symbol?.name || json.signature?.symbol?.raw_name || "??";
    const points =
        json.signature?.bottom_layout?.points ??
        json.signature?.points?.digit ??
        "??";
    const layout = json.signature?.bottom_layout?.layout || "??";

    let message = "";

    if (frontMatch && frontMatch.card) {
        const matchedCard = frontMatch.card.id || "Inconnue";
        const matchedScore = Number(frontMatch.score ?? 0);

        message += `✅ CARTE MATCHÉE : ${matchedCard}\n`;
        message += `Score match JS : ${matchedScore.toFixed(3)}\n`;
        message += `---------------------------\n`;
        message += `Couleur  : ${color}\n`;
        message += `Symbole  : ${symbol}\n`;
        message += `Points   : ${points}\n`;
        message += `Layout   : ${layout}`;
    } else {
        message += `⚠️ AUCUN MATCH FINAL JS\n`;
        message += `---------------------------\n`;
        message += `Couleur  : ${color}\n`;
        message += `Symbole  : ${symbol}\n`;
        message += `Points   : ${points}\n`;
        message += `Layout   : ${layout}`;
    }

    resultEl.textContent = message;
    lastResultText = resultEl.textContent;

    // On stocke aussi le match front dans le JSON exporté
    if (!window.lastUploadJson) {
        window.lastUploadJson = json;
    }
    window.lastUploadJson.front_match = frontMatch || null;

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
