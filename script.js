const fileInput = document.getElementById("file");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");


fileInput.addEventListener("change", e => {

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

});


async function sendToPython() {

    canvas.toBlob(async blob => {

        const form = new FormData();

        form.append("image", blob, "capture.jpg");

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

            }

        }

    }, "image/jpeg");

}


function drawRects(rects) {

    ctx.lineWidth = 3;

    rects.forEach(r => {

        ctx.strokeStyle = "yellow";

        ctx.strokeRect(
            r.x,
            r.y,
            r.w,
            r.h
        );

    });

}


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
