function toArray(v) {
    if (Array.isArray(v)) return v;
    if (v == null) return [];
    if (typeof v === "number") return [v];
    if (typeof v === "object") return Object.values(v).flatMap(toArray);
    return [];
}

function euclideanDistance(a, b) {
    const aa = toArray(a);
    const bb = toArray(b);
    const n = Math.min(aa.length, bb.length);

    if (n === 0) return Infinity;

    let sum = 0;
    for (let i = 0; i < n; i++) {
        const da = Number(aa[i]) || 0;
        const db = Number(bb[i]) || 0;
        const d = da - db;
        sum += d * d;
    }

    return Math.sqrt(sum / n);
}

function similarityScore(a, b) {
    const dist = euclideanDistance(a, b);
    if (!isFinite(dist)) return 0;
    return 1 / (1 + dist);
}

function meanStdScore(a, b) {

    if (!a || !b) return 0;

    if (a.mean == null || b.mean == null) {
        return similarityScore(a, b);
    }

    const dMean = Math.abs(a.mean - b.mean);
    const dStd = Math.abs(a.std - b.std);

    const max = 255;

    const score = 1 - (dMean + dStd) / (2 * max);

    return Math.max(0, score);
}

function getScanPart(signature, part) {
    if (!signature) return null;

    if (signature.scan && signature.scan[part] != null)
        return signature.scan[part];

    if (signature[part] != null)
        return signature[part];

    return null;
}

function enrichCandidate(querySig, card) {

    const cardSig = card.signature || card;

 const qColor = getScanPart(querySig, "color")?.color;
const cColor = getScanPart(cardSig, "color")?.color;

let colorScore = 0;

if (qColor && cColor) {

    const d =
        Math.abs(qColor[0] - cColor[0]) +
        Math.abs(qColor[1] - cColor[1]) +
        Math.abs(qColor[2] - cColor[2]);

    colorScore = 1 / (1 + d / 50);

}

const symbolScore = meanStdScore(
    getScanPart(querySig, "symbol"),
    getScanPart(cardSig, "symbol")
);

const bottomScore = meanStdScore(
    getScanPart(querySig, "bottom"),
    getScanPart(cardSig, "bottom")
);

const globalScore = meanStdScore(
    getScanPart(querySig, "global"),
    getScanPart(cardSig, "global")
);

    return {
        card,
        colorScore,
        symbolScore,
        bottomScore,
        globalScore
    };
}

function keepBestBy(candidates, key, options = {}) {

    const {
        keepTop = 8,
        ratio = 0.92,
        minKeep = 2
    } = options;

    if (!candidates.length) return [];

    const sorted = [...candidates].sort((a, b) => b[key] - a[key]);

    const best = sorted[0][key];

    let kept = sorted.filter(c => c[key] >= best * ratio);

    if (kept.length < minKeep) {
        kept = sorted.slice(0, Math.min(keepTop, sorted.length));
    } else {
        kept = kept.slice(0, keepTop);
    }

    return kept;
}

function finalTieBreak(candidates) {

    if (!candidates.length) return null;

    const scored = candidates.map(c => ({

        ...c,

        finalScore:
            (c.colorScore * 0.10) +
            (c.symbolScore * 0.50) +
            (c.bottomScore * 0.30) +
            (c.globalScore * 0.10)

    }));

    scored.sort((a, b) => b.finalScore - a.finalScore);

    return scored[0];
}


function matchSignature(querySig, cardsDb) {

    if (!querySig || !cardsDb || !cardsDb.length) {

        return {
            card: cardsDb ? cardsDb[0] : null,
            score: 0
        };
    }

    let candidates = cardsDb.map(card =>
        enrichCandidate(querySig, card)
    );

    // -----------------
    // COLOR
    // -----------------

    let stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });
// -----------------
// SYMBOL NAME FROM SCAN
// -----------------

let detectedSymbol =
    getScanPart(querySig, "symbol")?.name || null;

let stepSymbol = stepColor;

if (detectedSymbol) {

    stepSymbol = stepColor.filter(c => {

        if (!c.card.symbol) return true;

        return (
            c.card.symbol &&
            detectedSymbol &&
            c.card.symbol.toUpperCase().trim() ===
            detectedSymbol.toUpperCase().trim()
        );

    });

}

// fallback si rien trouvé
if (!stepSymbol.length) {

    console.log("SYMBOL FILTER FAILED → fallback");

    stepSymbol = keepBestBy(stepColor, "symbolScore", {
        keepTop: 3,
        ratio: 0.97,
        minKeep: 1
    });

}


    // -----------------
    // BOTTOM
    // -----------------

    let stepBottom = keepBestBy(stepSymbol, "bottomScore", {
        keepTop: 2,
        ratio: 0.97,
        minKeep: 1
    });

    let best = finalTieBreak(stepBottom);

    // -----------------
    // fallback
    // -----------------

    if (!best) {

        const fallback = [...candidates]
            .map(c => ({
                ...c,
                finalScore:
                    (c.colorScore * 0.25) +
                    (c.symbolScore * 0.25) +
                    (c.bottomScore * 0.25) +
                    (c.globalScore * 0.25)
            }))
            .sort((a, b) => b.finalScore - a.finalScore);

        best = fallback[0];
    }

    if (!best || !best.card) {

        return {
            card: cardsDb[0],
            score: 0
        };
    }
return {

    card: best.card,

    score:
        best.finalScore ??
        (
            best.colorScore * 0.15 +
            best.symbolScore * 0.35 +
            best.bottomScore * 0.40 +
            best.globalScore * 0.10
        ),

    debug: {
        colorScore: best.colorScore,
        symbolScore: best.symbolScore,
        bottomScore: best.bottomScore,
        globalScore: best.globalScore
    }

};
}

