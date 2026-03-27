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

function patchScore(a, b) {
    if (!a || !b) return 0;

    const vectorA = a.vector || null;
    const vectorB = b.vector || null;

    const vecScore =
        vectorA && vectorB
            ? similarityScore(vectorA, vectorB)
            : 0;

    const statsScore = meanStdScore(a, b);

    if (vectorA && vectorB) {
        return (vecScore * 0.75) + (statsScore * 0.25);
    }

    return statsScore;
}

function getScanPart(signature, part) {
    if (!signature) return null;

    if (signature.scan && signature.scan[part] != null) {
        return signature.scan[part];
    }

    if (signature[part] != null) {
        return signature[part];
    }

    return null;
}

function getReliableDetectedPoints(querySig) {
    const points = getScanPart(querySig, "points");
    if (!points) return null;
    if (points.digit == null) return null;

    const digit = Number(points.digit);
    if (!isFinite(digit)) return null;

    const score = Number(points.score ?? points.points_score ?? 0);
    const gap = Number(points.gap ?? points.points_gap ?? 0);

    if (score < 0.70) return null;
    if (gap < 0.02) return null;

    return digit;
}

function pointsMatchScore(querySig, card) {
    const detectedPoints = getReliableDetectedPoints(querySig);

    if (detectedPoints == null) {
        return 0.50;
    }

    const cardPoints = Number(card.points);
    if (!isFinite(cardPoints)) {
        return 0.50;
    }

    if (cardPoints === detectedPoints) {
        return 1.00;
    }

    if (Math.abs(cardPoints - detectedPoints) === 1) {
        return 0.15;
    }

    return 0.00;
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

    const pointsScore = pointsMatchScore(querySig, card);

    const bottomScore = patchScore(
        getScanPart(querySig, "bottom"),
        getScanPart(cardSig, "bottom")
    );

    const globalScore = patchScore(
        getScanPart(querySig, "global"),
        getScanPart(cardSig, "global")
    );

    return {
        card,
        colorScore,
        symbolScore,
        pointsScore,
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
            (c.colorScore * 0.20) +
            (c.symbolScore * 0.30) +
            (c.pointsScore * 0.05) +
            (c.bottomScore * 0.30) +
            (c.globalScore * 0.15)
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

    const candidates = cardsDb.map(card => enrichCandidate(querySig, card));

    const stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });

    const detectedSymbol = getScanPart(querySig, "symbol")?.name || null;
    const detectedSymbolScore = Number(getScanPart(querySig, "symbol")?.score || 0);

    let stepSymbol = stepColor;

    if (detectedSymbol && detectedSymbolScore >= 0.55) {
        const filtered = stepColor.filter(c => {
            if (!c.card.symbol) return false;
            return c.card.symbol.toUpperCase().trim() === detectedSymbol.toUpperCase().trim();
        });

        if (filtered.length > 0) {
            stepSymbol = filtered;
            console.log("SYMBOL EXACT =", detectedSymbol);
        } else {
            console.log("SYMBOL FILTER FAILED -> keep color");
        }
    } else {
        console.log("SYMBOL UNKNOWN");
    }

    const stepPoints = stepSymbol.map(c => ({
        ...c,
        boostedBottom: (c.bottomScore * 0.90) + (c.pointsScore * 0.10)
    }));

    const stepBottom = keepBestBy(stepPoints, "boostedBottom", {
        keepTop: 4,
        ratio: 0.97,
        minKeep: 1
    });

    const stepGlobal = keepBestBy(stepBottom, "globalScore", {
        keepTop: 2,
        ratio: 0.97,
        minKeep: 1
    });

    let best = finalTieBreak(stepGlobal);

    if (!best) {
        const fallback = [...candidates]
            .map(c => ({
                ...c,
                finalScore:
                    (c.colorScore * 0.20) +
                    (c.symbolScore * 0.30) +
                    (c.pointsScore * 0.05) +
                    (c.bottomScore * 0.30) +
                    (c.globalScore * 0.15)
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
        score: best.finalScore ?? (
            (best.colorScore * 0.20) +
            (best.symbolScore * 0.30) +
            (best.pointsScore * 0.05) +
            (best.bottomScore * 0.30) +
            (best.globalScore * 0.15)
        ),
        debug: {
            colorScore: best.colorScore,
            symbolScore: best.symbolScore,
            pointsScore: best.pointsScore,
            bottomScore: best.bottomScore,
            globalScore: best.globalScore
        }
    };
}
